"""Servidor HTTP para una red local de confianza, con clave y sesiones."""
from collections import defaultdict, deque
import hashlib
import hmac
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit

STATIC = Path(__file__).resolve().parent / "static"
IMAGENES = frozenset((
    "panel-control-blanco", "persona-blanca", "gesto-mano-blanco", "camara-blanca",
    "logo-claud-ia", "cabeza-robotica", "persona", "expresion-facial", "gesto-mano", "camara",
    "panel-control", "modo-demo", "mano-abierta", "puno", "pulgar-arriba",
    "pulgar-abajo", "indice", "dos-dedos", "tres-dedos", "cuatro-dedos", "ok",
))
ITERACIONES = 600_000


def crear_acceso(clave):
    if not isinstance(clave, str) or not 12 <= len(clave) <= 128:
        raise ValueError("Usa una clave de entre 12 y 128 caracteres.")
    sal = secrets.token_hex(16)
    huella = hashlib.pbkdf2_hmac("sha256", clave.encode(), bytes.fromhex(sal), ITERACIONES).hex()
    return {"sal": sal, "huella": huella, "iteraciones": ITERACIONES}


def verificar_clave(clave, acceso):
    if not isinstance(clave, str) or len(clave) > 128:
        return False
    huella = hashlib.pbkdf2_hmac("sha256", clave.encode(), bytes.fromhex(acceso["sal"]), acceso["iteraciones"]).hex()
    return hmac.compare_digest(huella, acceso["huella"])


def direcciones_locales():
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                       if not info[4][0].startswith("127.")})
    except OSError:
        return []


class Servidor(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 24

    def __init__(self, direccion, control, acceso):
        self.control, self.acceso = control, acceso
        self.sesiones = {}
        self.intentos = defaultdict(deque)
        self.seguridad = threading.RLock()
        self.hosts = {"localhost", "127.0.0.1", *direcciones_locales()}
        super().__init__(direccion, Peticion)


class Peticion(BaseHTTPRequestHandler):
    server_version = "CLAUD-IA"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, formato, *args):
        pass  # No registrar claves, cookies ni nombres en el log HTTP.

    def responder(self, codigo, contenido, tipo="application/json; charset=utf-8", cookie=None):
        if isinstance(contenido, dict):
            contenido = json.dumps(contenido, ensure_ascii=False).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(contenido)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'none'")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        try:
            self.wfile.write(contenido)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def host_valido(self):
        try:
            host = urlsplit("//" + self.headers.get("Host", ""))
            permitidos = self.server.hosts | {self.connection.getsockname()[0]}
            return host.hostname in permitidos and host.port == self.server.server_port and not host.username
        except ValueError:
            return False

    def sesion(self):
        try:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            identificador = cookie["claudia_sesion"].value
        except (KeyError, ValueError, CookieError):
            return None, None
        with self.server.seguridad:
            sesion = self.server.sesiones.get(identificador)
            if sesion and sesion["vence"] > time.monotonic():
                return identificador, sesion
            self.server.sesiones.pop(identificador, None)
        return None, None

    def do_GET(self):
        if not self.host_valido():
            return self.responder(403, {"error": "Dirección no permitida."})
        ruta = urlsplit(self.path).path
        archivos = {"/": ("index.html", "text/html; charset=utf-8"),
                    "/app.css": ("app.css", "text/css; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                    "/marca.svg": ("marca.svg", "image/svg+xml")}
        archivos.update({f'/imagenes/posturas/{n}.png': (f'imagenes/posturas/{n}.png', 'image/png')
                         for n in ('frente', 'derecha', 'izquierda', 'arriba', 'abajo')})
        archivos['/posturas.svg'] = ('posturas.svg', 'image/svg+xml')
        archivos.update({f"/imagenes/{nombre}.png": (f"imagenes/{nombre}.png", "image/png")
                         for nombre in IMAGENES})
        if ruta in archivos:
            archivo, tipo = archivos[ruta]
            return self.responder(200, (STATIC / archivo).read_bytes(), tipo)
        _, sesion = self.sesion()
        if sesion is None:
            return self.responder(401, {"error": "Ingresa la clave para continuar."})
        try:
            if ruta == "/api/estado":
                return self.responder(200, {"estado": self.server.control.ver(), "csrf": sesion["csrf"]})
            if ruta == '/api/guia-gestos':
                from vision.manos import guia_gestos
                return self.responder(200,{'gestos':guia_gestos()})
            if ruta == "/api/personas":
                return self.responder(200, {"personas": self.server.control.personas()})
            if ruta == "/api/registro/preparacion":
                return self.responder(200, self.server.control.preparacion_registro())
            if ruta == "/api/puertos":
                return self.responder(200, {"puertos": self.server.control.puertos()})
            if ruta == "/api/camara.jpg":
                jpeg = self.server.control.foto()
                if jpeg is None:
                    return self.responder(204, b"", "image/jpeg")
                return self.responder(200, jpeg, "image/jpeg")
            return self.responder(404, {"error": "Página no encontrada."})
        except Exception:
            logging.exception("Consulta web fallida")
            return self.responder(500, {"error": "No se pudo consultar la información. Revisa el computador de CLAUD-IA."})

    def do_POST(self):
        if not self.host_valido() or self.headers.get("Origin") != "http://" + self.headers.get("Host", ""):
            return self.responder(403, {"error": "Solicitud no permitida."})
        if self.headers.get_content_type() != "application/json":
            return self.responder(415, {"error": "Formato no permitido."})
        try:
            cantidad = int(self.headers.get("Content-Length", "0"))
            if not 0 < cantidad <= 4096 or self.headers.get("Transfer-Encoding"):
                return self.responder(413, {"error": "Solicitud demasiado grande."})
            datos = json.loads(self.rfile.read(cantidad))
            if not isinstance(datos, dict):
                raise ValueError()
        except (ValueError, UnicodeError):
            return self.responder(400, {"error": "Solicitud no válida."})
        ruta = urlsplit(self.path).path
        if ruta == "/api/entrar":
            return self.entrar(datos)
        identificador, sesion = self.sesion()
        if sesion is None:
            return self.responder(401, {"error": "Ingresa la clave para continuar."})
        if not hmac.compare_digest(self.headers.get("X-CSRF-Token", ""), sesion["csrf"]):
            return self.responder(403, {"error": "Actualiza la página y vuelve a intentarlo."})
        try:
            if ruta == "/api/salir":
                with self.server.seguridad:
                    self.server.sesiones.pop(identificador, None)
                return self.responder(200, {"ok": True}, cookie="claudia_sesion=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
            if ruta == "/api/iniciar":
                modo = datos.get("modo", "demo")
                if modo not in ("demo", "fisico"):
                    raise ValueError("Selecciona Demo o Cabeza física.")
                if modo == "fisico":
                    if datos.get("confirmar_fisico") is not True:
                        raise ValueError("Debes confirmar el arranque físico antes de iniciar.")
                    self.server.control.iniciar("sistema", cam_idx=datos.get("camara", 0), modo=modo,
                        puerto=datos.get("puerto"), confirmar_fisico=True)
                else:
                    self.server.control.iniciar("sistema", cam_idx=datos.get("camara", 0))
            elif ruta == "/api/detener":
                self.server.control.detener()
            elif ruta == "/api/motor":
                self.server.control.seleccionar_motor(datos.get("motor"))
            elif ruta == "/api/opciones":
                self.server.control.opciones(datos.get("opciones"))
            elif ruta == "/api/registrar":
                if not isinstance(datos.get("nombre"), str):
                    raise ValueError("Escribe el nombre de la persona.")
                self.server.control.iniciar("registro", nombre=datos["nombre"], cam_idx=datos.get("camara", 0))
            elif ruta == '/api/evaluar-duplicados':
                self.server.control.iniciar('registro', cam_idx=datos.get('camara', 0), evaluacion=datos.get('evaluacion', {}))
            elif ruta == '/api/iniciar-recorrido':
                self.server.control.iniciar_recorrido(datos.get('nombre'), cam_idx=datos.get('camara', 0))
            elif ruta == '/api/probar-registro':
                self.server.control.iniciar('registro', cam_idx=datos.get('camara', 0), prueba=True)
            elif ruta == '/api/reintentar-registro':
                self.server.control.reintentar()
            elif ruta == '/api/registro/repetir-postura':
                self.server.control.resolver_registro(datos.get('registro_id'))
            elif ruta == '/api/registro/confirmar-nueva':
                self.server.control.resolver_registro(datos.get('registro_id'),manual=True,
                    confirmacion=datos.get('confirmacion'))
            elif ruta == "/api/capturar":
                self.server.control.capturar()
            elif ruta == "/api/eliminar":
                nombre = datos.get("nombre")
                if not isinstance(nombre, str) or datos.get("confirmacion") != nombre:
                    raise ValueError("Confirma el nombre de la persona antes de eliminar.")
                respaldo = self.server.control.eliminar(nombre)
                return self.responder(200, {"ok": True, "respaldo": respaldo})
            else:
                return self.responder(404, {"error": "Acción no encontrada."})
            return self.responder(200, {"ok": True})
        except ValueError as error:
            return self.responder(409, {"error": str(error)})
        except Exception:
            logging.exception("Accion web fallida")
            return self.responder(500, {"error": "No se pudo iniciar la actividad. Comprueba el sistema en el computador de CLAUD-IA."})

    def entrar(self, datos):
        ip, ahora = self.client_address[0], time.monotonic()
        with self.server.seguridad:
            # Limite por direccion y global para no sobrecargar el equipo.
            for clave in (ip, "global"):
                cola = self.server.intentos[clave]
                while cola and cola[0] < ahora-60:
                    cola.popleft()
                if len(cola) >= (8 if clave == ip else 40):
                    return self.responder(429, {"error": "Demasiados intentos. Espera un minuto."})
            self.server.intentos[ip].append(ahora)
            self.server.intentos["global"].append(ahora)
        if not verificar_clave(datos.get("clave"), self.server.acceso):
            return self.responder(401, {"error": "La clave no es correcta."})
        with self.server.seguridad:
            self.server.sesiones = {k: v for k, v in self.server.sesiones.items() if v["vence"] > ahora}
            if len(self.server.sesiones) >= 128:
                return self.responder(429, {"error": "Hay demasiadas sesiones abiertas."})
            identificador = secrets.token_urlsafe(32)
            csrf = secrets.token_urlsafe(32)
            self.server.sesiones[identificador] = {"csrf": csrf, "vence": ahora+8*3600}
        return self.responder(200, {"ok": True, "csrf": csrf},
            cookie=f"claudia_sesion={identificador}; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800")
