import copy
import hashlib
from html.parser import HTMLParser
import http.client
import json
from pathlib import Path
import pickle
from queue import Queue
import re
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from web_local.control import Control
from web_local.servidor import Servidor, crear_acceso, verificar_clave


def trabajador_simulado(accion, parametros, eventos, cuadros, comandos, parada, video_limpio):
    """Proceso real de prueba: nunca importa ni abre hardware."""
    assert video_limpio is True
    assert parametros["demo"] is True
    eventos.put(("conexion", "Demo"))
    eventos.put(("camara", "Activa simulada"))
    eventos.put(("deteccion", {"rostros": 1, "gesto": "puno", "deepface": {"ok": True, "intentado": True}}))
    motor = parametros.get('motor', 'lbph')
    eventos.put(('motor_reconocimiento', {'motor': motor, 'version': 'simulada' if motor=='sface' else None}))
    eventos.put(("reconocimiento", {"motor": motor, "nombre": "Alison",
        "estado_reconocimiento": "aceptado", "similitud_coseno": .6 if motor=='sface' else None,
        "similitud_estimada": .7 if motor=='lbph' else None}))
    cuadros.put(b"jpeg-del-proceso-simulado")
    parada.wait(10)
    eventos.put(("finalizado", accion))


def setUpModule():
    global guardia_camara, guardia_arduino
    guardia_camara = patch("cv2.VideoCapture", side_effect=AssertionError("Camara real prohibida"))
    guardia_arduino = patch("serial.Serial", side_effect=AssertionError("Arduino prohibido"))
    guardia_camara.start()
    guardia_arduino.start()


def tearDownModule():
    guardia_camara.stop()
    guardia_arduino.stop()


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acceso = crear_acceso("clave-laboratorio-2026")

    def setUp(self):
        self.control = MagicMock()
        self.control.ver.return_value = Control.vacio()
        self.control.personas.return_value = [{"nombre": "Alison", "fotos": 30}]
        self.control.foto.return_value = b"jpeg-simulado"
        self.server = Servidor(("127.0.0.1", 0), self.control, self.acceso)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.host = f"127.0.0.1:{self.server.server_port}"
        self.cookie = ""
        self.csrf = ""

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def peticion(self, ruta, datos=None, origen=True, csrf=True, host=None):
        headers = {"Host": host or self.host}
        if self.cookie:
            headers["Cookie"] = self.cookie
        if datos is not None:
            headers["Content-Type"] = "application/json"
            if origen:
                headers["Origin"] = "http://" + self.host
            if csrf:
                headers["X-CSRF-Token"] = self.csrf
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        conn.request("POST" if datos is not None else "GET", ruta,
                     json.dumps(datos) if datos is not None else None, headers)
        respuesta = conn.getresponse()
        codigo, cabeceras, cuerpo = respuesta.status, dict(respuesta.getheaders()), respuesta.read()
        conn.close()
        return codigo, cabeceras, cuerpo

    def entrar(self):
        codigo, headers, body = self.peticion("/api/entrar", {"clave": "clave-laboratorio-2026"})
        self.assertEqual(codigo, 200)
        self.cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.csrf = json.loads(body)["csrf"]
        return headers

    def test_pagina_estatica_y_endpoints_privados(self):
        self.assertEqual(self.peticion("/")[0], 200)
        codigo, headers, cuerpo = self.peticion("/imagenes/cabeza-robotica.png")
        self.assertEqual(codigo, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(cuerpo, (Path(__file__).resolve().parents[1] /
                                  "web_local/static/imagenes/cabeza-robotica.png").read_bytes())
        codigo, headers, cuerpo = self.peticion('/posturas.svg')
        self.assertEqual(codigo, 200)
        self.assertEqual(headers['Content-Type'], 'image/svg+xml')
        for postura in ('frente', 'derecha', 'izquierda', 'arriba', 'abajo'):
            self.assertIn(('id="'+postura+'"').encode(), cuerpo)
        for nombre in ("panel-control-blanco", "persona-blanca", "gesto-mano-blanco", "camara-blanca"):
            with self.subTest(imagen=nombre):
                codigo, headers, cuerpo = self.peticion(f"/imagenes/{nombre}.png")
                self.assertEqual(codigo, 200)
                self.assertEqual(headers["Content-Type"], "image/png")
                self.assertEqual(cuerpo, (Path(__file__).resolve().parents[1] /
                                          f"web_local/static/imagenes/{nombre}.png").read_bytes())
        for ruta in ("/api/camara.jpg", "/api/personas", "/api/estado", "/api/puertos", "/api/registro/preparacion"):
            self.assertEqual(self.peticion(ruta)[0], 401)
        self.assertEqual(self.peticion("/api/iniciar", {})[0], 401)

    def test_guia_autenticada_coincide_con_las_secuencias_actuales(self):
        from vision.manos import guia_gestos,GESTO_ACCION
        self.assertEqual(self.peticion('/api/guia-gestos')[0],401)
        self.entrar()
        codigo,_,cuerpo=self.peticion('/api/guia-gestos')
        self.assertEqual(codigo,200)
        guia=json.loads(cuerpo)['gestos']
        self.assertEqual(guia,guia_gestos())
        self.assertEqual({g['gesto'] for g in guia},set(GESTO_ACCION))
        self.assertTrue(all(g['disponible'] and g['coincide_con_codigo'] for g in guia))
        self.control.iniciar.assert_not_called()

    def test_preparacion_registro_y_bloqueo_antes_de_capturar(self):
        from interfaz.duplicados import FALTA_CONFIGURAR
        self.entrar()
        self.control.preparacion_registro.return_value = {'permitido': False, 'mensaje': FALTA_CONFIGURAR}
        codigo, _, cuerpo = self.peticion('/api/registro/preparacion')
        self.assertEqual(codigo, 200)
        self.assertEqual(json.loads(cuerpo), {'permitido': False, 'mensaje': FALTA_CONFIGURAR})
        self.control.iniciar.side_effect = ValueError(FALTA_CONFIGURAR)
        codigo, _, cuerpo = self.peticion('/api/registrar', {'nombre': 'Otra persona', 'camara': 0})
        self.assertEqual(codigo, 409)
        self.assertEqual(json.loads(cuerpo)['error'], FALTA_CONFIGURAR)
        self.control.capturar.assert_not_called()

    def test_prueba_guiada_sin_nombre_y_reintento_con_sesion_y_csrf(self):
        self.assertEqual(self.peticion('/api/probar-registro', {'camara': 0})[0], 401)
        self.entrar()
        self.assertEqual(self.peticion('/api/probar-registro', {'camara': 0}, csrf=False)[0], 403)
        self.control.iniciar.assert_not_called()
        self.assertEqual(self.peticion('/api/probar-registro', {'camara': 1})[0], 200)
        self.control.iniciar.assert_called_once_with('registro', cam_idx=1, prueba=True)
        self.assertEqual(self.peticion('/api/reintentar-registro', {})[0], 200)
        self.control.reintentar.assert_called_once()

    def test_evaluacion_endpoint_separado_autenticado_y_csrf(self):
        datos={'camara':1,'evaluacion':{'tipo':'registrada','identificador':'Alison','acepta_conservar':True}}
        self.assertEqual(self.peticion('/api/evaluar-duplicados',datos)[0],401)
        self.entrar()
        self.assertEqual(self.peticion('/api/evaluar-duplicados',datos,csrf=False)[0],403)
        self.control.iniciar.assert_not_called()
        self.assertEqual(self.peticion('/api/evaluar-duplicados',datos)[0],200)
        self.control.iniciar.assert_called_once_with('registro',cam_idx=1,evaluacion=datos['evaluacion'])

    def test_inicio_captura_nominal_separado_del_alta(self):
        datos={'nombre':'Persona Nueva','camara':0}
        self.assertEqual(self.peticion('/api/iniciar-recorrido',datos)[0],401)
        self.entrar()
        self.assertEqual(self.peticion('/api/iniciar-recorrido',datos,csrf=False)[0],403)
        self.assertEqual(self.peticion('/api/iniciar-recorrido',datos)[0],200)
        self.control.iniciar_recorrido.assert_called_once_with('Persona Nueva',cam_idx=0)
        self.control.iniciar.assert_not_called()

    def test_revision_y_repeticion_requieren_sesion_y_csrf(self):
        datos={'registro_id':'actual','confirmacion':True}
        for ruta in ('/api/registro/confirmar-nueva','/api/registro/repetir-postura'):
            self.assertEqual(self.peticion(ruta,datos)[0],401)
        self.entrar()
        for ruta in ('/api/registro/confirmar-nueva','/api/registro/repetir-postura'):
            self.assertEqual(self.peticion(ruta,datos,csrf=False)[0],403)
            self.assertEqual(self.peticion(ruta,datos)[0],200)
        self.control.resolver_registro.assert_any_call('actual',manual=True,confirmacion=True)
        self.control.resolver_registro.assert_any_call('actual')

    def test_login_cookie_video_nombres_y_logout(self):
        self.assertEqual(self.peticion("/api/entrar", {"clave": "incorrecta"})[0], 401)
        headers = self.entrar()
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        self.assertEqual(self.peticion("/api/camara.jpg")[2], b"jpeg-simulado")
        self.assertEqual(json.loads(self.peticion("/api/personas")[2])["personas"][0]["nombre"], "Alison")
        self.assertEqual(self.peticion("/api/salir", {})[0], 200)
        self.assertEqual(self.peticion("/api/camara.jpg")[0], 401)

    def test_csrf_origen_host_y_sesiones_vencidas(self):
        self.entrar()
        self.assertEqual(self.peticion("/api/iniciar", {}, csrf=False)[0], 403)
        self.assertEqual(self.peticion("/api/iniciar", {}, origen=False)[0], 403)
        self.assertEqual(self.peticion("/api/estado", host="ejemplo.com:8765")[0], 403)
        self.control.iniciar.assert_not_called()
        for sesion in self.server.sesiones.values():
            sesion["vence"] = 0
        self.assertEqual(self.peticion("/api/estado")[0], 401)

    def test_confirmacion_fisica_y_todas_las_acciones_conectadas(self):
        self.entrar()
        self.assertEqual(self.peticion("/api/iniciar", {"modo": "fisico"})[0], 409)
        self.control.iniciar.assert_not_called()
        self.assertEqual(self.peticion("/api/iniciar", {"modo": "demo", "camara": 1})[0], 200)
        self.control.iniciar.assert_called_with("sistema", cam_idx=1)
        self.assertEqual(self.peticion("/api/iniciar", {"modo": "fisico", "puerto": "COM7", "confirmar_fisico": True})[0], 200)
        self.control.iniciar.assert_called_with("sistema", cam_idx=0, modo="fisico", puerto="COM7", confirmar_fisico=True)
        self.assertEqual(self.peticion("/api/registrar", {"nombre": "Nueva"})[0], 200)
        self.control.iniciar.assert_called_with("registro", nombre="Nueva", cam_idx=0)
        self.assertEqual(self.peticion("/api/capturar", {})[0], 200)
        self.control.capturar.assert_called_once()
        self.assertEqual(self.peticion("/api/opciones", {"opciones": {"emociones": False}})[0], 200)
        self.control.opciones.assert_called_with({"emociones": False})
        self.assertEqual(self.peticion("/api/detener", {})[0], 200)
        self.control.detener.assert_called_once()

    def test_selector_motor_con_sesion_y_csrf(self):
        self.assertEqual(self.peticion('/api/motor', {'motor':'sface'})[0], 401)
        self.entrar()
        self.assertEqual(self.peticion('/api/motor', {'motor':'sface'}, csrf=False)[0], 403)
        self.control.seleccionar_motor.assert_not_called()
        self.assertEqual(self.peticion('/api/motor', {'motor':'sface'})[0], 200)
        self.control.seleccionar_motor.assert_called_once_with('sface')

    def test_limite_intentos_y_no_exposicion_de_archivos(self):
        with patch("web_local.servidor.verificar_clave", return_value=False):
            for _ in range(8):
                self.assertEqual(self.peticion("/api/entrar", {"clave": "x"})[0], 401)
            self.assertEqual(self.peticion("/api/entrar", {"clave": "x"})[0], 429)
        self.server.intentos.clear()
        self.entrar()
        for ruta in ("/web_local/.privado/acceso.json", "/configuracion/rostros_conocidos.pkl", "/../web.py"):
            self.assertEqual(self.peticion(ruta)[0], 404)

    def test_errores_tecnicos_no_llegan_al_usuario(self):
        self.entrar()
        self.control.iniciar.side_effect = RuntimeError("C:/ruta/privada/archivo.json")
        with self.assertLogs(level="ERROR"):
            codigo, _, body = self.peticion("/api/iniciar", {})
        self.assertEqual(codigo, 500)
        self.assertNotIn(b"ruta/privada", body)

    def test_eliminacion_requiere_sesion_csrf_y_confirmacion_exacta(self):
        datos = {"nombre": "Persona prueba", "confirmacion": "Persona prueba"}
        self.assertEqual(self.peticion("/api/eliminar", datos)[0], 401)
        self.entrar()
        self.assertEqual(self.peticion("/api/eliminar", datos, csrf=False)[0], 403)
        self.assertEqual(self.peticion("/api/eliminar", {"nombre": "Persona prueba"})[0], 409)
        self.assertEqual(self.peticion("/api/eliminar", {"nombre": "Persona prueba", "confirmacion": "Otra"})[0], 409)
        self.control.eliminar.assert_not_called()
        self.control.eliminar.return_value = "retiro_prueba"
        self.assertEqual(self.peticion("/api/eliminar", datos)[0], 200)
        self.control.eliminar.assert_called_once_with("Persona prueba")


class EstadoTests(unittest.TestCase):
    def test_proceso_real_simulado_inicio_video_parada_y_reinicio(self):
        control = Control()
        with patch("web_local.control.ejecutar_trabajador", trabajador_simulado):
            for _ in range(2):
                control.iniciar("sistema")
                try:
                    limite = time.monotonic()+7
                    while time.monotonic() < limite and control.foto() is None:
                        time.sleep(.03)
                    self.assertEqual(control.ver()["fase"], "activo")
                    self.assertEqual(control.foto(), b"jpeg-del-proceso-simulado")
                    self.assertEqual(control.ver()["nombre"], "Alison")
                    self.assertEqual(control.ver()['motor_seleccionado'], 'sface')
                    self.assertEqual(control.ver()['motor_activo'], 'sface')
                    self.assertEqual(control.ver()['similitud_coseno'], .6)
                    self.assertIsNone(control.ver()['similitud_estimada'])
                finally:
                    control.cerrar()
                self.assertIsNone(control.proceso)
                self.assertIsNone(control.foto())
                self.assertEqual(control.ver()["fase"], "detenido")

    def test_rechaza_actividades_simultaneas_y_opciones_invalidas(self):
        control = Control()
        control.proceso = MagicMock()
        with self.assertRaises(ValueError):
            control.iniciar("registro", "Nueva")
        with self.assertRaises(ValueError):
            control.opciones({"gestos": "false"})
        control.parada = threading.Event()
        control.detener()
        self.assertTrue(control.parada.is_set())
        control.proceso.terminate.assert_not_called()

    def test_captura_una_a_la_vez_y_estados_sin_rutas(self):
        control = Control()
        control.estado.update(actividad="registro", camara=True, fase="activo")
        control.comandos = Queue()
        with self.assertRaises(ValueError):
            control.capturar()
        control.proceso = MagicMock()
        control.estado['recorrido'] = {'pausado': True}
        control.reintentar()
        self.assertEqual(control.comandos.get_nowait(), ('reintentar', None))
        with self.assertRaises(ValueError):
            control.reintentar()
        self.assertFalse(control.ver()["captura_lista"])
        control.evento("secuencia", {"disponible": False, "archivo": "ruta/interna.json"})
        self.assertNotIn("interna", json.dumps(control.ver()))
        control.evento("deteccion", {"rostros": 1, "gesto": "puno", "deepface": {"ok": True, "intentado": True}})
        control.evento("expresion", {"nombre": "Feliz", "confianza": .8})
        control.evento("reconocimiento", {"nombre": "Alison"})
        self.assertEqual(control.ver()["expresion"], "Feliz")
        control.opciones({"emociones": False, "reconocimiento": False})
        self.assertIsNone(control.ver()["expresion"])
        self.assertIsNone(control.ver()["nombre"])

    def test_cero_rostros_limpia_lecturas_y_foto_caduca(self):
        control = Control()
        control.estado.update(nombre="Alison", expresion="Feliz", camara=True)
        control.evento("deteccion", {"rostros": 0, "gesto": None, "deepface": {"ok": False, "intentado": False}})
        self.assertIsNone(control.ver()["nombre"])
        control.jpeg = b"foto"
        control.ultimo_frame = time.monotonic()
        self.assertEqual(control.foto(), b"foto")
        control.ultimo_frame -= 10
        self.assertIsNone(control.foto())

    def test_base_existente_y_nombre_duplicado_intactos(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d)/"base.pkl"
            db.write_bytes(pickle.dumps({"nombres": ["Alison"], "imagenes": ["foto.jpg"]}))
            antes = db.read_bytes()
            control = Control(db)
            self.assertEqual(control.personas(), [{"nombre": "Alison", "fotos": 1}])
            with self.assertRaises(ValueError):
                control.iniciar("registro", " ALISON ")
            self.assertEqual(db.read_bytes(), antes)
            self.assertIsNone(control.proceso)

    def test_clave_con_sal_y_sin_texto_plano(self):
        acceso = crear_acceso("clave-laboratorio-2026")
        self.assertNotIn("clave-laboratorio", json.dumps(acceso))
        self.assertTrue(verificar_clave("clave-laboratorio-2026", acceso))
        self.assertFalse(verificar_clave("otra", acceso))
        with self.assertRaises(ValueError):
            crear_acceso("corta")

    def test_referencias_dom_y_sin_camara_de_visitante(self):
        static = Path(__file__).resolve().parents[1]/"web_local/static"
        html = (static/"index.html").read_text(encoding="utf-8")
        js = (static/"app.js").read_text(encoding="utf-8")
        class IDs(HTMLParser):
            ids = []
            def handle_starttag(self, tag, attrs):
                for k, v in attrs:
                    if k == "id": self.ids.append(v)
        parser = IDs()
        parser.feed(html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        for identificador in re.findall(r'\$\("([^"+]+)"\)', js):
            self.assertIn(identificador, parser.ids)
        self.assertNotIn("getUserMedia", js)
        self.assertNotIn("innerHTML", js)
        self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
