"""Adaptador de consola para el trabajador web; sin ventanas ni Arduino."""
from queue import Queue

from hardware.exclusion import Exclusivo


class ConsolaRegistro:
    def __init__(self, pedir=input, mostrar=print):
        self.comandos = Queue()
        self.pedir, self.mostrar = pedir, mostrar
        self.cancelado = False
        self.guardado = False
        self.ultimos = {}

    def detener(self):
        return self.cancelado

    def fotograma(self, frame):
        pass  # La consola no crea ventanas ni guarda vistas previas.

    def emitir(self, tipo, valor):
        if tipo == "recorrido":
            clave = (valor["paso"], valor["instruccion"], valor["confirmacion"])
            if self.ultimos.get("paso_consola") != clave:
                self.ultimos["paso_consola"] = clave
                self.mostrar(f"Paso {valor['paso']} de 5. {valor['instruccion']} {valor['confirmacion']}")
            return
        if tipo == "guardado":
            self.guardado = True
        if self.ultimos.get(tipo) == valor:
            return
        self.ultimos[tipo] = valor
        if tipo == "progreso":
            self.mostrar(f"Capturas validas: {valor}/5")
        elif tipo in ("instruccion", "aviso_registro", "guardado"):
            self.mostrar(str(valor))


def registrar_terminal(nombre, cam_idx=0, pedir=input, mostrar=print, prueba=False):
    from configuracion.config import DB_PATH, FOTOS_DIR
    from interfaz.duplicados import preparacion_registro
    from interfaz.registro import leer_base, validar_nombre
    from interfaz.trabajador import registrar
    try:
        # Comparte el bloqueo historico con la web, no una interfaz de escritorio.
        with Exclusivo("escritorio"):
            if not prueba:
                validar_nombre(nombre, leer_base(DB_PATH))
            estado = preparacion_registro(DB_PATH, FOTOS_DIR) if not prueba else {"permitido": True}
            if not estado["permitido"]:
                mostrar(estado["mensaje"])
                return False
            puente = ConsolaRegistro(pedir, mostrar)
            mostrar("Cinco pasos automaticos. Sigue las instrucciones. Ctrl+C cancela. La web permite reintentar un paso.")
            registrar(puente, nombre, cam_idx, prueba=prueba)
            return puente.guardado
    except (OSError, ValueError, RuntimeError) as error:
        mostrar(str(error))
        return False
