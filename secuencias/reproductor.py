"""Propuestas pendientes de validacion fisica; reproduccion exclusiva."""
import json
from pathlib import Path
import threading
import time

from propuestas_movimientos.preparar import CARPETA, validar
from vision.manos import GESTO_ACCION

NOMBRES = frozenset(nombre for nombre, _ in GESTO_ACCION.values())


def resolver_json(nombre):
    nombre = nombre if nombre.endswith(".json") else nombre + ".json"
    ruta = CARPETA / nombre
    return str(ruta.resolve()) if nombre in NOMBRES and ruta.is_file() else None


def cargar(nombre):
    ruta = resolver_json(nombre)
    if ruta is None:
        raise ValueError(f"Propuesta no encontrada: {nombre} en {CARPETA}")
    try:
        bloques = json.loads(Path(ruta).read_text(encoding="utf-8"))
        validar(bloques)
    except (ValueError, OSError, TypeError) as exc:
        raise ValueError(f"Propuesta invalida {nombre}: {exc}") from exc
    return sorted(bloques, key=lambda b: b[2])


class Reproductor:
    def __init__(self, arduino, informar=None):
        self.arduino = arduino
        self.informar = informar or (lambda mensaje: None)
        self._t = None
        self._lock = threading.Lock()
        self._cancelar = threading.Event()
        self.error = None
        # Revisar los nueve antes de permitir la primera ejecucion.
        for nombre in sorted(NOMBRES):
            try:
                cargar(nombre)
            except ValueError as exc:
                self._avisar(str(exc))

    def _avisar(self, mensaje):
        self.error = mensaje
        print(f" AVISO: {mensaje}")
        self.informar(mensaje)

    def reproducir(self, nombre):
        with self._lock:
            if self._t and self._t.is_alive():
                return False
            try:
                bloques = cargar(nombre)
            except ValueError as exc:
                self._avisar(str(exc))
                return False
            self.error = None
            self._cancelar.clear()
            self._t = threading.Thread(target=self._run, args=(nombre, bloques), daemon=True)
            self._t.start()
            return True

    def detener(self):
        self._cancelar.set()
        if self._t:
            self._t.join()

    def _run(self, nombre, bloques):
        try:
            with self.arduino.secuencia_exclusiva():
                t0 = time.monotonic()
                for pca, canal, inicio, duracion, posicion, velocidad in bloques:
                    if self._cancelar.wait(max(0, t0 + inicio - time.monotonic())):
                        return
                    self.arduino.enviar(pca, canal, posicion, velocidad)
                # Margen tras el ultimo envio: el protocolo no confirma llegada.
                self._cancelar.wait(0.5)
            print(f" {Path(nombre).stem.replace(chr(95), chr(32)).capitalize()} finalizada")
        except Exception as exc:
            self._avisar(f"No se pudo reproducir {nombre}: {exc}")
