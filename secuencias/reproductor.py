"""Propuestas pendientes de validacion fisica; reproduccion exclusiva."""
import json
from pathlib import Path
import threading
import time
import types

from propuestas_movimientos.preparar import CARPETA, validar
from vision.manos import GESTO_ACCION

NOMBRES = frozenset(nombre for nombre, _ in GESTO_ACCION.values())


class Cancelada(Exception):
    pass


class AccionCancelable:
    def __init__(self, arduino, cancelar):
        self.arduino, self.cancelar = arduino, cancelar

    def esperar(self, segundos):
        if self.cancelar.wait(segundos):
            raise Cancelada()

    def enviar(self, *args, **kwargs):
        if self.cancelar.is_set():
            raise Cancelada()
        self.arduino.enviar(*args, **kwargs)

    def __getattr__(self, nombre):
        valor = getattr(self.arduino, nombre)
        if isinstance(valor, types.MethodType):
            return types.MethodType(valor.__func__, self)
        return valor


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
        self._cerrado = False
        self.origen = None
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

    def reproducir(self, nombre, origen='gesto'):
        with self._lock:
            if self._cerrado or self._t and self._t.is_alive():
                return False
            try:
                bloques = cargar(nombre)
            except ValueError as exc:
                self._avisar(str(exc))
                return False
            self.error = None
            self.origen = origen
            self._cancelar.clear()
            self._t = threading.Thread(target=self._run, args=(nombre, bloques), daemon=True)
            self._t.start()
            return True

    def accion(self, nombre, funcion):
        """Sin cola: una accion ocupada no reaparece al acabar otra."""
        with self._lock:
            if self._cerrado or self._t and self._t.is_alive():
                return False
            self._cancelar.clear()
            self.origen = nombre
            self._t = threading.Thread(target=self._accion, args=(funcion,), daemon=True)
            self._t.start()
            return True

    def _accion(self, funcion):
        try:
            with self.arduino.secuencia_exclusiva():
                funcion(AccionCancelable(self.arduino, self._cancelar))
        except Cancelada:
            pass
        except Exception as exc:
            self._avisar(f"No se pudo ejecutar {self.origen}: {exc}")

    def cancelar_emocion(self):
        with self._lock:
            if self.origen == 'emocion':
                self._cancelar.set()

    def detener(self):
        with self._lock:
            self._cerrado = True
            self._cancelar.set()
            hilo = self._t
        if hilo:
            hilo.join()

    def _run(self, nombre, bloques):
        try:
            with self.arduino.secuencia_exclusiva():
                t0 = time.monotonic()
                for pca, canal, inicio, duracion, posicion, velocidad in bloques:
                    if self._cancelar.wait(max(0, t0 + inicio - time.monotonic())):
                        return
                    self.arduino.enviar(pca, canal, posicion, velocidad)
                # Margen tras el ultimo envio: el protocolo no confirma llegada.
                if self._cancelar.wait(0.5):
                    return
            print(f" {Path(nombre).stem.replace(chr(95), chr(32)).capitalize()} finalizada")
        except Exception as exc:
            self._avisar(f"No se pudo reproducir {nombre}: {exc}")


class EmocionesMovimiento:
    """Dos inferencias distintas coherentes; nunca una cola de movimientos."""
    MAPA = {'Feliz': 'alegria.json', 'Triste': 'tristeza.json',
            'Enojado': 'enojo.json', 'Sorprendido': 'sorpresa.json'}

    def __init__(self, reproductor):
        self.rep = reproductor
        self.ultima = None
        self.candidata = None
        self.repetidas = 0
        self.ultimo_movimiento = float('-inf')

    def actualizar(self, respuesta, activo=True):
        if not activo or not respuesta:
            self.rep.cancelar_emocion()
            self.candidata = None
            self.repetidas = 0
            self.ultima = None
            return False
        clave = (respuesta.get('sesion'), respuesta.get('id'))
        if None in clave or clave == self.ultima:
            return False
        self.ultima = clave
        nombre = respuesta['nombre']
        self.repetidas = self.repetidas + 1 if nombre == self.candidata else 1
        self.candidata = nombre
        if self.repetidas < 2 or time.monotonic() - self.ultimo_movimiento < 5:
            return False
        if nombre == 'Neutro':
            from movimientos.expresiones import expresion_neutro
            iniciada = self.rep.accion('emocion', expresion_neutro)
        elif nombre in self.MAPA:
            iniciada = self.rep.reproducir(self.MAPA[nombre], origen='emocion')
        else:
            return False  # Asustado y Disgustado no tienen secuencia propia.
        if iniciada:
            self.ultimo_movimiento = time.monotonic()
        return iniciada
