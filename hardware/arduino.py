import time
import threading
from contextlib import contextmanager
import serial
import serial.tools.list_ports

from configuracion.config import (
    PCA,
    BAUD,
    SERVO,
    PCA0,
    SERVO_CUELLO,
    PARPADOS_ABIERTOS,
    PARPADOS_CERRADOS,
    MAXILAR_PCA,
    MAXILAR_CANAL,
    MAXILAR_ABRE,
    MAXILAR_CIERRA
)


# ======================================================
# DETECCIÓN DEL PUERTO ARDUINO
# ======================================================

ARDUINO_KW = [
    "arduino",
    "ch340",
    "ch341",
    "cp210",
    "ftdi",
    "usb serial",
    "usb-serial"
]


def detectar_puerto():
    for p in serial.tools.list_ports.comports():
        if any(
            k in (p.description or "").lower()
            or k in (p.manufacturer or "").lower()
            for k in ARDUINO_KW
        ):
            return p.device

    puertos = list(serial.tools.list_ports.comports())

    return puertos[0].device if len(puertos) == 1 else None


# ======================================================
# CLASE ARDUINO
# ======================================================

class Arduino:

    def __init__(self, port=None, demo=False):

        self.demo = demo
        self.ser = None
        self._orden_lock = threading.RLock()
        self._secuencia = None


        if demo:
            print("Modo DEMO")
            return

        port = port or detectar_puerto()

        if not port:
            print("Arduino no encontrado - modo demo")
            self.demo = True
            return

        try:
            self.ser = serial.Serial(
                port,
                BAUD,
                timeout=0
            )

            time.sleep(2)

            print(f"Arduino en {port}")

        except Exception as e:
            print(f"{e} - modo demo")
            self.demo = True


    # ==================================================
    # ENVIAR MOVIMIENTO
    # ==================================================

    def enviar(self, pca, canal, pos, vel=5):

        with self._orden_lock:
            if self._secuencia is not None and self._secuencia != threading.get_ident():
                return  # No acumular ordenes ajenas para enviarlas al terminar.
            self._enviar(pca, canal, pos, vel)

    @contextmanager
    def secuencia_exclusiva(self):
        # Adquisicion atomica respecto a cualquier envio en curso.
        with self._orden_lock:
            if self._secuencia is not None:
                raise RuntimeError("Ya hay una secuencia en reproduccion")
            self._secuencia = threading.get_ident()
        try:
            yield
        finally:
            with self._orden_lock:
                self._secuencia = None

    def _enviar(self, pca, canal, pos, vel=5):

        pos = int(max(0, min(1000, pos)))

        cmd = f"PCA,{pca},{canal},{pos},{int(vel)}\n"

        if not self.demo and self.ser and self.ser.is_open:

            try:
                self.ser.write(cmd.encode())

            except Exception as e:
                print(f"Serial: {e}")


    # ==================================================
    # POSICIÓN CENTRAL
    # ==================================================

    def ir_a_centro(self, vel=3):

        for canal, (mn, mx, centro) in SERVO.items():

            self.enviar(
                PCA,
                canal,
                centro,
                vel
            )

            time.sleep(0.02)

        # Cuello y maxilar
        for canal, (
            mn,
            mx,
            centro,
            velocidad,
            nombre,
            *_
        ) in SERVO_CUELLO.items():

            self.enviar(
                PCA0,
                canal,
                centro,
                velocidad
            )

            time.sleep(0.02)


    # ==================================================
    # PÁRPADOS
    # ==================================================

    def parpado(self, abrir=True, vel=2):

        posiciones = (
            PARPADOS_ABIERTOS
            if abrir
            else PARPADOS_CERRADOS
        )

        for canal, posicion in posiciones.items():

            self.enviar(
                PCA,
                canal,
                posicion,
                vel
            )


    # ==================================================
    # MAXILAR
    # ==================================================

    def mover_maxilar(self, abrir=True, vel=5):

        posicion = (
            MAXILAR_ABRE
            if abrir
            else MAXILAR_CIERRA
        )

        self.enviar(
            MAXILAR_PCA,
            MAXILAR_CANAL,
            posicion,
            vel
        )
