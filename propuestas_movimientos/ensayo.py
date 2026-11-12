"""Un JSON propuesto por ensayo; por defecto solo simula, sin hardware."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from contextlib import nullcontext
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propuestas_movimientos.preparar import CARPETA, propuestas, validar
from secuencias.reproductor import Reproductor


class ReceptorSimulado:
    def __init__(self):
        self.ordenes = []

    def enviar(self, *orden):
        self.ordenes.append(orden)

    def secuencia_exclusiva(self):
        return nullcontext()


class RelojSimulado:
    ahora = 0

    def time(self):
        return self.ahora

    monotonic = time

    def sleep(self, segundos):
        self.ahora += segundos


def simular(ruta):
    receptor = ReceptorSimulado()
    reproductor = Reproductor(receptor)
    bloques = sorted(json.loads(ruta.read_text(encoding="utf-8")), key=lambda b: b[2])
    validar(bloques)
    reloj = RelojSimulado()
    def esperar(segundos):
        reloj.sleep(segundos)
        return False
    with patch("secuencias.reproductor.time", reloj), patch.object(reproductor._cancelar, "wait", side_effect=esperar):
        reproductor._run(ruta.name, bloques)
    esperadas = [(b[0], b[1], b[4], b[5]) for b in bloques]
    if receptor.ordenes != esperadas:
        raise ValueError("La simulacion no coincide con los bloques del JSON")
    return len(receptor.ordenes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo", choices=sorted(propuestas()))
    parser.add_argument("--fisico", action="store_true")
    parser.add_argument("--puerto")
    args = parser.parse_args()
    ruta = CARPETA / args.archivo
    print("SECUENCIA PROPUESTA; no original ni validada mecanicamente.")
    print(validar(json.loads(ruta.read_text(encoding="utf-8"))))
    if not args.fisico:
        print(f"SIMULACION: {simular(ruta)} ordenes comprobadas; sin Arduino ni camara.")
        return
    if not args.puerto:
        parser.error("Indica --puerto con el puerto conocido para el ensayo supervisado.")
    print("Solo con la profesora: el primer envio ya puede mover servos. No hay parada de emergencia por software.")
    if input(f"Para autorizar este unico ensayo escribe {args.archivo}: ").strip() != args.archivo:
        print("Cancelado. No se abrio Arduino.")
        return
    from hardware.exclusion import Exclusivo
    from hardware.arduino import Arduino
    from interfaz.conexion import SerialObservado
    with Exclusivo("escritorio"), Exclusivo("camara"):
        arduino = Arduino(port=args.puerto, demo=False)
        if arduino.demo or arduino.ser is None or not arduino.ser.is_open:
            if arduino.ser is not None:
                arduino.ser.close()
            raise RuntimeError("No se abrio el puerto. No se reproduce la propuesta.")
        reproductor = Reproductor(arduino)
        def informar(tipo, valor):
            reproductor._cancelar.set()
            print("Error/desconexion de puerto. No hay confirmacion de movimientos.")
        serial = SerialObservado(arduino.ser, informar, threading.Event())
        arduino.ser = serial
        try:
            # El reproductor resuelve la ruta absoluta dentro de propuestas.
            if not reproductor.reproducir(args.archivo):
                raise RuntimeError("Propuesta rechazada; no se envia su contenido.")
            reproductor._t.join()
            time.sleep(1.5)
            if serial.fallo:
                raise RuntimeError("Ensayo interrumpido por error de comunicacion.")
            print("Ordenes programadas finalizadas; comprobar el resultado fisico con la profesora.")
        finally:
            reproductor.detener()
            serial.close()


if __name__ == "__main__":
    main()
