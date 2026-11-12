"""Intérprete headless, stdin/stdout privados. Nunca captura ni guarda imágenes."""
import argparse
import json
import os
from pathlib import Path
import queue
import struct
import sys
import threading
import time

if os.name == 'nt':
    import ctypes
    import msvcrt
    from ctypes import wintypes
    _read_file = ctypes.WinDLL('kernel32', use_last_error=True).ReadFile
    _read_file.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    _read_file.restype = wintypes.BOOL
    _peek_pipe = ctypes.WinDLL('kernel32', use_last_error=True).PeekNamedPipe
    _peek_pipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                          wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    _peek_pipe.restype = wintypes.BOOL


def leer_exacto(n):
    datos = bytearray()
    while len(datos) < n:
        if os.name == 'nt':
            # Leer el pipe sin retener el lock del stream Python/CRT durante
            # imports nativos que consultan isatty del mismo descriptor.
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            disponibles = wintypes.DWORD()
            if not _peek_pipe(handle, None, 0, None, ctypes.byref(disponibles), None):
                raise EOFError()
            if not disponibles.value:
                time.sleep(.01)
                continue
            buffer = ctypes.create_string_buffer(min(n-len(datos), disponibles.value))
            leidos = wintypes.DWORD()
            ok = _read_file(handle, buffer,
                            len(buffer), ctypes.byref(leidos), None)
            parte = buffer.raw[:leidos.value] if ok else b''
        else:
            parte = os.read(sys.stdin.fileno(), n-len(datos))
        if not parte:
            raise EOFError()
        datos.extend(parte)
    return bytes(datos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulado', choices=['normal', 'caida', 'error'])
    parser.add_argument('--demora', type=float, default=0.)
    args = parser.parse_args()
    salida = sys.stdout
    sys.stdout = sys.stderr  # DeepFace no debe contaminar el protocolo.
    entradas = queue.Queue(maxsize=1)

    def recibir():
        try:
            while True:
                n = struct.unpack('!I', leer_exacto(4))[0]
                if not 1 <= n <= 2048:
                    raise ValueError('Cabecera invalida')
                cabecera = json.loads(leer_exacto(n))
                if not 0 < cabecera['bytes'] <= 320*320*3:
                    raise ValueError('Imagen demasiado grande')
                datos = leer_exacto(cabecera['bytes'])
                entradas.put((cabecera, datos))
        except BaseException:
            os._exit(0)  # EOF también cierra durante una inferencia bloqueada si muere el padre.

    threading.Thread(target=recibir, daemon=True).start()

    def responder(valor):
        salida.write(json.dumps(valor)+'\n')
        salida.flush()

    try:
        import numpy as np
        import cv2
        import socket
        def prohibido(*args, **kwargs):
            raise RuntimeError('Este proceso no permite camara ni conexiones de red')
        cv2.VideoCapture = prohibido
        socket.socket.connect = prohibido
        if not args.simulado:
            carpeta = Path.home()/'.deepface'/'weights'
            for nombre in ('facial_expression_model_weights.onnx', 'haarcascade_eye.xml', 'haarcascade_frontalface_default.xml'):
                if not (carpeta/nombre).is_file():
                    raise FileNotFoundError('Falta modelo local: ' + nombre)
            os.environ['DEEPFACE_BACKEND_ENGINE'] = 'onnx'
            from deepface import DeepFace
            DeepFace.analyze(np.full((100,100,3),128,np.uint8), actions=['emotion'],
                             enforce_detection=False, silent=True)
        responder(dict(tipo='listo'))
    except Exception as error:
        responder(dict(tipo='error', error=str(error)))
        return 1
    etiquetas = {'angry':'Enojado', 'disgust':'Disgustado', 'fear':'Asustado',
                 'happy':'Feliz', 'sad':'Triste', 'surprise':'Sorprendido', 'neutral':'Neutro'}
    while True:
        cabecera, datos = entradas.get()
        try:
            if args.demora:
                time.sleep(args.demora)
            if args.simulado == 'caida':
                os._exit(7)
            if args.simulado == 'error':
                raise RuntimeError('Error de inferencia simulado')
            imagen = np.frombuffer(datos, np.uint8).reshape(cabecera['shape']).copy()
            if args.simulado:
                nombre, confianza = ('Feliz' if imagen[0,0,0] else 'Triste'), .5
            else:
                r = DeepFace.analyze(imagen, actions=['emotion'], enforce_detection=False, silent=True)
                r = r[0] if isinstance(r, list) else r
                nombre = etiquetas[r['dominant_emotion']]
                confianza = float(r['emotion'][r['dominant_emotion']])/100
            responder(dict(sesion=cabecera['sesion'], id=cabecera['id'], nombre=nombre, confianza=confianza))
        except Exception as error:
            responder(dict(sesion=cabecera['sesion'], id=cabecera['id'], error=str(error)))


if __name__ == '__main__':
    raise SystemExit(main())
