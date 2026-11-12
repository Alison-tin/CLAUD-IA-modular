"""Bloqueo local entre procesos; no controla hardware."""
import hashlib
import os
from pathlib import Path
import tempfile


class Exclusivo:
    def __init__(self, nombre):
        identidad = str(Path(__file__).resolve().parents[1]) + nombre
        self.ruta = Path(tempfile.gettempdir()) / (
            "claudia_" + hashlib.sha256(identidad.encode()).hexdigest() + ".lock")
        self.archivo = None

    def adquirir(self):
        archivo = self.ruta.open("a+b")
        archivo.seek(0, os.SEEK_END)
        if archivo.tell() == 0:
            archivo.write(b"0")
            archivo.flush()
        archivo.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(archivo.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(archivo, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            archivo.close()
            raise RuntimeError("Recurso ocupado por otra ejecucion de CLAUD-IA.")
        self.archivo = archivo
        return self

    def liberar(self):
        if self.archivo is not None:
            self.archivo.close()
            self.archivo = None

    def __enter__(self):
        return self.adquirir()

    def __exit__(self, *exc):
        self.liberar()
