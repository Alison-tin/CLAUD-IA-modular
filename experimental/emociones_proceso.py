"""Cliente no bloqueante: una petición en curso y solo el último recorte pendiente."""
import json
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def enviar(pipe, cabecera, datos=b''):
    cabecera = dict(cabecera, bytes=len(datos))
    texto = json.dumps(cabecera).encode('utf-8')
    pipe.write(struct.pack('!I', len(texto)) + texto + datos)
    pipe.flush()


class EmocionesProceso:
    def __init__(self, interprete=None, timeout=2.5, inicio_timeout=30., vigencia=2.5,
                 intervalo=.5, simulado=None, demora=0.):
        self.interprete = Path(interprete or ROOT/'.venv_prueba_emociones/Scripts/python.exe')
        self.timeout, self.inicio_timeout, self.vigencia = timeout, inicio_timeout, vigencia
        self.intervalo, self.simulado, self.demora = intervalo, simulado, demora
        self.lock = threading.RLock()
        self.cola = queue.Queue(maxsize=1)
        self.proceso = self.hilo = None
        self.sesion = uuid.uuid4().hex
        self.ejecucion = uuid.uuid4().hex
        self.numero = self.aceptada = 0
        self.activo = self.cerrado = False
        self.inicio = self.en_curso = self.ultimo_envio = 0.
        self.resultado = None
        self.mensaje = 'desactivado'
        self.fallo = False
        self.objetivo = None
        self.descartadas = 0

    def _vaciar(self):
        try:
            while True:
                self.cola.get_nowait()
                self.descartadas += 1
        except queue.Empty:
            pass

    def _invalidar(self):
        self.sesion = uuid.uuid4().hex
        self.resultado = None
        self.aceptada = 0
        self._vaciar()

    def _matar(self):
        p = self.proceso
        if p is not None and p.poll() is None:
            p.kill()  # No esperar en el bucle de visión.

    def desactivar(self):
        with self.lock:
            self.activo = False
            self.ejecucion = uuid.uuid4().hex
            self._invalidar()
            self.objetivo = None
            self.mensaje = 'desactivado'
            self.fallo = False
            self._matar()

    def _arrancar(self):
        if self.hilo is not None and self.hilo.is_alive():
            return
        self.inicio = time.monotonic()
        self.mensaje = 'cargando proceso separado'
        self.hilo = threading.Thread(target=self._trabajar, args=(self.ejecucion,), daemon=True)
        self.hilo.start()

    def _trabajar(self, ejecucion):
        p = None
        try:
            comando = [str(self.interprete), '-B', str(ROOT/'experimental/worker_emociones.py')]
            if self.simulado:
                comando += ['--simulado', self.simulado, '--demora', str(self.demora)]
            p = subprocess.Popen(comando, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with self.lock:
                self.proceso = p
                if not self.activo or self.cerrado or self.fallo or ejecucion != self.ejecucion:
                    self._matar()
            listo = json.loads(p.stdout.readline())
            if listo.get('tipo') != 'listo':
                raise RuntimeError(listo.get('error', 'Inicio fallido'))
            with self.lock:
                if not self.activo or self.cerrado or self.fallo or ejecucion != self.ejecucion:
                    return
                self.inicio = 0.
                self.mensaje = 'listo'
            while True:
                with self.lock:
                    if not self.activo or self.cerrado or ejecucion != self.ejecucion:
                        return
                try:
                    solicitud, datos = self.cola.get(timeout=.05)
                except queue.Empty:
                    if p.poll() is not None:
                        raise RuntimeError('El proceso de emociones terminó')
                    continue
                with self.lock:
                    if solicitud['sesion'] != self.sesion:
                        continue
                    self.en_curso = time.monotonic()
                enviar(p.stdin, solicitud, datos)
                respuesta = json.loads(p.stdout.readline())
                with self.lock:
                    self.en_curso = 0.
                    self._aceptar(respuesta, solicitud)
        except Exception as error:
            with self.lock:
                if self.activo and not self.cerrado and ejecucion == self.ejecucion and not self.fallo:
                    self.resultado = None
                    self.fallo = True
                    self.mensaje = f'no disponible: {type(error).__name__}: {error}'[:180]
                    self._vaciar()
        finally:
            if p is not None:
                if p.poll() is None:
                    p.kill()
                p.wait()
                p.stdin.close()
                p.stdout.close()
            with self.lock:
                self.en_curso = self.inicio = 0.

    def _aceptar(self, respuesta, solicitud):
        """Llamado bajo lock; nunca asociar otro resultado al rostro actual."""
        if (not self.activo or self.cerrado or self.fallo
                or respuesta.get('sesion') != self.sesion
                or respuesta.get('id') != solicitud['id']
                or respuesta['id'] <= self.aceptada
                or time.monotonic()-solicitud['captura'] > self.vigencia):
            self.descartadas += 1
            return
        if respuesta.get('error'):
            self.resultado = None
            self.fallo = True
            self.mensaje = 'no disponible: ' + str(respuesta['error'])[:150]
            self._matar()
            return
        confianza = respuesta.get('confianza')
        nombre = respuesta.get('nombre')
        if not isinstance(nombre, str) or not isinstance(confianza, (int, float)) or not 0 <= confianza <= 1:
            raise ValueError('Resultado de emociones invalido')
        self.aceptada = respuesta['id']
        self.resultado = dict(nombre=nombre, confianza=confianza, captura=solicitud['captura'],
                              id=respuesta['id'], sesion=self.sesion)
        self.mensaje = 'listo'

    def actualizar(self, recorte=None, activo=True, objetivo=None):
        """Copia acotada de un ROI BGR; no IPC, inferencia ni esperas en este hilo."""
        with self.lock:
            if self.cerrado:
                return None
            if not activo:
                if self.activo:
                    self.desactivar()
                return None
            if not self.activo:
                self.activo = True
                self.fallo = False
                self._invalidar()
            ahora = time.monotonic()
            if self.resultado and ahora-self.resultado['captura'] > self.vigencia:
                self.resultado = None
                self.mensaje = 'no disponible: resultado caducado'
            if self.inicio and ahora-self.inicio > self.inicio_timeout or self.en_curso and ahora-self.en_curso > self.timeout:
                self.fallo = True
                self.resultado = None
                self.mensaje = 'no disponible: tiempo de espera agotado'
                self._vaciar()
                self._matar()
            if not self.fallo:
                self._arrancar()
            if recorte is None or objetivo != self.objetivo:
                self._invalidar()
                self.objetivo = objetivo
            if recorte is None:
                return None
            if not self.fallo and ahora-self.ultimo_envio >= self.intervalo:
                if recorte.dtype != np.uint8 or recorte.ndim != 3 or recorte.shape[2] != 3 or not recorte.size:
                    raise ValueError('Recorte BGR invalido')
                h, w = recorte.shape[:2]
                escala = min(1., 320/max(h, w))
                copia = cv2.resize(recorte, (max(1, int(w*escala)), max(1, int(h*escala)))).copy()
                self.numero += 1
                solicitud = dict(sesion=self.sesion, id=self.numero, captura=ahora, shape=list(copia.shape))
                self._vaciar()
                self.cola.put_nowait((solicitud, copia.tobytes()))
                self.ultimo_envio = ahora
            return dict(self.resultado) if self.resultado else None

    def estado(self):
        with self.lock:
            return dict(ok=self.activo and not self.fallo and self.mensaje == 'listo',
                        intentado=True, externo=True, detalle=self.mensaje,
                        pendientes=self.cola.qsize(), descartadas=self.descartadas)

    def cerrar(self):
        self.desactivar()
        self.cerrado = True
        if self.hilo:
            self.hilo.join(timeout=5)
            if self.hilo.is_alive():
                raise RuntimeError('No se pudo cerrar el supervisor de emociones')
