"""IPC real con respuestas simuladas; no valida reconocimiento de emociones."""
from contextlib import ExitStack
from pathlib import Path
import sys
import json
import subprocess
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from experimental.emociones_proceso import EmocionesProceso, ROOT, enviar,interprete_emociones,EmocionesWeb


class ProcesoTests(unittest.TestCase):
    def test_web_habitual_conecta_y_cierra_backend_sin_fallback_local(self):
        from queue import Queue
        import threading
        from interfaz.trabajador import ejecutar_trabajador
        backend=MagicMock()
        with patch('experimental.emociones_proceso.EmocionesWeb',return_value=backend) as crear, \
             patch('aplicacion.sistema.ejecutar') as ejecutar,patch('hardware.exclusion.Exclusivo'):
            eventos=Queue()
            ejecutar_trabajador('sistema',{'demo':True},eventos,Queue(),Queue(),threading.Event(),True)
        crear.assert_called_once_with(demo=True)
        self.assertIs(ejecutar.call_args.kwargs['emociones_externas'],backend)
        backend.cerrar.assert_called_once()
        self.assertEqual(list(eventos.queue)[-1],('finalizado','sistema'))

    def test_backend_fisico_usa_proceso_aislado(self):
        cliente=MagicMock()
        with patch('experimental.emociones_proceso.EmocionesProceso',return_value=cliente) as crear:
            c=EmocionesWeb(demo=False)
            c.actualizar(self.imagen)
            crear.assert_called_once()
            cliente.actualizar.assert_called_once()
            c.cerrar();cliente.cerrar.assert_called_once()

    def test_fallo_backend_se_aisla_y_desactivar_permite_reintento(self):
        cliente=MagicMock();cliente.actualizar.side_effect=RuntimeError('Fallo simulado')
        cliente.estado.return_value=dict(ok=False,intentado=True,externo=True,detalle='desactivado')
        with patch('experimental.emociones_proceso.EmocionesProceso',return_value=cliente):
            c=EmocionesWeb()
            self.assertIsNone(c.actualizar(self.imagen))
            self.assertIn('seguir usando',c.estado()['mensaje_usuario'])
            cliente.desactivar.assert_called_once()
            cliente.actualizar.side_effect=None;cliente.actualizar.return_value=None
            c.actualizar(None,activo=False)
            c.actualizar(self.imagen,activo=True)
            self.assertEqual(c.error,'')
            c.cerrar();c.cerrar();cliente.cerrar.assert_called_once()

    def test_control_normal_muestra_fallo_y_limpia_al_parar(self):
        from web_local.control import Control
        import threading
        c=Control();c.estado.update(fase='activo',actividad='sistema',rostros=1,nombre='Registrada',expresion='Anterior')
        c.evento('deteccion',dict(rostros=1,gesto='puno',deepface=dict(ok=False,intentado=True,externo=True,
            detalle='no disponible: timeout',mensaje_usuario='Demora de emociones')))
        self.assertIsNone(c.estado['expresion'])
        self.assertEqual(c.estado['detalle_emociones'],'Demora de emociones')
        self.assertEqual(c.estado['nombre'],'Registrada')
        self.assertEqual(c.estado['gesto'],'Puño')
        c.proceso=MagicMock();c.parada=threading.Event();c.detener()
        c.evento('expresion',dict(nombre='Atrasada',confianza=.9))
        c.evento('deteccion',dict(rostros=1,gesto=None,deepface=dict(ok=True,intentado=True,externo=True,detalle='listo')))
        self.assertIsNone(c.estado['expresion'])
        self.assertEqual(c.estado['emociones_estado'],'detenido')

    def test_emociones_fisicas_se_habilitan_por_api(self):
        from web_local.control import Control
        from queue import Queue
        c=Control();c.proceso=MagicMock();c.comandos=Queue();c.estado['modo']='fisico'
        c.opciones({'emociones':True})
        self.assertTrue(c.estado['opciones']['emociones'])
        self.assertEqual(c.comandos.get(),('opciones',{'emociones':True}))

    def test_seleccion_entorno_nuevo_y_compatibilidad_con_existente(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td,patch('experimental.emociones_proceso.ROOT',Path(td)):
            esperado=Path(td)/'.venv_emociones/Scripts/python.exe'
            self.assertEqual(interprete_emociones(),esperado)
            legado=Path(td)/'.venv_prueba_emociones/Scripts/python.exe';legado.parent.mkdir(parents=True);legado.touch()
            self.assertEqual(interprete_emociones(),legado)
            esperado.parent.mkdir(parents=True);esperado.touch()
            self.assertEqual(interprete_emociones(),esperado)

    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Sin camara')))
        self.stack.enter_context(patch('serial.Serial', side_effect=AssertionError('Sin Arduino')))
        self.addCleanup(self.stack.close)
        self.imagen = np.ones((80, 80, 3), np.uint8)

    def cliente(self, **kwargs):
        c = EmocionesProceso(simulado=kwargs.pop('simulado', 'normal'), intervalo=0.,
                             inicio_timeout=kwargs.pop('inicio_timeout',10), **kwargs)
        self.addCleanup(c.cerrar)
        return c

    def esperar(self, c, condicion, imagen=True, objetivo='A', segundos=8):
        fin = time.monotonic()+segundos
        while time.monotonic() < fin:
            c.actualizar(self.imagen if imagen else None, objetivo=objetivo)
            if condicion():
                return
            time.sleep(.01)
        self.fail(str(c.estado()))

    def test_comunicacion_copia_y_cierre(self):
        c = self.cliente()
        self.esperar(c, lambda: c.resultado is not None)
        self.assertEqual(c.resultado['nombre'], 'Feliz')
        self.assertEqual(c.proceso.args[0], str(interprete_emociones()))
        proceso = c.proceso
        c.cerrar()
        self.assertIsNotNone(proceso.poll())
        self.assertFalse(c.hilo.is_alive())
        self.assertIsNone(c.resultado)
        self.assertNotIn('deepface', sys.modules)

    def test_cola_unica_sustituye_antiguos_y_copia(self):
        c = self.cliente(demora=.3)
        c.actualizar(self.imagen, objetivo='A')
        original = self.imagen.copy()
        self.imagen.fill(0)
        with c.lock:
            _, datos = list(c.cola.queue)[0]
            self.assertEqual(datos, original.tobytes())
        maximo = 0.
        for _ in range(100):
            t = time.monotonic()
            c.actualizar(self.imagen, objetivo='A')
            maximo = max(maximo, time.monotonic()-t)
            self.assertLessEqual(c.cola.qsize(), 1)
        self.assertGreater(c.descartadas, 0)
        self.assertLess(maximo, .25, 'No debe esperar inferencia o IPC')

    def test_timeout_limpia_y_para_proceso(self):
        c = self.cliente(demora=.5, timeout=.08)
        self.esperar(c, lambda: c.fallo)
        self.assertIsNone(c.resultado)
        self.assertIn('tiempo de espera', c.mensaje)
        c.cerrar()
        self.assertIsNotNone(c.proceso.poll())

    def test_caida_y_error_no_salen_al_bucle_principal(self):
        for modo in ('caida', 'error'):
            with self.subTest(modo=modo):
                c = self.cliente(simulado=modo)
                self.esperar(c, lambda: c.fallo)
                self.assertIsNone(c.resultado)
                self.assertIn('no disponible', c.mensaje)
                c.cerrar()

    def test_desactivar_en_curso_y_reiniciar_sin_resucitar_resultado(self):
        c = self.cliente(demora=.15)
        self.esperar(c, lambda: c.en_curso > 0)
        sesion = c.sesion
        proceso = c.proceso
        c.desactivar()
        self.assertIsNone(c.resultado)
        self.assertNotEqual(sesion, c.sesion)
        self.esperar(c, lambda: c.resultado is not None, objetivo='B')
        self.assertEqual(c.resultado['sesion'], c.sesion)
        self.assertIsNotNone(proceso.poll())
        self.assertIsNot(c.proceso, proceso)

    def test_rechaza_sesion_id_incorrectos_caducados_y_sin_rostro(self):
        c = self.cliente()
        c.activo = True
        solicitud = dict(id=1, sesion=c.sesion, captura=time.monotonic())
        respuesta = dict(id=1, sesion=c.sesion, nombre='Feliz', confianza=.5)
        with c.lock:
            c._aceptar(dict(respuesta, sesion='anterior'), solicitud)
            c._aceptar(dict(respuesta, id=2), solicitud)
            c._aceptar(respuesta, dict(solicitud, captura=time.monotonic()-10))
            self.assertIsNone(c.resultado)
            c._aceptar(respuesta, solicitud)
            self.assertIsNotNone(c.resultado)
        c.actualizar(None)
        self.assertIsNone(c.resultado)
        c.actualizar(self.imagen, objetivo='Otra')
        self.assertIsNone(c.resultado)

    def test_interprete_ausente_no_abre_hardware(self):
        c = self.cliente(interprete=ROOT/'no-existe/python.exe')
        self.esperar(c, lambda: c.fallo)
        self.assertIn('FileNotFoundError', c.mensaje)

    def test_timeout_inicio_y_resultado_que_caduca(self):
        c = self.cliente()
        with c.lock:
            c.actualizar(self.imagen, objetivo='A')
            c.inicio = time.monotonic()-20  # Deadline vencido, sin depender de la velocidad del PC.
            c.actualizar(self.imagen, objetivo='A')
        self.assertTrue(c.fallo)
        self.assertIsNone(c.resultado)
        otro = self.cliente()
        self.esperar(otro, lambda: otro.resultado is not None)
        with otro.lock:
            otro.resultado['captura'] = time.monotonic()-10
            otro.resultado['nombre'] = 'Vieja'
            otro.intervalo = 100
        r = otro.actualizar(self.imagen, objetivo='A')
        self.assertTrue(r is None or r['nombre'] != 'Vieja')

    def test_eof_cierra_hijo_incluso_durante_inferencia_lenta(self):
        p = subprocess.Popen([str(interprete_emociones()), '-B',
            str(ROOT/'experimental/worker_emociones.py'), '--simulado', 'normal', '--demora', '20'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            self.assertEqual(json.loads(p.stdout.readline())['tipo'], 'listo')
            enviar(p.stdin, dict(sesion='eof', id=1, captura=time.monotonic(), shape=[80,80,3]),self.imagen.tobytes())
            time.sleep(.1)
            p.stdin.close()
            p.wait(timeout=3)
            self.assertIsNotNone(p.poll())
        finally:
            if p.poll() is None:
                p.kill()
            p.wait()
            if not p.stdin.closed:
                p.stdin.close()
            p.stdout.close()

    def test_web_limpia_resultado_y_no_permite_altas_fisico_ni_bajas(self):
        from experimental.web_emociones import crear_control
        c = crear_control()
        c.estado.update(rostros=1, expresion='Anterior')
        c.evento('deteccion', dict(rostros=1, gesto=None,
            deepface=dict(ok=False, intentado=True, externo=True, detalle='no disponible: timeout')))
        self.assertIsNone(c.ver()['expresion'])
        self.assertEqual(c.ver()['emociones_estado'], 'no disponible')
        self.assertEqual(c.ver()['detalle_emociones'], 'no disponible: timeout')
        for actividad, kw in [('registro', {}), ('sistema', {'modo':'fisico'})]:
            with self.assertRaises(ValueError):
                c.iniciar(actividad, **kw)
        with self.assertRaises(ValueError):
            c.eliminar('Nadie')
        c.estado['expresion']='Anterior'
        c.detener()
        self.assertIsNone(c.ver()['expresion'])

    def test_sistema_usa_copia_bgr_y_no_lanza_deepface_local(self):
        from aplicacion import sistema
        from types import SimpleNamespace as NS
        with ExitStack() as s:
            cap=MagicMock(); cap.isOpened.return_value=True
            cap.get.side_effect=lambda p: 320 if p==3 else 240 if p==4 else 30
            frame=np.full((240,320,3),20,np.uint8)
            cap.read.side_effect=[(True,frame.copy()),(True,frame.copy()),(False,None)]
            cara=MagicMock(); cara.detect.side_effect=[NS(detections=[NS(bounding_box=NS(origin_x=20,origin_y=20,width=100,height=120))]),NS(detections=[])]
            mano=MagicMock(); mano.detect.return_value=NS(hand_landmarks=[])
            backend=MagicMock(); backend.estado.return_value={'ok':False,'intentado':True,'externo':True,'detalle':'no disponible'}
            backend.actualizar.return_value=None
            for nombre,valor in [('abrir_camara',cap),('Arduino',MagicMock(ser=None)),('cargar_db',{'nombres':[],'imagenes':[]})]:
                s.enter_context(patch.object(sistema,nombre,return_value=valor))
            for nombre in ('descargar_modelos','Reproductor'):
                s.enter_context(patch.object(sistema,nombre))
            cargar=s.enter_context(patch.object(sistema,'_cargar_deepface',side_effect=AssertionError('No DeepFace principal')))
            s.enter_context(patch.object(sistema.mp_vision.FaceDetector,'create_from_options',return_value=cara))
            s.enter_context(patch.object(sistema.mp_vision.HandLandmarker,'create_from_options',return_value=mano))
            s.enter_context(patch.object(sistema.time,'sleep'))
            puente=MagicMock(); puente.detener.return_value=False
            puente.opciones.return_value={'emociones':True,'gestos':True}
            sistema.ejecutar(demo=True,interfaz=puente,emociones_externas=backend)
            cargar.assert_not_called()
            roi=backend.actualizar.call_args_list[0].args[0]
            self.assertEqual(roi.shape,(120,100,3))
            self.assertTrue(np.all(roi==20))
            self.assertFalse(np.shares_memory(roi,frame))
            backend.cerrar.assert_called_once()
            self.assertTrue(any(c.args[0]=='deteccion' for c in puente.emitir.call_args_list))
