import json
import os
from pathlib import Path
from queue import Queue
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import numpy as np
from hardware.arduino import Arduino
from interfaz.trabajador import Puente
from secuencias.reproductor import Reproductor, cargar, resolver_json
from propuestas_movimientos.preparar import validar
from vision.manos import GESTO_ACCION
from web_local.control import Control


class Rapido(threading.Event):
    def wait(self, timeout=None):
        return self.is_set()


class IntegracionTests(unittest.TestCase):
    def test_confirmacion_web_advierte_propuestas(self):
        html = (Path(__file__).resolve().parents[1]/'web_local/static/index.html').read_text(encoding='utf-8')
        dialogo = html.split('id="physical-dialog"', 1)[1].split('</dialog>', 1)[0]
        self.assertIn('Los gestos de mano activan los movimientos validados.', dialogo)

    def test_camara_simulada_nueve_gestos_demo_y_web_fisica_simulada(self):
        from aplicacion import sistema
        for web in (False, True):
            for gesto, (nombre, _) in GESTO_ACCION.items():
                with self.subTest(web=web, gesto=gesto), ExitStack() as st:
                    cap = MagicMock()
                    cap.isOpened.return_value = True
                    cap.get.side_effect = lambda prop: 640 if prop == 3 else 480 if prop == 4 else 30
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cap.read.side_effect = [(True, frame)] * 24 + [(False, None)]
                    cara, mano = MagicMock(), MagicMock()
                    cara.detect.return_value = NS(detections=[])
                    mano.detect.return_value = NS(hand_landmarks=[[NS(x=.5, y=.5)] * 21], handedness=[])
                    serial = MagicMock()
                    serial.is_open = True
                    serial.port = 'SIMULADO'
                    serial.write.side_effect = lambda b: len(b)
                    puerto = st.enter_context(patch('serial.Serial', return_value=serial))
                    st.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camara real prohibida')))
                    st.enter_context(patch.object(sistema, 'abrir_camara', return_value=cap))
                    st.enter_context(patch.object(sistema, 'descargar_modelos'))
                    st.enter_context(patch.object(sistema, '_cargar_deepface'))
                    st.enter_context(patch.object(sistema, 'deepface_estado', return_value={'ok':False, 'intentado':True}))
                    st.enter_context(patch.object(sistema, 'cargar_db', return_value={'imagenes':[], 'nombres':[]}))
                    st.enter_context(patch.object(sistema.mp_vision.FaceDetector, 'create_from_options', return_value=cara))
                    st.enter_context(patch.object(sistema.mp_vision.HandLandmarker, 'create_from_options', return_value=mano))
                    st.enter_context(patch.object(sistema, 'clasificar_gesto', return_value=gesto))
                    st.enter_context(patch.object(sistema, 'dibujar_mano', return_value=(0,0,10,10)))
                    st.enter_context(patch.object(sistema, 'dedos_arriba', return_value=[True]*5))
                    hablar = st.enter_context(patch.object(sistema, 'hablar', side_effect=AssertionError('Boca duplicada')))
                    st.enter_context(patch.object(sistema.time, 'sleep'))
                    reps, ordenes = [], []
                    class Registrado(Arduino):
                        def _enviar(self, pca, canal, pos, vel=5):
                            if self._secuencia == threading.get_ident():
                                ordenes.append((pca, canal, pos, vel))
                            super()._enviar(pca, canal, pos, vel)
                    st.enter_context(patch.object(sistema, 'Arduino', Registrado))
                    def crear(arduino, informar):
                        rep = Reproductor(arduino, informar)
                        rep._cancelar = Rapido()
                        reps.append(rep)
                        return rep
                    st.enter_context(patch.object(sistema, 'Reproductor', side_effect=crear))
                    puente = Puente(Queue(), Queue(), Queue(), threading.Event())
                    puente.video_limpio = web
                    def cuadro(frame):
                        # Los primeros siete fotogramas no confirman el gesto.
                        if cap.read.call_count < 8:
                            self.assertFalse(reps[0]._t)
                        if reps and reps[0]._t:
                            reps[0]._t.join(2)
                    puente.fotograma = cuadro
                    sistema.ejecutar(port='SIMULADO', demo=not web, interfaz=puente)
                    self.assertEqual(ordenes, [(b[0],b[1],b[4],b[5]) for b in cargar(nombre)])
                    eventos = list(puente.eventos.queue)
                    sec = [v for t,v in eventos if t == 'secuencia']
                    self.assertEqual(len(sec), 1)
                    self.assertEqual(sec[0]['archivo'], nombre)
                    self.assertTrue(sec[0]['iniciada'])
                    control = Control()
                    for t,v in eventos:
                        if t in ('secuencia', 'aviso_secuencia'):
                            control.evento(t,v)
                    self.assertTrue(control.ver()['aviso'].endswith(('iniciada','iniciado')))
                    if not web:
                        puerto.assert_not_called()
                        serial.write.assert_not_called()
                    hablar.assert_not_called()
                    cap.release.assert_called_once()

    def test_exclusion_entre_hilos_y_reproducciones(self):
        with patch('serial.Serial') as puerto:
            a = Arduino(demo=True)
            a.demo = False
            a.ser = MagicMock(is_open=True)
            with a.secuencia_exclusiva():
                t = threading.Thread(target=lambda: a.enviar(2,1,464,3))
                t.start(); t.join()
                a.ser.write.assert_not_called()
                a.enviar(2,1,464,3)
                a.ser.write.assert_called_once()
            a.enviar(2,1,464,3)
            self.assertEqual(a.ser.write.call_count, 2)
            rep = Reproductor(a)
            self.assertTrue(rep.reproducir('risa.json'))
            self.assertFalse(rep.reproducir('alegria.json'))
            rep.detener()
            puerto.assert_not_called()

    def test_invalidos_sin_envios_y_aviso_web(self):
        ejemplos = [[], {}, [[True,12,0,0,300,3]], [[0,12,False,0,300,3]],
                    [[0,12,float('nan'),0,300,3]], [[0,12,float('inf'),0,300,3]],
                    [[0,12,'0',0,300,3]], [[0,12,0,-1,300,3]],
                    [[0,12,0,0,900,3]], [[0,13,0,0,300,3]],
                    [[0,12,0,0,300,0]], [[0,12,0,0,300,True]],
                    [[0,12,10**400,0,300,3]], [[0,12,0,0,300,3,9]],
                    [[0,12,0,0,300,3],[0,12,0,0,300,3]]]
        a = Arduino(demo=True)
        control = Control()
        rep = Reproductor(a, lambda m: control.evento('aviso_secuencia', m))
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d)/'invalido.json'
            for datos in ejemplos:
                with self.subTest(datos=datos):
                    with self.assertRaises(ValueError):
                        validar(datos)
                    ruta.write_text(json.dumps(datos), encoding='utf-8')
                    with patch('secuencias.reproductor.resolver_json', return_value=str(ruta)), patch.object(a, 'enviar') as enviar:
                        self.assertFalse(rep.reproducir('risa.json'))
                        enviar.assert_not_called()
                        self.assertIn('invalida risa.json', control.ver()['aviso'])

    def test_rutas_desde_otro_directorio(self):
        anterior = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                for nombre,_ in GESTO_ACCION.values():
                    self.assertTrue(Path(resolver_json(nombre)).is_absolute())
                    self.assertTrue(cargar(nombre))
            finally:
                os.chdir(anterior)
