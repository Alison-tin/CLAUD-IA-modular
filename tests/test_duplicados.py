"""Flujos con datos temporales; no evalua exactitud biometrica."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import pickle
from queue import Empty
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from interfaz.duplicados import (ComprobacionFacial, VerificadorDuplicados,
                                decidir, REPETIR)
from interfaz.registro import guardar_persona, leer_base
from interfaz.trabajador import registrar
from web_local.control import Control


class DuplicadosTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camara prohibida')))
        self.stack.enter_context(patch('serial.Serial', side_effect=AssertionError('Arduino prohibido')))
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.db, self.fotos = self.root/'base.pkl', self.root/'fotos'
        self.jpeg = cv2.imencode('.jpg', np.arange(40000, dtype=np.uint8).reshape(200, 200))[1].tobytes()
        self.original = self.root/'original.jpg'
        self.original.write_bytes(self.jpeg)
        self.base = {'nombres': ['Alison'], 'imagenes': [str(self.original)]}
        self.db.write_bytes(pickle.dumps(self.base))
        self.previo = self.snapshot()

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_votos_consistentes_dudosos_y_una_lectura(self):
        limites = {'Alison': [20, 50], 'Beatriz': [25, 60]}  # SOLO datos sinteticos.
        self.assertEqual(decidir([{'Alison': 10, 'Beatriz': 90}]*5, limites), ('duplicado', 'Alison'))
        self.assertEqual(decidir([{'Alison': 70, 'Beatriz': 90}]*5, limites), ('sin_coincidencia', None))
        for lecturas in ([{'Alison': 30, 'Beatriz': 90}]*5,
                         [{'Alison': 10, 'Beatriz': 15}]*5,
                         [{'Alison': 10, 'Beatriz': 90}]*4 + [{'Alison': 70, 'Beatriz': 90}],
                         [{'Alison': 70, 'Beatriz': 90}],
                         [{'Alison': float('nan'), 'Beatriz': 90}]*5,
                         [{'Alison': 10}]*5):
            self.assertEqual(decidir(lecturas, limites)[0], 'dudoso')

    def test_sin_calibracion_y_fotos_invalidas_no_permite_alta(self):
        v = VerificadorDuplicados(self.base, self.db, self.fotos)
        self.assertEqual(v.comprobar([self.jpeg]*5)[0], 'dudoso')
        with self.assertRaisesRegex(ComprobacionFacial, 'Resultado dudoso'):
            guardar_persona('Otro nombre', [self.jpeg]*5, self.db, self.fotos)
        self.assertEqual(self.snapshot(), self.previo)
        vacia = VerificadorDuplicados({'nombres': [], 'imagenes': []}, self.db, self.fotos)
        self.assertEqual(vacia.comprobar([b'invalida']*30)[0], 'dudoso')

    def test_compara_todas_las_personas_con_lbph_simulado_y_politica_vinculada(self):
        base = {'nombres': ['Alison', 'Beatriz'], 'imagenes': [str(self.original)]*2}
        v = VerificadorDuplicados(base, self.db, self.fotos)
        politica = {'version': 1, 'protocolo': 'guiado-v1-5', 'base_sha256': v.huella,
                    'validacion': {'informe': 'SIMULACION: no usar estos limites en produccion'},
                    'limites': {'Alison': [20, 50], 'Beatriz': [25, 60]}}
        archivo = Path(str(self.db)+'.duplicados.json')
        archivo.write_text(json.dumps(politica), encoding='utf-8')
        a, b = MagicMock(), MagicMock()
        a.predict.return_value, b.predict.return_value = (0, 10), (0, 90)
        with patch('cv2.face.LBPHFaceRecognizer_create', side_effect=[a, b]):
            v = VerificadorDuplicados(base, self.db, self.fotos)
        self.assertEqual(v.comprobar([self.jpeg]*5), ('duplicado', 'Alison'))
        self.assertEqual(a.predict.call_count, 5)
        self.assertEqual(b.predict.call_count, 5)
        self.original.write_bytes(self.jpeg+b'cambio')
        self.assertIsNone(VerificadorDuplicados(base, self.db, self.fotos).limites)
        self.original.unlink()
        self.assertIsNone(VerificadorDuplicados(base, self.db, self.fotos).limites)

    def test_guardado_final_rechazo_nombre_y_cancelacion_sin_archivos(self):
        for resultado in [('duplicado', 'Alison'), ('dudoso', None)]:
            with patch('interfaz.duplicados.VerificadorDuplicados') as factory:
                factory.return_value.comprobar.return_value = resultado
                with self.assertRaises(ComprobacionFacial):
                    guardar_persona('Otro nombre', [self.jpeg]*5, self.db, self.fotos)
                self.assertEqual(factory.call_args.args[0], self.base)
            self.assertEqual(self.snapshot(), self.previo)
        with self.assertRaises(ValueError):
            guardar_persona(' ALISON ', [self.jpeg]*5, self.db, self.fotos)
        with patch('interfaz.duplicados.VerificadorDuplicados') as factory:
            factory.return_value.comprobar.return_value = ('sin_coincidencia', None)
            for cancelar in (MagicMock(return_value=True), MagicMock(side_effect=[False, True])):
                with self.assertRaisesRegex(ValueError, 'cancelado'):
                    guardar_persona('Nueva', [self.jpeg]*5, self.db, self.fotos, cancelar=cancelar)
                self.assertEqual(self.snapshot(), self.previo)
            with patch('interfaz.registro.os.replace', side_effect=OSError('fallo simulado')):
                with self.assertRaises(OSError):
                    guardar_persona('Nueva', [self.jpeg]*5, self.db, self.fotos)
            self.assertEqual(self.snapshot(), self.previo)

    def flujo(self, resultado, rostros=None, cancelar_en=40, final=None):
        class Comandos:
            alternar = False
            def get_nowait(self):
                self.alternar = not self.alternar
                if self.alternar:
                    return 'capturar', None
                raise Empty
        from vision.orientacion import Observacion
        cap, puente = MagicMock(), MagicMock()
        estimador = MagicMock()
        poses = [(0,0,0), (15,0,0), (-15,0,0), (0,15,0), (0,-15,0)]
        if rostros is not None and len(rostros) != 1:
            estimador.analizar.return_value = Observacion(len(rostros))
        else:
            estimador.analizar.side_effect = [Observacion(1,a,np.random.default_rng(i).integers(0,256,(200,200),dtype=np.uint8),100) for i,a in enumerate(poses) for _ in range(6)]
        cap.isOpened.return_value = True
        cap.read.return_value = True, np.zeros((400, 400, 3), np.uint8)
        llamadas = 0
        def detener():
            nonlocal llamadas
            llamadas += 1
            return llamadas > cancelar_en
        puente.detener.side_effect = detener
        puente.comandos = Comandos()
        # Primera instancia: comprobacion por lotes; segunda: control final real.
        temprano, ultimo = MagicMock(), MagicMock()
        temprano.vacia = ultimo.vacia = False
        temprano.protocolo = ultimo.protocolo = "guiado-v1-5"
        temprano.comprobar.return_value = resultado
        ultimo.comprobar.return_value = final or resultado
        # Mock conserva argumentos mutables por referencia; la guia los limpia
        # al terminar. Capturar ahora permite contrastar las fotos realmente usadas.
        for comprobador in (temprano, ultimo):
            comprobador.muestras_recibidas = []
            def comprobar(muestras, actual=comprobador):
                actual.muestras_recibidas.append(tuple(muestras))
                return actual.comprobar.return_value
            comprobador.comprobar.side_effect = comprobar
        with ExitStack() as stack:
            stack.enter_context(patch('configuracion.config.DB_PATH', str(self.db)))
            stack.enter_context(patch('configuracion.config.FOTOS_DIR', str(self.fotos)))
            stack.enter_context(patch('hardware.camara.abrir_camara', return_value=cap))
            stack.enter_context(patch('vision.orientacion.EstimadorOrientacion', return_value=estimador))
            stack.enter_context(patch('interfaz.trabajador.time.monotonic', side_effect=iter(np.arange(0,100,.2))))
            stack.enter_context(patch('interfaz.trabajador.time.sleep'))
            stack.enter_context(patch('datos.personas.detectar_rostros', return_value=[(1, 1, 200, 200)] if rostros is None else rostros))
            stack.enter_context(patch('datos.personas.extraer_rostro', return_value=np.zeros((200, 200), np.uint8)))
            stack.enter_context(patch('interfaz.duplicados.VerificadorDuplicados', side_effect=[temprano, ultimo]))
            registrar(puente, 'Otro nombre', 0)
        cap.release.assert_called_once()
        return puente, temprano, ultimo

    def test_misma_persona_otro_nombre(self):
        puente, temprano, _ = self.flujo(('duplicado', 'Alison'))
        temprano.comprobar.assert_called_once()
        puente.emitir.assert_any_call('aviso_registro', 'Este rostro coincide con un registro existente: Alison')
        self.assertEqual(self.snapshot(), self.previo)

    def test_persona_diferente_y_comprobacion_final(self):
        puente, temprano, ultimo = self.flujo(('sin_coincidencia', None))
        self.assertEqual(temprano.comprobar.call_count, 1)
        ultimo.comprobar.assert_called_once()
        self.assertEqual(leer_base(self.db)['nombres'], ['Alison']+['Otro nombre']*5)
        puente.emitir.assert_any_call('guardado', 'Otro nombre')
        base = leer_base(self.db)
        guardadas = [Path(p).read_bytes() for p in base['imagenes'][1:]]
        hashes = lambda muestras: [hashlib.sha256(b).hexdigest() for b in muestras]
        self.assertEqual(hashes(guardadas), hashes(temprano.muestras_recibidas[0]))
        self.assertEqual(hashes(guardadas), hashes(ultimo.muestras_recibidas[0]))
        self.assertTrue(all(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE).shape == (200, 200)
                            for b in guardadas))
        self.assertEqual(len(set(base['imagenes'][1:])), 5)

    def test_final_puede_rechazar_aunque_previa_permita(self):
        puente, _, ultimo = self.flujo(('sin_coincidencia', None), final=('duplicado', 'Alison'))
        ultimo.comprobar.assert_called_once()
        puente.emitir.assert_any_call('aviso_registro', 'Este rostro coincide con un registro existente: Alison')
        self.assertEqual(self.snapshot(), self.previo)

    def test_dudoso_repite_sin_guardar(self):
        puente, _, _ = self.flujo(('dudoso', None))
        puente.emitir.assert_any_call('aviso_registro', REPETIR)
        self.assertEqual(self.snapshot(), self.previo)

    def test_sin_rostro_varios_rostros_y_cancelacion(self):
        for rostros in ([], [(1, 1, 50, 50), (80, 80, 50, 50)]):
            puente, temprano, _ = self.flujo(('sin_coincidencia', None), rostros=rostros, cancelar_en=3)
            temprano.comprobar.assert_not_called()
            self.assertEqual(self.snapshot(), self.previo)
        self.flujo(('sin_coincidencia', None), cancelar_en=3)
        self.assertEqual(self.snapshot(), self.previo)

    def test_aviso_visible_en_estado_web(self):
        control = Control()
        mensaje = 'Este rostro coincide con un registro existente: Alison'
        control.evento('aviso_registro', mensaje)
        self.assertEqual(control.ver()['aviso'], mensaje)
