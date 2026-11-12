"""Recogida y preflight con base temporal y hardware prohibido."""
from contextlib import ExitStack
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import calibrar_duplicados as guia
from interfaz.duplicados import FALTA_CONFIGURAR, preparacion_registro
from interfaz.terminal import registrar_terminal
from interfaz.trabajador import registrar
from web_local.control import Control


class CalibracionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camara prohibida')))
        self.stack.enter_context(patch('serial.Serial', side_effect=AssertionError('Arduino prohibido')))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db, self.fotos, self.estudio = self.root/'base.pkl', self.root/'fotos', self.root/'estudio'
        self.fotos.mkdir()
        self.imagenes = [np.random.default_rng(i).integers(0, 256, (200, 200), dtype=np.uint8) for i in range(32)]
        for i, nombre in enumerate(('Alison', 'Beatriz')):
            (self.fotos/(nombre+'.jpg')).write_bytes(cv2.imencode('.jpg', self.imagenes[i])[1].tobytes())
        self.db.write_bytes(pickle.dumps({'nombres': ['Alison', 'Beatriz'],
                                        'imagenes': [str(self.fotos/(n+'.jpg')) for n in ('Alison', 'Beatriz')]}))
        self.originales = self.snapshot()

    def snapshot(self):
        return {str(p): p.read_bytes() for p in [self.db, *self.fotos.glob('*')]}

    def preparar(self):
        return guia.preparar(self.estudio, self.db, self.fotos)

    def capturar(self, **extras):
        args = dict(salida=self.estudio, grupo='registrada', persona='Alison', sesion='cal01',
                    conjunto='calibracion', condiciones='SIMULADO', db_path=self.db, fotos_dir=self.fotos,
                    pedir=MagicMock(return_value=''), mostrar=MagicMock())
        args.update(extras)
        return guia.capturar(**args)

    def simular_camara(self, rostros=None):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = True, np.zeros((400, 400, 3), np.uint8)
        self.stack.enter_context(patch('hardware.camara.abrir_camara', return_value=cap))
        self.stack.enter_context(patch('datos.personas.detectar_rostros', return_value=[(1, 1, 200, 200)] if rostros is None else rostros))
        self.stack.enter_context(patch('datos.personas.extraer_rostro', side_effect=self.imagenes[2:]))
        return cap

    def test_preparar_aislado_sin_hardware_ni_politica(self):
        self.preparar()
        manifiesto = json.loads((self.estudio/'estudio.json').read_text(encoding='utf-8'))
        self.assertEqual(manifiesto['personas_registradas'], ['Alison', 'Beatriz'])
        self.assertEqual(self.snapshot(), self.originales)
        self.assertFalse(Path(str(self.db)+'.duplicados.json').exists())
        contenido = (self.estudio/'estudio.json').read_bytes()
        self.preparar()
        self.assertEqual((self.estudio/'estudio.json').read_bytes(), contenido)
        for salida in (self.fotos, self.fotos/'subcarpeta', self.root, self.db):
            with self.assertRaises(ValueError):
                guia.preparar(salida, self.db, self.fotos)

    def test_30_capturas_y_medicion_lbph_offline_sin_altas(self):
        self.preparar()
        cap = self.simular_camara()
        self.assertTrue(self.capturar())
        cap.release.assert_called_once()
        self.assertEqual(len(list((self.estudio/'sesion_cal01').glob('*.jpg'))), 30)
        informe = self.estudio/'distancias.json'
        guia.medir(self.estudio, 'calibracion', informe, self.db, self.fotos)
        datos = json.loads(informe.read_text(encoding='utf-8'))
        self.assertEqual(len(datos['sesiones'][0]['distancias']), 30)
        for lectura in datos['sesiones'][0]['distancias']:
            self.assertEqual(set(lectura), {'Alison', 'Beatriz'})
        self.assertEqual(self.snapshot(), self.originales)
        self.assertFalse(Path(str(self.db)+'.duplicados.json').exists())
        self.assertFalse(preparacion_registro(self.db, self.fotos)['permitido'])
        with self.assertRaises(FileExistsError):
            guia.medir(self.estudio, 'calibracion', informe, self.db, self.fotos)

    def test_cancelar_y_cero_varios_rostros_no_deja_parciales(self):
        self.preparar()
        cap = self.simular_camara(rostros=[])
        self.assertFalse(self.capturar(pedir=MagicMock(side_effect=['', 'q'])))
        self.assertEqual(list(self.estudio.iterdir()), [self.estudio/'estudio.json'])
        cap.release.assert_called_once()
        with patch('datos.personas.detectar_rostros', return_value=[(1, 1, 20, 20)]*2):
            self.assertFalse(self.capturar(pedir=MagicMock(side_effect=['', 'q'])))
        self.assertEqual(list(self.estudio.iterdir()), [self.estudio/'estudio.json'])
        with patch('datos.personas.detectar_rostros', return_value=[(1, 1, 200, 200)]):
            self.assertFalse(self.capturar(pedir=MagicMock(side_effect=['', 'q'])))
        self.assertEqual(list(self.estudio.iterdir()), [self.estudio/'estudio.json'])
        self.assertEqual(self.snapshot(), self.originales)

    def test_fallo_lectura_y_cambio_base_invalidan_estudio(self):
        self.preparar()
        cap = self.simular_camara()
        cap.read.return_value = False, None
        with self.assertRaises(RuntimeError):
            self.capturar()
        self.assertEqual(list(self.estudio.iterdir()), [self.estudio/'estudio.json'])
        (self.fotos/'Alison.jpg').write_bytes(cv2.imencode('.jpg', self.imagenes[3])[1].tobytes())
        with self.assertRaisesRegex(ValueError, 'cambiaron'):
            guia.abrir_estudio(self.estudio, self.db, self.fotos)

    def test_separacion_personas_externas_y_capturas_inmutables(self):
        self.preparar()
        self.simular_camara()
        self.assertTrue(self.capturar(grupo='diferente', persona='externa01'))
        with self.assertRaisesRegex(ValueError, 'distintas'):
            self.capturar(grupo='diferente', persona='externa01', conjunto='prueba', sesion='prueba01')
        captura = self.estudio/'sesion_cal01'/'foto_001.jpg'
        captura.write_bytes(b'cambio')
        with self.assertRaisesRegex(ValueError, 'cambio'):
            guia.medir(self.estudio, 'calibracion', self.estudio/'informe.json', self.db, self.fotos)
        self.assertFalse((self.estudio/'informe.json').exists())
        self.assertEqual(self.snapshot(), self.originales)

    def test_preflight_web_y_trabajador_no_abren_camara(self):
        self.assertEqual(preparacion_registro(self.db, self.fotos), {'permitido': False, 'mensaje': FALTA_CONFIGURAR})
        control = Control(self.db, self.fotos)
        with patch.object(control.ctx, 'Process') as proceso:
            with self.assertRaisesRegex(ValueError, 'Comenzar registro'):
                control.iniciar('registro', 'Nueva')
            proceso.assert_not_called()
        with patch('configuracion.config.DB_PATH', str(self.db)), patch('configuracion.config.FOTOS_DIR', str(self.fotos)), patch('hardware.camara.abrir_camara') as camara:
            puente = MagicMock()
            registrar(puente, 'Nueva', 0)
            puente.emitir.assert_any_call('aviso_registro', FALTA_CONFIGURAR)
            camara.assert_not_called()
        self.assertEqual(self.snapshot(), self.originales)

    def test_terminal_historico_no_puede_saltar_control(self):
        from datos.personas import registrar_persona
        mensajes = []
        with patch('configuracion.config.DB_PATH', str(self.db)), patch('configuracion.config.FOTOS_DIR', str(self.fotos)), patch('hardware.camara.abrir_camara') as camara:
            self.assertFalse(registrar_terminal('Otro nombre', pedir=MagicMock(), mostrar=mensajes.append))
            self.assertIn(FALTA_CONFIGURAR, mensajes)
            self.assertFalse(registrar_persona('Otro nombre'))
            camara.assert_not_called()
        self.assertEqual(self.snapshot(), self.originales)

    def test_terminal_delega_en_mismo_trabajador_y_no_crea_ventanas(self):
        with patch('configuracion.config.DB_PATH', str(self.db)), patch('configuracion.config.FOTOS_DIR', str(self.fotos)), patch('interfaz.duplicados.preparacion_registro', return_value={'permitido': True}), patch('interfaz.trabajador.registrar') as trabajador, patch('cv2.imshow', side_effect=AssertionError('Ventanas prohibidas')):
            def simulado(puente, nombre, cam_idx, prueba=False):
                self.assertTrue(puente.comandos.empty())
                puente.emitir('guardado', nombre)
            trabajador.side_effect = simulado
            self.assertTrue(registrar_terminal('Nueva', pedir=lambda _: '', mostrar=lambda _: None))
            trabajador.assert_called_once()
