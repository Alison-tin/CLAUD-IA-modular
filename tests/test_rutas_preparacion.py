from contextlib import ExitStack
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibrar_duplicados import preparar
from interfaz.duplicados import cargar_referencias, resolver_referencia


class RutasPreparacionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camara prohibida')))
        self.stack.enter_context(patch('serial.Serial', side_effect=AssertionError('Arduino prohibido')))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.fotos = self.root/'configuracion'/'rostros_fotos'
        self.fotos.mkdir(parents=True)
        self.db = self.root/'configuracion'/'base.pkl'
        self.salida = self.root/'estudio'
        self.base = {'nombres': ['Primera', 'Segunda'], 'imagenes': []}
        for i, sesion in enumerate(('registro_abc', 'registro_def')):
            carpeta = self.fotos/sesion
            carpeta.mkdir()
            imagen = np.random.default_rng(i).integers(0, 256, (200, 200), dtype=np.uint8)
            (carpeta/'foto_001.jpg').write_bytes(cv2.imencode('.jpg', imagen)[1].tobytes())
            self.base['imagenes'].append(f'C:\\equipo_anterior\\configuracion\\rostros_fotos\\{sesion}\\foto_001.jpg')
        # Foto homonima que jamas debe sustituir una muestra de otra sesion.
        (self.fotos/'foto_001.jpg').write_bytes(b'no pertenece a ninguna sesion')
        self.db.write_bytes(pickle.dumps(self.base))

    def test_respeta_sesion_y_no_cambia_base_ni_fotos(self):
        antes = {str(p): p.read_bytes() for p in self.fotos.rglob('*.jpg')}
        base_previa = self.db.read_bytes()
        for ruta, sesion in zip(self.base['imagenes'], ('registro_abc', 'registro_def')):
            self.assertEqual(resolver_referencia(ruta, self.fotos), (self.fotos/sesion/'foto_001.jpg').resolve())
        grupos, _ = cargar_referencias(self.base, self.fotos)
        self.assertEqual(set(grupos), {'Primera', 'Segunda'})
        preparar(self.salida, self.db, self.fotos)
        self.assertEqual(self.db.read_bytes(), base_previa)
        self.assertEqual({str(p): p.read_bytes() for p in self.fotos.rglob('*.jpg')}, antes)

    def test_entrenamiento_principal_incluye_sesiones_sin_cambiar_parametros(self):
        from datos.personas import entrenar_reconocedor
        previo = self.db.read_bytes()
        modelo, nombres = entrenar_reconocedor(db=self.base, fotos_dir=self.fotos)
        self.assertEqual(set(nombres.values()), {'Primera', 'Segunda'})
        self.assertEqual(len(modelo.getHistograms()), 2)
        self.assertEqual((modelo.getRadius(), modelo.getNeighbors(), modelo.getGridX(), modelo.getGridY()), (1, 8, 8, 8))
        self.assertEqual(self.db.read_bytes(), previo)

    def test_faltantes_identifica_todos_los_registros_no_usa_homonimos(self):
        for sesion in ('registro_abc', 'registro_def'):
            (self.fotos/sesion/'foto_001.jpg').unlink()
        with self.assertRaises(ValueError) as contexto:
            preparar(self.salida, self.db, self.fotos)
        self.assertIn('registro Primera', str(contexto.exception))
        self.assertIn('registro Segunda', str(contexto.exception))
        self.assertFalse(self.salida.exists())

    def test_rutas_relativas_y_rechazo_de_busqueda_por_nombre(self):
        for ruta in ('configuracion/rostros_fotos/registro_abc/foto_001.jpg',
                     'registro_abc/foto_001.jpg'):
            self.assertEqual(resolver_referencia(ruta, self.fotos), (self.fotos/'registro_abc/foto_001.jpg').resolve())
        for ruta in ('C:\\otra_carpeta\\foto_001.jpg',
                     'C:\\anterior\\rostros_fotos\\..\\foto_001.jpg',
                     'C:\\anterior\\rostros_fotos\\sesion_inexistente\\foto_001.jpg'):
            with self.assertRaises((ValueError, FileNotFoundError)):
                resolver_referencia(ruta, self.fotos)

    def test_recuperar_vacio_y_repetir_preserva_sesiones(self):
        self.salida.mkdir()
        preparar(self.salida, self.db, self.fotos)
        previo = (self.salida/'estudio.json').read_bytes()
        sesion = self.salida/'sesion_ya_capturada'
        sesion.mkdir()
        (sesion/'foto_001.jpg').write_bytes(b'datos validos que no hay que tocar')
        preparar(self.salida, self.db, self.fotos)
        self.assertEqual((self.salida/'estudio.json').read_bytes(), previo)
        self.assertEqual((sesion/'foto_001.jpg').read_bytes(), b'datos validos que no hay que tocar')

    def test_recuperacion_explicita_respalda_manifiesto_truncado(self):
        self.salida.mkdir()
        roto = b'{"version": 1, '
        (self.salida/'estudio.json').write_bytes(roto)
        with self.assertRaisesRegex(ValueError, '--recuperar'):
            preparar(self.salida, self.db, self.fotos)
        preparar(self.salida, self.db, self.fotos, recuperar=True)
        self.assertEqual(next(self.salida.glob('estudio.json.incompleto_*')).read_bytes(), roto)
        self.assertEqual(json.loads((self.salida/'estudio.json').read_text(encoding='utf-8'))['personas_registradas'], ['Primera', 'Segunda'])

    def test_no_sobrescribe_estudio_otra_base_ni_parcial_con_datos(self):
        preparar(self.salida, self.db, self.fotos)
        manifiesto = self.salida/'estudio.json'
        datos = json.loads(manifiesto.read_text(encoding='utf-8'))
        datos['base_sha256'] = 'otra_base'
        manifiesto.write_text(json.dumps(datos), encoding='utf-8')
        previo = manifiesto.read_bytes()
        with self.assertRaisesRegex(ValueError, 'otra base'):
            preparar(self.salida, self.db, self.fotos, recuperar=True)
        self.assertEqual(manifiesto.read_bytes(), previo)
        manifiesto.write_bytes(b'{')
        (self.salida/'foto_conservar.jpg').write_bytes(b'conservar')
        with self.assertRaisesRegex(ValueError, 'otros datos'):
            preparar(self.salida, self.db, self.fotos, recuperar=True)
        self.assertEqual(manifiesto.read_bytes(), b'{')

    def test_fallo_escritura_no_deja_manifiesto_truncado_y_permite_repetir(self):
        with patch('calibrar_duplicados.escribir_json', side_effect=OSError('disco simulado')):
            with self.assertRaises(OSError):
                preparar(self.salida, self.db, self.fotos)
        self.assertEqual(list(self.salida.iterdir()), [])
        preparar(self.salida, self.db, self.fotos)
        self.assertTrue((self.salida/'estudio.json').is_file())
