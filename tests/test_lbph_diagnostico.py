import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import cv2
import numpy as np

from datos.personas import entrenar_reconocedor, extraer_rostro
from vision.fotogramas import imagen_mediapipe


class Diagnostico(unittest.TestCase):
    def test_distancias_por_identidad_son_del_mismo_modelo(self):
        from vision.reconocimiento import distancias_por_identidad
        fotos = [np.random.default_rng(i).integers(0, 256, (200, 200), dtype=np.uint8)
                 for i in range(3)]
        modelo = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        modelo.train(fotos, np.array([0, 1, 0], np.int32))
        nombres = {0: 'Alexa', 1: 'daniela'}
        ds = distancias_por_identidad(modelo, fotos[0], nombres)
        etiqueta, distancia = modelo.predict(fotos[0])
        self.assertEqual(set(ds), set(nombres.values()))
        self.assertAlmostEqual(ds[nombres[etiqueta]], distancia)
        self.assertAlmostEqual(min(ds.values()), distancia)

    def test_etiquetas_y_no_segunda_ecualizacion(self):
        with tempfile.TemporaryDirectory() as temporal:
            rutas = []
            for i in range(3):
                ruta = Path(temporal) / f'{i}.jpg'
                imagen = np.random.default_rng(i).integers(0, 256, (200, 200), dtype=np.uint8)
                cv2.imwrite(str(ruta), cv2.equalizeHist(imagen))
                rutas.append(str(ruta))
            base = {'nombres': ['Alexa', 'Alison', 'Alexa'], 'imagenes': rutas}
            entradas = []
            with patch('datos.personas.cv2.face.LBPHFaceRecognizer_create') as crear:
                _, nombres = entrenar_reconocedor(db=base, fotos_dir=temporal, entradas_debug=entradas)
                caras, etiquetas = crear.return_value.train.call_args.args
            self.assertEqual(nombres, {0: 'Alexa', 1: 'Alison'})
            np.testing.assert_array_equal(etiquetas, [0, 1, 0])
            for cara, ruta in zip(caras, rutas):
                np.testing.assert_array_equal(cara, cv2.imread(ruta, 0))
            for cara, etiqueta, entrada in zip(caras, etiquetas, entradas):
                np.testing.assert_array_equal(entrada['imagen'], cara)
                self.assertEqual(entrada['etiqueta'], etiqueta)
                self.assertFalse(np.shares_memory(entrada['imagen'], cara))

    def test_no_truncar_base(self):
        with self.assertRaises(ValueError):
            entrenar_reconocedor(db={'nombres': ['Alexa'], 'imagenes': []})

    def test_recorte_exacto(self):
        frame = np.random.default_rng(4).integers(0, 256, (300, 400, 3), dtype=np.uint8)
        esperado = cv2.equalizeHist(cv2.resize(cv2.cvtColor(frame[35:165, 85:215], cv2.COLOR_BGR2GRAY), (200, 200)))
        diagnostico = {}
        np.testing.assert_array_equal(extraer_rostro(frame, (100, 50, 100, 100), diagnostico), esperado)
        self.assertEqual(diagnostico['recorte'], (85, 35, 215, 165))
        np.testing.assert_array_equal(extraer_rostro(frame, (-10, -5, 100, 100)),
                                      extraer_rostro(frame, (0, 0, 100, 100)))
        self.assertIsNone(extraer_rostro(frame, (500, 0, 100, 100)))

    def test_mediapipe_vistas_y_propiedad(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        for vista in (frame, frame[:, ::-1], frame[::2, ::2]):
            imagen = imagen_mediapipe(vista)
            self.assertEqual((imagen.height, imagen.width), vista.shape[:2])
            frame[:] = 255
            self.assertFalse(np.shares_memory(imagen.numpy_view(), frame))
            frame[:] = 0
        for invalido in (None, frame[:0], frame[:, :, 0], frame.astype(float)):
            with self.assertRaises(ValueError):
                imagen_mediapipe(invalido)


if __name__ == '__main__':
    unittest.main()
