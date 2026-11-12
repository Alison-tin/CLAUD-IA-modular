import unittest
from unittest.mock import MagicMock, patch
from web_local.control import Control

class SimilitudTests(unittest.TestCase):
    def test_conversion_y_umbral(self):
        from vision.reconocimiento import decidir_reconocimiento
        for d, expected in [(0, 1), (30, .7), (80, .2), (100, 0), (120, 0)]:
            self.assertAlmostEqual(decidir_reconocimiento(0, d, {0: 'Alison'}, 150)['similitud_estimada'], expected)
        for d in (58, 80, -1, float('nan'), float('inf')):
            result = decidir_reconocimiento(0, d, {0: 'Alison'}, 58)
            self.assertEqual(result['nombre'], 'Desconocido')
            self.assertIsNone(result['similitud_estimada'])
        self.assertEqual(decidir_reconocimiento(99, 0, {0: 'Alison'}, 58)['nombre'], 'Desconocido')

    def test_estado_y_limpieza(self):
        with patch('cv2.VideoCapture', side_effect=AssertionError('Camara prohibida')), patch('serial.Serial', side_effect=AssertionError('Arduino prohibido')):
            c = Control()
            c.estado['motor_seleccionado']='lbph'  # Solo prueba de saneamiento legado, no una opcion web  # Esta prueba valida el porcentaje exclusivo de LBPH.
            def lectura(nombre='Alison', valor=.7):
                c.evento('reconocimiento', {'nombre': nombre, 'similitud_estimada': valor})
            lectura()
            self.assertEqual(c.ver()['similitud_estimada'], .7)
            lectura('Desconocido')
            self.assertEqual(c.ver()['nombre'], 'Desconocido')
            self.assertIsNone(c.ver()['similitud_estimada'])
            for value in (None, float('nan'), float('inf'), -1, 2, True):
                lectura(valor=value)
                self.assertIsNone(c.ver()['similitud_estimada'])
            lectura()
            c.evento('deteccion', {'rostros': 0, 'gesto': None, 'deepface': {'ok': True}})
            self.assertIsNone(c.ver()['similitud_estimada'])
            lectura()
            c.opciones({'reconocimiento': False})
            lectura()
            self.assertIsNone(c.ver()['similitud_estimada'])
            c.opciones({'reconocimiento': True})
            lectura()
            c.proceso = MagicMock()
            c.parada = MagicMock()
            c.detener()
            lectura()
            self.assertIsNone(c.ver()['nombre'])
            self.assertIsNone(c.ver()['similitud_estimada'])
