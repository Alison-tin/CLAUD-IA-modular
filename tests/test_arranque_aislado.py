"""Comunicación de fallos del trabajador, sin abrir dispositivos."""
from queue import Queue
import threading
import unittest
from unittest.mock import patch

from interfaz.trabajador import ejecutar_trabajador
from web_local.control import Control


class ArranqueTests(unittest.TestCase):
    def test_error_camara_llega_a_web_y_limpia_identidad(self):
        eventos = Queue()
        with patch('hardware.exclusion.Exclusivo'), \
             patch('aplicacion.sistema.ejecutar', side_effect=RuntimeError('Camara 9 no disponible')), \
             patch('cv2.VideoCapture', side_effect=AssertionError('Hardware prohibido')) as camara, \
             patch('serial.Serial', side_effect=AssertionError('Hardware prohibido')) as serie:
            ejecutar_trabajador('sistema', {'demo': True}, eventos, Queue(), Queue(), threading.Event())
        camara.assert_not_called()
        serie.assert_not_called()
        mensajes = list(eventos.queue)
        self.assertIn(('error', 'RuntimeError: Camara 9 no disponible'), mensajes)
        self.assertEqual(mensajes[-1], ('finalizado', 'sistema'))
        control = Control()
        control.estado.update(nombre='Anterior', similitud_estimada=90, fase='activo')
        for tipo, valor in mensajes:
            control.evento(tipo, valor)
        estado = control.ver()
        self.assertEqual(estado['fase'], 'error')
        self.assertIsNone(estado['nombre'])
        self.assertIsNone(estado['similitud_estimada'])
        self.assertRegex(estado['aviso'], r'[cC][aá]mara')
