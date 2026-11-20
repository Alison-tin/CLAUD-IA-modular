"""Comunicación de fallos del trabajador, sin abrir dispositivos."""
from queue import Queue
import threading
import unittest
from unittest.mock import patch

from interfaz.trabajador import ejecutar_trabajador
from web_local.control import Control


class ArranqueTests(unittest.TestCase):
    def test_primer_arranque_crea_base_vacia_sin_reemplazar_registros(self):
        from pathlib import Path
        import tempfile,pickle
        from interfaz.registro import inicializar_base,leer_base
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/'configuracion'/'base.pkl'
            self.assertTrue(inicializar_base(db))
            self.assertEqual(leer_base(db),{'nombres':[],'imagenes':[]})
            previa=pickle.dumps({'nombres':['Existente'],'imagenes':['conservar.jpg']})
            db.write_bytes(previa)
            self.assertFalse(inicializar_base(db))
            self.assertEqual(db.read_bytes(),previa)

    def test_fallo_al_inicializar_no_deja_base_parcial_ni_reemplaza_una_corrupta(self):
        from pathlib import Path
        import tempfile
        from interfaz.registro import inicializar_base
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/'base.pkl'
            with patch('interfaz.registro.os.replace',side_effect=OSError('Fallo simulado')):
                with self.assertRaises(OSError):inicializar_base(db)
            self.assertFalse(list(Path(td).iterdir()))
            db.write_bytes(b'No reemplazar')
            with self.assertRaises(Exception):inicializar_base(db)
            self.assertEqual(db.read_bytes(),b'No reemplazar')

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
