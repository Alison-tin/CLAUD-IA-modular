import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from interfaz.recorrido import Recorrido, indicacion_postura, postura_correcta
from vision.orientacion import Observacion, EstimadorOrientacion, angulos_persona


def imagen(i=0):
    return np.random.default_rng(i+500).integers(0, 256, (200, 200), dtype=np.uint8)


class DiagnosticoTests(unittest.TestCase):
    def test_convencion_fisica_y_espejo_no_intercambian_derecha_e_izquierda(self):
        for giro in (15, -15):
            y = np.radians(-giro)
            r = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
            m = np.eye(4)
            m[:3, :3] = r
            nariz = r @ np.array([0, 0, 2])
            # Normal hacia -x es derecha fisica de quien mira a la camara.
            self.assertEqual(np.sign(nariz[0]), -np.sign(giro))
            # Reflejar solo la vista lleva la nariz al lado indicado por el SVG.
            self.assertEqual(np.sign(-nariz[0]), np.sign(giro))
            self.assertAlmostEqual(angulos_persona(m)[0], giro)

    def test_rangos_originales_y_mensajes_de_mas_y_menos_giro(self):
        casos = [(1, (5, 0, 0), 'derecha'), (1, (30, 0, 0), 'demasiado'),
                 (2, (-5, 0, 0), 'izquierda'), (2, (-30, 0, 0), 'demasiado'),
                 (3, (0, 5, 0), 'Levanta'), (3, (0, 26, 0), 'Baja'),
                 (4, (0, -5, 0), 'Baja'), (4, (0, -26, 0), 'Sube'),
                 (0, (0, 0, 11), 'Endereza')]
        for paso, angulos, texto in casos:
            codigo, mensaje = indicacion_postura(paso, angulos)
            self.assertTrue(codigo)
            self.assertIn(texto, mensaje)
            self.assertFalse(postura_correcta(paso, angulos))
        for a in (10, 25):
            self.assertTrue(postura_correcta(1, (a, 0, 0)))
        self.assertFalse(postura_correcta(1, (9.99, 0, 0)))
        self.assertFalse(postura_correcta(1, (25.01, 0, 0)))

    def test_postura_correcta_con_mala_luz_se_marca_sin_autorizar_captura(self):
        g = Recorrido(0)
        for i in range(8):
            g.actualizar(Observacion(1, (0, 0, 0), motivo='Falta luz', diagnostico={'bloqueo': 'luz_baja'}), .2*(i+1))
        e = g.estado()
        self.assertTrue(e['postura_ok'])
        self.assertFalse(e['calidad_ok'])
        self.assertEqual(e['lecturas_estables'], 0)
        self.assertEqual(e['diagnostico']['bloqueo'], 'luz_baja')
        self.assertEqual(g.fotos, [])
        json.dumps(e, allow_nan=False)

    def test_progreso_refleja_lecturas_estables_y_no_ventana_llena_inestable(self):
        g = Recorrido(0)
        for i, yaw in enumerate((0, 0, 4, 4)):
            g.actualizar(Observacion(1, (yaw, 0, 0), imagen(), 100), .2*(i+1))
        self.assertEqual(g.estado()['lecturas_estables'], 2)
        self.assertTrue(g.estado()['postura_ok'])
        self.assertIn('movimiento', g.mensaje)
        self.assertEqual(g.fotos, [])
        for i in range(4):
            g.actualizar(Observacion(1, (4, 0, 0), imagen(), 100), 1+.2*i)
        self.assertEqual(g.paso, 1)
        self.assertEqual(len(g.fotos), 1)
        self.assertFalse(g.estado()['postura_ok'])  # No arrastrar verde al paso siguiente.

    def test_elegir_candidata_util_aunque_la_mas_nitida_sea_repetida(self):
        g = Recorrido(0)
        for i in range(6):
            g.actualizar(Observacion(1, (0, 0, 0), imagen(), 100), .2*(i+1))
        for i in range(6):
            repetida = i == 5
            g.actualizar(Observacion(1, (15, 0, 0), imagen(0 if repetida else 1), 999 if repetida else 100), 2+.2*i)
        self.assertEqual(g.paso, 2)
        self.assertEqual(g.detalles[-1]['nitidez'], 100)
        self.assertEqual(len(g.fotos), 2)

    def test_diagnostico_no_mutable_ni_imagenes_y_cancelar_limpia_progreso(self):
        g = Recorrido(0)
        g.actualizar(Observacion(1, (0, 0, 0), imagen(), 100, diagnostico={'filtros': {'luz': True}}), .2)
        e = g.estado()
        e['diagnostico']['filtros']['luz'] = False
        self.assertTrue(g.diagnostico['filtros']['luz'])
        g.cancelar()
        e = g.estado()
        self.assertFalse(e['postura_ok'])
        self.assertEqual(e['lecturas_estables'], 0)
        self.assertEqual(e['seleccionadas'], 0)
        json.dumps(e, allow_nan=False)


class FiltrosOrientacionTests(unittest.TestCase):
    def setUp(self):
        self.e = EstimadorOrientacion.__new__(EstimadorOrientacion)
        self.e.mp, self.e.modelo = MagicMock(), MagicMock()
        self.puntos = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
        self.e.modelo.detect.return_value = SimpleNamespace(face_landmarks=[self.puntos], facial_transformation_matrixes=[np.eye(4)])

    def test_luz_contraste_y_nitidez_se_informan_separadamente(self):
        gradiente = np.broadcast_to(np.linspace(30, 220, 400, dtype=np.uint8)[None, :, None], (400, 400, 3)).copy()
        casos = [(np.zeros((400, 400, 3), np.uint8), 'luz_baja'),
                 (np.full((400, 400, 3), 255, np.uint8), 'luz_alta'),
                 (np.full((400, 400, 3), 128, np.uint8), 'contraste'),
                 (gradiente, 'nitidez')]
        with patch('datos.personas.detectar_rostros', return_value=[(50, 50, 280, 280)]):
            for frame, codigo in casos:
                o = self.e.analizar(frame)
                self.assertEqual(o.diagnostico['bloqueo'], codigo)
                self.assertIsNotNone(o.angulos)  # La calidad no oculta la postura.
                self.assertIsNone(o.imagen)
                self.assertIn('nitidez', o.diagnostico)
                json.dumps(o.diagnostico, allow_nan=False)

    def test_detectores_en_desacuerdo_y_puntos_no_finitos(self):
        frame = np.zeros((400, 400, 3), np.uint8)
        with patch('datos.personas.detectar_rostros', return_value=[]):
            o = self.e.analizar(frame)
            self.assertEqual(o.diagnostico['detector_rostros'], 0)
            self.assertEqual(o.diagnostico['landmarker_rostros'], 1)
            self.assertEqual(o.diagnostico['bloqueo'], 'rostros')
        self.puntos[1] = SimpleNamespace(x=float('nan'), y=.5)
        self.puntos[263] = SimpleNamespace(x=.7, y=.5)
        with patch('datos.personas.detectar_rostros', return_value=[(50, 50, 280, 280)]):
            o = self.e.analizar(frame)
            self.assertEqual(o.diagnostico['bloqueo'], 'encuadre')
            json.dumps(o.diagnostico, allow_nan=False)
