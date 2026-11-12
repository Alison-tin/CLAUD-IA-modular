from contextlib import ExitStack
from pathlib import Path
import pickle
from queue import Queue
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from interfaz.recorrido import Recorrido, PROTOCOLO, postura_correcta
from vision.orientacion import Observacion, angulos_persona

ANGULOS = [(0, 0, 0), (15, 0, 0), (-15, 0, 0), (0, 15, 0), (0, -15, 0)]


def imagen(paso):
    return np.random.default_rng(paso+100).integers(0, 256, (200, 200), dtype=np.uint8)


def lecturas():
    return [Observacion(1, a, imagen(i), 100+k) for i, a in enumerate(ANGULOS) for k in range(6)]


class MotorTests(unittest.TestCase):
    def test_repetir_una_conserva_otras_cuatro(self):
        g=Recorrido(0,conservar_originales=True)
        for i,o in enumerate(lecturas()):
            o.original_bgr=np.full((150,150,3),i,np.uint8)
            g.actualizar(o,(i+1)*.2)
        fotos=list(g.fotos);originales=[f.copy() for f in g.originales]
        g.repetir_postura(3,7)
        self.assertEqual(g.estado()['completados'],4)
        self.assertEqual(g.estado()['paso'],3)
        for k in range(6):
            o=Observacion(1,ANGULOS[2],imagen(99),200+k)
            o.original_bgr=np.full((150,150,3),255,np.uint8)
            g.actualizar(o,7+(k+1)*.2)
        self.assertTrue(g.completo)
        self.assertEqual(len(g.fotos),5)
        for i in (0,1,3,4):
            self.assertEqual(g.fotos[i],fotos[i])
            np.testing.assert_array_equal(g.originales[i],originales[i])
        self.assertNotEqual(g.fotos[2],fotos[2])

    def test_orientacion_usa_matriz_no_posicion_del_rectangulo(self):
        from vision.orientacion import EstimadorOrientacion
        e = EstimadorOrientacion.__new__(EstimadorOrientacion)
        e.mp = MagicMock()
        e.modelo = MagicMock()
        puntos = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
        frame = np.random.default_rng(0).integers(0, 256, (400, 400, 3), dtype=np.uint8)
        with patch('datos.personas.detectar_rostros', return_value=[(100, 100, 180, 180)]), patch('datos.personas.extraer_rostro', return_value=imagen(0)):
            for giro in (15, -15):
                y = np.radians(-giro)
                m = np.eye(4)
                m[:3, :3] = [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
                e.modelo.detect.return_value = SimpleNamespace(face_landmarks=[puntos], facial_transformation_matrixes=[m])
                o = e.analizar(frame)
                self.assertAlmostEqual(o.angulos[0], giro)
                self.assertIsNotNone(o.imagen)
            self.assertIsNone(e.analizar(np.zeros_like(frame)).imagen)

    def test_orden_cinco_fotos_y_mejor_lectura_sin_treinta_archivos(self):
        g = Recorrido(0)
        for i, obs in enumerate(lecturas()):
            g.actualizar(obs, (i+1)*.2)
        self.assertTrue(g.completo)
        self.assertEqual(g.analizados, 30)
        self.assertEqual(len(g.fotos), 5)
        self.assertEqual([d['paso'] for d in g.detalles], [1, 2, 3, 4, 5])
        self.assertTrue(all(d['nitidez'] == 105 for d in g.detalles))
        self.assertEqual(g.confirmacion, 'Captura del paso 5 confirmada.')

    def test_postura_incorrecta_extrema_o_inestable_no_avanza(self):
        for a in ((15, 0, 0), (0, 0, 20), (45, 0, 0), (0, 45, 0)):
            g = Recorrido(0)
            for i in range(8):
                g.actualizar(Observacion(1, a, imagen(0), 100), (i+1)*.2)
            self.assertEqual(g.paso, 0)
            self.assertEqual(g.fotos, [])
        g = Recorrido(0)
        for i in range(8):
            g.actualizar(Observacion(1, ((-1)**i*4, 0, 0), imagen(0), 100), (i+1)*.2)
        self.assertEqual(g.paso, 0)
        self.assertFalse(postura_correcta(1, ANGULOS[2]))
        self.assertFalse(postura_correcta(2, ANGULOS[1]))

    def test_un_solo_fotograma_ni_temporizador_autorizan_captura(self):
        g = Recorrido(0)
        g.actualizar(lecturas()[0], .1)
        g.actualizar(None, 10)
        self.assertEqual(g.fotos, [])
        g.actualizar(None, 21)
        self.assertTrue(g.pausado)
        self.assertEqual(g.paso, 0)
        g.reintentar(22)
        self.assertFalse(g.pausado)
        g.actualizar(None, 181)
        self.assertTrue(g.cancelado)
        self.assertEqual(g.fotos, [])

    def test_perdida_y_varios_rostros_descartan_lecturas_y_muestras(self):
        for cantidad in (0, 2):
            g = Recorrido(0)
            for i, o in enumerate(lecturas()[:6]):
                g.actualizar(o, (i+1)*.2)
            self.assertEqual(len(g.fotos), 1)
            g.actualizar(Observacion(cantidad), 2)
            g.actualizar(Observacion(cantidad), 4.1)
            self.assertEqual(g.paso, 0)
            self.assertEqual(g.fotos, [])
            self.assertEqual(g.ventana, [])

    def test_calidad_insuficiente_y_casi_duplicados_no_se_acumulan(self):
        g = Recorrido(0)
        for i in range(8):
            g.actualizar(Observacion(1, motivo='Falta luz'), (i+1)*.2)
        self.assertEqual(g.paso, 0)
        for i, o in enumerate(lecturas()[:6]):
            g.actualizar(o, 2+(i+1)*.2)
        for i in range(8):
            g.actualizar(Observacion(1, ANGULOS[1], imagen(0), 100), 4+(i+1)*.2)
        self.assertEqual(g.paso, 1)
        self.assertEqual(len(g.fotos), 1)
        g.cancelar()
        self.assertEqual(g.fotos, [])
        self.assertEqual(g.detalles, [])
        self.assertEqual(g.ventana, [])

    def test_limite_reintentos(self):
        g = Recorrido(0)
        for t in (21, 43, 65):
            g.actualizar(None, t)
            self.assertTrue(g.pausado)
            g.reintentar(t+1)
        self.assertTrue(g.cancelado)

    def test_matriz_derecha_izquierda_arriba_abajo_y_rechazo_espejo(self):
        for yaw, pitch, _ in ANGULOS:
            # Rotaciones canonicas: girar normal hacia -x es derecha del sujeto.
            x, y = np.radians([-pitch, -yaw])
            rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
            ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
            m = np.eye(4)
            m[:3, :3] = (ry @ rx)*1.2
            np.testing.assert_allclose(angulos_persona(m), [yaw, pitch, 0], atol=1e-6)
        m = np.eye(4)
        m[0, 0] = -1
        with self.assertRaises(ValueError):
            angulos_persona(m)


class TrabajadorGuiadoTests(unittest.TestCase):
    def test_cadencia_sincronizada_no_descarta_lecturas_por_inferencia_variable(self):
        reloj = [0.]
        numero = [0]
        def analizar(frame):
            reloj[0] += .11 if numero[0] % 2 == 0 else .01
            numero[0] += 1
            paso = self.guias[0].paso
            return Observacion(1, ANGULOS[paso], imagen(paso), 100)
        self.estimador.analizar.side_effect = analizar
        def dormir(_):
            reloj[0] += .04
        def leer():
            reloj[0] += .01
            return True, self.frame
        self.cap.read.side_effect = leer
        with patch('interfaz.trabajador.time.monotonic', side_effect=lambda: reloj[0]), patch('interfaz.trabajador.time.sleep', side_effect=dormir):
            self.ejecutar(prueba=True)
        self.assertEqual(self.estimador.analizar.call_count, 30)
        self.assertTrue(self.guias[0].completo)
        eventos = [c.args[1] for c in self.puente.emitir.call_args_list if c.args[0] == 'recorrido']
        self.assertTrue(all(e['diagnostico'].get('intervalo_entre_analisis_s', .12) is None or e['diagnostico'].get('intervalo_entre_analisis_s', .12) >= .12 for e in eventos))

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camara prohibida')))
        self.stack.enter_context(patch('serial.Serial', side_effect=AssertionError('Arduino prohibido')))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db, self.fotos = self.root/'base.pkl', self.root/'fotos'
        self.original = self.root/'original.jpg'
        self.original.write_bytes(cv2.imencode('.jpg', imagen(10))[1].tobytes())
        self.db.write_bytes(pickle.dumps({'nombres': ['Existente'], 'imagenes': [str(self.original)]}))
        self.previo = self.db.read_bytes()
        self.stack.enter_context(patch('configuracion.config.DB_PATH', str(self.db)))
        self.stack.enter_context(patch('configuracion.config.FOTOS_DIR', str(self.fotos)))
        self.cap = MagicMock()
        self.frame = np.zeros((400, 400, 3), np.uint8)
        self.frame[:, :100] = 255
        self.cap.read.return_value = True, self.frame
        self.cap.isOpened.return_value = True
        self.abrir = self.stack.enter_context(patch('hardware.camara.abrir_camara', return_value=self.cap))
        self.estimador = MagicMock()
        self.estimador.analizar.side_effect = lecturas()
        self.crear_estimador = self.stack.enter_context(patch('vision.orientacion.EstimadorOrientacion', return_value=self.estimador))
        self.factory = self.stack.enter_context(patch('interfaz.duplicados.VerificadorDuplicados'))
        self.verificador = self.factory.return_value
        self.verificador.vacia = False
        self.verificador.limites = {'Existente': [1, 2]}
        self.verificador.protocolo = PROTOCOLO
        self.verificador.comprobar.return_value = ('sin_coincidencia', None)
        self.stack.enter_context(patch('interfaz.trabajador.time.sleep'))
        reloj = iter(np.arange(0, 100, .2))
        self.stack.enter_context(patch('interfaz.trabajador.time.monotonic', side_effect=lambda: next(reloj)))
        self.guias = []
        def crear_g(t):
            g = Recorrido(t)
            self.guias.append(g)
            return g
        self.stack.enter_context(patch('interfaz.recorrido.Recorrido', side_effect=crear_g))
        self.puente = MagicMock()
        self.puente.detener.return_value = False
        self.puente.comandos = Queue()

    def ejecutar(self, **kwargs):
        from interfaz.trabajador import registrar
        registrar(self.puente, 'Nueva', 0, **kwargs)

    def test_prueba_captura_cinco_sin_guardar_ni_comprobar_identidad(self):
        self.ejecutar(prueba=True)
        self.factory.assert_not_called()
        self.assertEqual(self.db.read_bytes(), self.previo)
        self.assertFalse(self.fotos.exists())
        self.assertEqual(self.guias[0].fotos, [])
        eventos = [c.args[1] for c in self.puente.emitir.call_args_list if c.args[0] == 'recorrido']
        self.assertEqual(eventos[-1]['seleccionadas'], 5)
        self.assertEqual(eventos[-1]['guardadas'], 0)
        np.testing.assert_array_equal(self.estimador.analizar.call_args.args[0], self.frame)
        np.testing.assert_array_equal(self.puente.fotograma.call_args.args[0], cv2.flip(self.frame, 1))
        self.cap.release.assert_called_once()
        self.estimador.close.assert_called_once()

    def test_mismas_cinco_muestras_se_comprueban_y_guardan_con_control_final(self):
        from interfaz.registro import leer_base
        self.ejecutar()
        self.assertEqual(self.verificador.comprobar.call_count, 2)
        base = leer_base(self.db)
        self.assertEqual(base['nombres'], ['Existente']+['Nueva']*5)
        self.assertEqual(len(list(self.fotos.rglob('*.jpg'))), 5)
        self.assertEqual(self.guias[0].fotos, [])
        self.puente.emitir.assert_any_call('guardado', 'Nueva')

    def test_duplicado_o_duda_no_guarda_ni_pide_otro_recorrido(self):
        self.verificador.comprobar.return_value = ('duplicado', 'Existente')
        self.ejecutar()
        self.assertEqual(self.db.read_bytes(), self.previo)
        self.assertFalse(self.fotos.exists())
        self.assertEqual(self.guias[0].fotos, [])
        self.assertEqual(self.estimador.analizar.call_count, 30)
        self.puente.emitir.assert_any_call('aviso_registro', 'Este rostro coincide con un registro existente: Existente')

    def test_sin_politica_guiada_no_abre_camara_ni_estimador(self):
        self.verificador.protocolo = None
        self.ejecutar()
        self.abrir.assert_not_called()
        self.crear_estimador.assert_not_called()

    def test_cancelacion_descarta_muestras_y_cierra_recursos(self):
        self.puente.detener.side_effect = [False]*9+[True, True]
        self.ejecutar(prueba=True)
        self.assertEqual(self.guias[0].fotos, [])
        self.assertEqual(self.guias[0].ventana, [])
        self.assertEqual(self.db.read_bytes(), self.previo)
        self.assertFalse(self.fotos.exists())
        self.cap.release.assert_called_once()
        self.estimador.close.assert_called_once()

    def test_fallo_camara_limpia_y_cierra_estimador(self):
        self.cap.read.return_value = False, None
        with self.assertRaises(RuntimeError):
            self.ejecutar(prueba=True)
        self.assertEqual(self.guias[0].fotos, [])
        self.cap.release.assert_called_once()
        self.estimador.close.assert_called_once()
