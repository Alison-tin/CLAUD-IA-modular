from contextlib import ExitStack
from queue import Queue
import threading
import unittest
from unittest.mock import MagicMock, patch, call

from interfaz.conexion import SerialObservado, preparar_conexion
from interfaz.trabajador import Puente
from web_local.control import Control


class FisicoTests(unittest.TestCase):
    def setUp(self):
        self.guardia_serial = patch("serial.Serial", side_effect=AssertionError("Puerto real prohibido"))
        self.guardia_camara = patch("cv2.VideoCapture", side_effect=AssertionError("Camara real prohibida"))
        self.guardia_serial.start()
        self.guardia_camara.start()

    def tearDown(self):
        self.guardia_serial.stop()
        self.guardia_camara.stop()

    def puente(self):
        puente = Puente(Queue(), Queue(), Queue(), threading.Event())
        puente.video_limpio = True
        return puente

    def test_puertos_solo_enumera_sin_abrir_serie(self):
        puerto = MagicMock(device="COM7", description="Arduino USB")
        with patch("serial.tools.list_ports.comports", return_value=[puerto]):
            self.assertEqual(Control().puertos(), [{"puerto": "COM7", "descripcion": "Arduino USB"}])

    def test_confirmacion_requerida_y_puerto_desaparecido(self):
        control = Control()
        with self.assertRaises(ValueError):
            control.iniciar("sistema", modo="fisico")
        with patch.object(control, "puertos", return_value=[]):
            with self.assertRaises(ValueError):
                control.iniciar("sistema", modo="fisico", puerto="COM7", confirmar_fisico=True)
        self.assertIsNone(control.proceso)

    def test_demo_no_conecta_y_fallo_no_se_presenta_como_conectado(self):
        puente = self.puente()
        arduino = MagicMock(demo=True, ser=None)
        preparar_conexion(puente, arduino, True)
        self.assertEqual(puente.eventos.get()[1]["estado"], "demo")
        with self.assertRaises(RuntimeError):
            preparar_conexion(puente, arduino, False)
        self.assertEqual(puente.eventos.get()[1]["estado"], "error")
        arduino.ir_a_centro.assert_not_called()

    def test_observador_conserva_bytes_sin_reintentos(self):
        original = MagicMock(port="COM7", is_open=True)
        datos = b"PCA,2,12,600,2\n"
        original.write.return_value = len(datos)
        informar, parar = MagicMock(), threading.Event()
        serial = SerialObservado(original, informar, parar)
        self.assertEqual(serial.write(datos), len(datos))
        original.write.assert_called_once_with(datos)
        informar.assert_not_called()
        original.write.side_effect = OSError("USB simulado")
        with self.assertRaises(OSError):
            serial.write(datos)
        self.assertEqual(original.write.call_count, 2)
        self.assertTrue(parar.is_set())
        self.assertEqual(informar.call_args.args[1]["estado"], "error")

    def test_cierre_y_escritura_parcial_se_notifican(self):
        original = MagicMock(port="COM7", is_open=False)
        informar, parar = MagicMock(), threading.Event()
        serial = SerialObservado(original, informar, parar)
        self.assertFalse(serial.is_open)
        self.assertEqual(informar.call_args.args[1]["estado"], "desconectado")
        self.assertTrue(parar.is_set())
        original.is_open = True
        original.write.return_value = 1
        informar.reset_mock()
        SerialObservado(original, informar, parar).write(b"123")
        self.assertEqual(informar.call_args.args[1]["estado"], "error")

    def test_estado_fisico_y_error_no_es_ocultado_por_camara(self):
        control = Control()
        control.estado.update(modo="fisico", fase="iniciando")
        control.evento("arduino", {"estado": "conectado", "puerto": "COM7"})
        self.assertEqual(control.ver()["arduino"], "conectado")
        self.assertIn("sin confirmación", control.ver()["conexion"])
        control.evento("arduino", {"estado": "error", "puerto": "COM7"})
        control.evento("camara", "Activa")
        self.assertEqual(control.ver()["fase"], "error")
        self.assertEqual(control.ver()["arduino"], "error")

    def ejecutar_hasta_detector(self, arduino):
        from aplicacion.sistema import ejecutar
        with ExitStack() as stack:
            stack.enter_context(patch("aplicacion.sistema.Arduino", return_value=arduino))
            stack.enter_context(patch("aplicacion.sistema.descargar_modelos"))
            stack.enter_context(patch("aplicacion.sistema.threading.Thread"))
            stack.enter_context(patch("aplicacion.sistema.cargar_db", return_value={"nombres": [], "imagenes": []}))
            stack.enter_context(patch("aplicacion.sistema.time.sleep"))
            stack.enter_context(patch("aplicacion.sistema.mp_vision.FaceDetector.create_from_options", side_effect=RuntimeError("fin simulado antes de camara")))
            with self.assertRaises(RuntimeError):
                ejecutar(demo=False, port="COM7", interfaz=self.puente())

    def test_arranque_existente_se_invoca_una_sola_vez_con_arduino_simulado(self):
        serial = MagicMock(port="COM7", is_open=True)
        arduino = MagicMock(demo=False, ser=serial)
        self.ejecutar_hasta_detector(arduino)
        arduino.ir_a_centro.assert_called_once_with()
        arduino.parpado.assert_called_once_with(abrir=True)
        self.assertEqual(arduino.mover_maxilar.call_args_list,
                         [call(abrir=True, vel=5), call(abrir=False, vel=5)])
        serial.close.assert_called_once()

    def test_fallo_de_conexion_aborta_antes_de_ordenes_de_arranque(self):
        arduino = MagicMock(demo=True, ser=None)
        self.ejecutar_hasta_detector(arduino)
        arduino.ir_a_centro.assert_not_called()
        arduino.parpado.assert_not_called()
        arduino.mover_maxilar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
