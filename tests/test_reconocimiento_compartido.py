from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch
import unittest
import numpy as np


class FlujoCompartido(unittest.TestCase):
    def test_terminal_web_sin_retencion_con_rechazo_error_y_relevo(self):
        from aplicacion import sistema
        salidas = []
        for web in (False, True):
            with self.subTest(web=web), ExitStack() as st:
                frame = np.zeros((240, 320, 3), np.uint8)
                cap = MagicMock()
                cap.isOpened.return_value = True
                cap.get.side_effect = lambda p: 320 if p == 3 else 240 if p == 4 else 30
                cap.read.side_effect = [(True, frame.copy()) for _ in range(7)] + [(False, None)]
                cara, mano = MagicMock(), MagicMock()
                caja = NS(origin_x=60, origin_y=50, width=100, height=100)
                cara.detect.side_effect = [NS(detections=[NS(bounding_box=caja)]) for _ in range(6)] + [NS(detections=[])]
                mano.detect.return_value = NS(hand_landmarks=[])
                modelo = MagicMock()
                modelo.predict.side_effect = [(0, 55), (1, 80), (1, 54), RuntimeError('Fallo simulado'), (0, 55)]
                gris = np.zeros((200, 200), np.uint8)
                # Se mide cada fotograma. En 4 falla el recorte y en 5 predict.
                extraer = st.enter_context(patch.object(sistema, 'extraer_rostro', side_effect=[gris, gris, gris, None, gris, gris]))
                st.enter_context(patch.object(sistema, 'abrir_camara', return_value=cap))
                st.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Hardware prohibido')))
                st.enter_context(patch('serial.Serial', side_effect=AssertionError('Hardware prohibido')))
                st.enter_context(patch.object(sistema, 'Arduino', return_value=MagicMock(ser=None)))
                st.enter_context(patch.object(sistema, 'Reproductor'))
                st.enter_context(patch.object(sistema, 'descargar_modelos'))
                st.enter_context(patch.object(sistema.threading, 'Thread'))
                st.enter_context(patch.object(sistema.time, 'sleep'))
                st.enter_context(patch.object(sistema, 'deepface_estado', return_value={'ok':False, 'intentado':True}))
                st.enter_context(patch.object(sistema, 'cargar_db', return_value={'nombres':['Alexa','daniela'], 'imagenes':['a','b']}))
                st.enter_context(patch.object(sistema, 'entrenar_reconocedor', return_value=(modelo, {0:'Alexa',1:'daniela'})))
                st.enter_context(patch.object(sistema.mp_vision.FaceDetector, 'create_from_options', return_value=cara))
                st.enter_context(patch.object(sistema.mp_vision.HandLandmarker, 'create_from_options', return_value=mano))
                for nombre in ('namedWindow', 'setWindowProperty', 'imshow', 'destroyAllWindows'):
                    st.enter_context(patch.object(sistema.cv2, nombre))
                st.enter_context(patch.object(sistema.cv2, 'waitKey', return_value=-1))
                put = st.enter_context(patch.object(sistema, 'put'))
                puente = MagicMock()
                puente.detener.return_value = False
                puente.opciones.return_value = {'emociones':False, 'gestos':False}
                sistema.ejecutar(demo=True, interfaz=puente if web else None)
                textos = [c.args[1] for c in put.call_args_list
                          if c.args[1] in ('Sin lectura', 'Desconocido') or c.args[1].startswith(('Alexa (','daniela ('))]
                esperados = ['Alexa (45%)','Desconocido','daniela (46%)',
                             'Sin lectura','Sin lectura','Alexa (45%)']
                self.assertEqual(textos, esperados)
                if web:
                    lecturas = [c.args[1] for c in puente.emitir.call_args_list if c.args[0]=='reconocimiento']
                    self.assertEqual([l['nombre'] for l in lecturas],
                                     ['Alexa','Desconocido','daniela',None,None,'Alexa',None])
                for llamada in extraer.call_args_list:
                    np.testing.assert_array_equal(llamada.args[0], frame)
                salidas.append(textos)
        self.assertEqual(*salidas)
