from contextlib import ExitStack
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import evaluar_sface as e


class EvaluacionSFaceTests(unittest.TestCase):
    def test_letterbox_y_landmarks_en_coordenadas_originales(self):
        im, escalas = e.letterbox(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(im.shape, (320, 320, 3))
        cara = np.arange(15, dtype=np.float32)
        original = cara.copy()
        nueva = e.desescalar_cara(cara, escalas)
        np.testing.assert_array_equal(cara, original)
        np.testing.assert_allclose(nueva[:14], original[:14]*4)
        self.assertEqual(nueva[14], cara[14])

    def test_coseno_provisional_no_distancia_lbph_ni_porcentaje(self):
        sf = object.__new__(e.SFace)
        sf.reconocedor = MagicMock()
        for score, decision in ((.362, 'Desconocido'), (.363, 'Alexa'), (.8, 'Alexa')):
            sf.reconocedor.match.return_value = score
            candidato, resultado, nombre = sf.comparar(np.ones((1,128)), {'Alexa':[np.ones((1,128))]})
            self.assertEqual((candidato, resultado, nombre), ('Alexa', score, decision))

    def test_categorias_no_ocultan_errores_ni_confusiones(self):
        for tipo, decision, esperado in (
                ('registrada','Alexa','acierto'), ('registrada','Alison','confusion'),
                ('registrada','Desconocido','rechazo'),
                ('no_registrada','Alexa','aceptacion_desconocido'),
                ('no_registrada','Desconocido','rechazo_correcto'),
                ('no_registrada','Sin lectura','sin_lectura')):
            self.assertEqual(e.categoria('Alexa', tipo, decision), esperado)

    def test_rutas_solo_dentro_de_sesion(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                e.archivo_local(Path(td), '../foto.png')

    def test_modelos_reales_y_alineacion_sin_hardware(self):
        # Solo ejecución técnica: no depende de fotografías personales ni mide precisión.
        imagen=np.random.default_rng(2026).integers(0,256,(320,320,3),np.uint8)
        puntos=np.array([65,65,160,190,110,120,180,120,145,160,115,200,175,200,.99],np.float32)
        with patch('cv2.VideoCapture', side_effect=AssertionError('Camara prohibida')), \
             patch('serial.Serial', side_effect=AssertionError('Arduino prohibido')):
            sf = e.SFace()
            feature, error = sf.extraer_cara(imagen,puntos)
            self.assertEqual(error, '')
            self.assertEqual(feature.shape, (1,128))
            self.assertTrue(np.isfinite(feature).all())
            candidato, score, _ = sf.comparar(feature, {'solo_prueba_tecnica':[feature.copy()]})
            self.assertEqual(candidato, 'solo_prueba_tecnica')
            self.assertAlmostEqual(score, 1., places=5)
            self.assertIsNone(sf.extraer(np.zeros((720,1280,3),np.uint8))[0])

    def test_captura_y_prueba_aisladas_mismos_frames_sin_fugas(self):
        with tempfile.TemporaryDirectory() as td, ExitStack() as st:
            raiz = Path(td)
            st.enter_context(patch.object(e, 'ROOT', raiz))
            sf, lb = MagicMock(), MagicMock()
            sf.extraer.return_value = (np.ones((1,128), np.float32), '')
            lb.extraer.return_value = (np.zeros((200,200), np.uint8), '')
            sf.comparar.return_value = ('Alexa', .7, 'Alexa')
            st.enter_context(patch.object(e, 'SFace', return_value=sf))
            st.enter_context(patch.object(e, 'RecorteLBPH', return_value=lb))
            st.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camara real prohibida')))
            st.enter_context(patch('serial.Serial', side_effect=AssertionError('Arduino prohibido')))
            st.enter_context(patch.object(e.cv2, 'imshow'))
            st.enter_context(patch.object(e.cv2, 'destroyAllWindows'))
            st.enter_context(patch.object(e.cv2, 'waitKey', return_value=32))
            st.enter_context(patch.object(e.time, 'monotonic', side_effect=itertools.count(0, 3)))
            activo = MagicMock()
            activo.predict.return_value = (0, 65.)
            st.enter_context(patch('datos.personas.entrenar_reconocedor', return_value=(activo, {0:'Alexa'})))
            cap = MagicMock()
            cap.isOpened.return_value = True
            frames = [np.random.default_rng(i).integers(0,256,(240,320,3),np.uint8) for i in range(4)]
            cap.read.side_effect = [(True, f) for f in frames]
            abrir = st.enter_context(patch('hardware.camara.abrir_camara', return_value=cap))
            args = NS(accion='capturar', persona='Alexa', tipo='registrada', cantidad=2,
                      cam=0, intervalo=1., galeria_parcial=True)
            e.capturar(args)
            args.accion = 'probar'
            e.capturar(args)
            abrir.assert_called_with(0)
            self.assertEqual(cap.release.call_count, 2)
            refs = list((raiz/'referencias').glob('*/capturas.csv'))
            evaluaciones = list((raiz/'sesiones').glob('*/capturas.csv'))
            self.assertEqual(len(refs), 1)
            self.assertEqual(len(evaluaciones), 1)
            h_ref = {f['sha256'] for f in e.csv_leer(refs[0])}
            h_eval = {f['sha256'] for f in e.csv_leer(evaluaciones[0])}
            self.assertFalse(h_ref & h_eval)
            resultados = e.csv_leer(evaluaciones[0].parent/'resultados.csv')
            self.assertEqual(len(resultados), 6)
            self.assertEqual({f['metodo'] for f in resultados}, {'SFace','LBPH_nuevo','LBPH_activo'})
            # Cada consulta SF/LBPH recibe el mismo frame sin textos.
            for llamada_sf, llamada_lb in zip(sf.extraer.call_args_list, lb.extraer.call_args_list):
                self.assertIs(llamada_sf.args[0], llamada_lb.args[0])
            # Duplicar un manifiesto de referencias no se permite silenciosamente.
            duplicado = raiz/'referencias'/'duplicado'
            duplicado.mkdir()
            for p in refs[0].parent.iterdir():
                (duplicado/p.name).write_bytes(p.read_bytes())
            with self.assertRaisesRegex(ValueError, 'repetida'):
                e.cargar_galeria(sf, lb, parcial=True)

    def test_modelo_alterado_no_se_carga(self):
        with tempfile.TemporaryDirectory() as td, patch.object(e, 'MODELOS', Path(td)):
            with self.assertRaisesRegex(ValueError, 'hash incorrecto'):
                e.verificar_modelos()

    def test_informe_separa_ausencia_y_rechazo(self):
        filas = [dict(persona='externa01', tipo='no_registrada', metodo='SFace',
                      categoria=c, total_ms=10) for c in ('sin_lectura','rechazo_correcto','aceptacion_desconocido')]
        informe = e.resumen(filas)
        self.assertIn('externa01/no_registrada', informe)
        self.assertIn('PROVISIONAL', informe)
        self.assertIn('no se cuenta como rechazo correcto', informe)


if __name__ == '__main__':
    unittest.main()
