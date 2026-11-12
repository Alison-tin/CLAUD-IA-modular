from contextlib import ExitStack
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import vision.galeria_sface as g
from vision.motor_sface import MotorSFace, decidir_sface, sin_resultado
from web_local.control import Control
from evaluar_sface import csv_guardar


class GaleriasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root/'base.pkl'
        self.evaluacion = self.root/'evaluacion'
        self.destino = self.root/'galerias'
        self.base(['Alexa','daniela'])
        self.referencia('Alexa', 1)
        self.referencia('daniela', 2)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(g, 'verificar_modelos'))
        self.motor = self.stack.enter_context(patch.object(g, 'SFace')).return_value
        self.motor.extraer.return_value = (np.ones((1,128),np.float32), '')

    def base(self, nombres):
        self.db.write_bytes(pickle.dumps({'nombres':nombres, 'imagenes':['sin_cambio']*len(nombres)}))

    def referencia(self, persona, semilla):
        carpeta = self.evaluacion/'referencias'/(persona+'_sesion')
        carpeta.mkdir(parents=True)
        frame = np.random.default_rng(semilla).integers(0,256,(120,120,3),np.uint8)
        cv2.imwrite(str(carpeta/'captura_001.png'), frame)
        fila = dict(persona=persona, archivo='captura_001.png', tipo='registrada', uso='referencias',
                    sesion=carpeta.name, sha256=g.digest((carpeta/'captura_001.png').read_bytes()))
        csv_guardar(carpeta/'capturas.csv', [fila])

    def preparar(self):
        return g.preparar(self.db, self.evaluacion, self.destino)

    def cargar(self):
        return g.cargar(self.db, self.evaluacion, self.destino)

    def test_version_y_etiquetas_sin_modificar_originales(self):
        protegidos = {p:p.read_bytes() for p in self.evaluacion.rglob('*') if p.is_file()}
        base = self.db.read_bytes()
        p = self.preparar()
        grupos, _, version = self.cargar()
        self.assertEqual(set(grupos), {'Alexa','daniela'})
        self.assertIn(version, p.name)
        self.assertEqual(self.preparar(), p)
        self.assertEqual(self.db.read_bytes(), base)
        self.assertTrue(all(p.read_bytes()==data for p,data in protegidos.items()))

    def test_retirada_invalida_y_nueva_version_excluye_identidad(self):
        vieja = self.preparar()
        self.base(['Alexa'])
        with self.assertRaises(g.GaleriaInvalida):
            self.cargar()
        nueva = self.preparar()
        self.assertNotEqual(vieja, nueva)
        self.assertEqual(set(self.cargar()[0]), {'Alexa'})
        self.assertTrue((self.evaluacion/'referencias/daniela_sesion/captura_001.png').exists())

    def test_alta_sin_referencias_no_habilita_sface(self):
        self.preparar()
        self.base(['Alexa','daniela','Alison'])
        with self.assertRaisesRegex(g.GaleriaInvalida, 'Faltan referencias'):
            self.preparar()

    def test_hash_invalido_y_modelos_ausentes(self):
        with patch.object(g, 'verificar_modelos', side_effect=ValueError('Modelo ausente')):
            with self.assertRaisesRegex(g.GaleriaInvalida, 'Modelo ausente'):
                self.cargar()
        self.preparar()
        p = self.evaluacion/'referencias/Alexa_sesion/captura_001.png'
        p.write_bytes(b'alterado')
        with self.assertRaisesRegex(g.GaleriaInvalida, 'Hash'):
            self.cargar()

    def test_prueba_no_puede_ser_referencia(self):
        referencia = self.evaluacion/'referencias/Alexa_sesion/captura_001.png'
        carpeta = self.evaluacion/'sesiones/prueba'
        carpeta.mkdir(parents=True)
        (carpeta/'foto.png').write_bytes(referencia.read_bytes())
        csv_guardar(carpeta/'capturas.csv', [dict(archivo='foto.png', sesion='prueba',
                    uso='evaluacion', sha256=g.digest(referencia.read_bytes()))])
        with self.assertRaisesRegex(g.GaleriaInvalida, 'tambien es captura'):
            self.preparar()

    def test_npz_alterado_no_carga(self):
        p = self.preparar()
        (p/'embeddings.npz').write_bytes(b'alterado')
        with self.assertRaisesRegex(g.GaleriaInvalida, 'Integridad'):
            self.cargar()


class MotorTests(unittest.TestCase):
    def motor(self, carpeta):
        motor = object.__new__(MotorSFace)
        motor.db_path = carpeta/'base.pkl'
        motor.db_path.write_bytes(b'base')
        motor.base_sha = g.digest(b'base')
        motor.modelo = MagicMock()
        motor.galeria = {'Alexa':[np.ones((1,128),np.float32)]}
        cara = np.array([10,10,80,80]+[30,30]*5+[.99],np.float32)
        motor.modelo.detectar.return_value = [cara]
        motor.modelo.extraer_cara.return_value = (np.ones((1,128),np.float32),'')
        motor.modelo.comparar.return_value = ('Alexa',.5,'Alexa')
        return motor, cara

    def test_umbral_sin_porcentaje_y_limpieza_de_rechazo(self):
        for score in (.363,.8):
            r = decidir_sface('Alexa',score,{'Alexa'})
            self.assertEqual(r['nombre'],'Alexa')
            self.assertEqual(r['similitud_coseno'],score)
            self.assertIsNone(r['similitud_estimada'])
        for score in (.2,float('nan'),float('inf'),-2):
            r = decidir_sface('Alexa',score,{'Alexa'})
            self.assertIsNone(r['similitud_coseno'])
            self.assertNotEqual(r['nombre'],'Alexa')
        self.assertIsNone(decidir_sface('retirada',.9,{'Alexa'})['nombre'])

    def test_caja_y_landmarks_misma_persona_cero_varios_y_cambio_base(self):
        with tempfile.TemporaryDirectory() as td:
            motor, cara = self.motor(Path(td))
            frame = np.zeros((120,120,3),np.uint8)
            self.assertEqual(motor.reconocer(frame,[])['estado_reconocimiento'],'sin_rostro')
            self.assertEqual(motor.reconocer(frame,[(10,10,80,80)]*2)['estado_reconocimiento'],'varios_rostros')
            motor.modelo.detectar.return_value = [cara,cara]
            self.assertEqual(motor.reconocer(frame,[(10,10,80,80)])['estado_reconocimiento'],'varios_rostros')
            motor.modelo.detectar.return_value = [cara]
            self.assertEqual(motor.reconocer(frame,[(300,300,80,80)])['estado_reconocimiento'],'sin_correspondencia')
            motor.modelo.extraer_cara.assert_not_called()
            self.assertEqual(motor.reconocer(frame,[(10,10,80,80)])['nombre'],'Alexa')
            llamada = motor.modelo.extraer_cara.call_args.args
            self.assertIs(llamada[0],frame)
            self.assertIs(llamada[1],cara)
            motor.db_path.write_bytes(b'cambio')
            with self.assertRaises(g.GaleriaInvalida):
                motor.reconocer(frame,[(10,10,80,80)])


class ControlSFaceTests(unittest.TestCase):
    def test_reinicio_servidor_restaura_sface_y_lbph_sigue_disponible(self):
        anterior = Control()
        with self.assertRaises(ValueError):
            anterior.seleccionar_motor('lbph')
        self.assertEqual(anterior.ver()['motor_seleccionado'],'sface')
        for _ in range(2):
            nuevo = Control()
            self.assertEqual(nuevo.ver()['motor_seleccionado'],'sface')
            with patch('vision.galeria_sface.cargar') as galeria, \
                 patch.object(nuevo.ctx,'Process') as proceso, \
                 patch('web_local.control.threading.Thread'), \
                 patch('cv2.VideoCapture',side_effect=AssertionError('Sin camara')), \
                 patch('serial.Serial',side_effect=AssertionError('Sin Arduino')):
                nuevo.iniciar('sistema',modo='demo')
                galeria.assert_called_once()
                self.assertEqual(proceso.call_args.kwargs['args'][1]['motor'],'sface')
                self.assertTrue(proceso.call_args.kwargs['args'][1]['demo'])

    def test_selector_inicial_y_lanzador_habitual_coinciden_con_servidor(self):
        root=Path(__file__).resolve().parents[1]
        html=(root/'web_local/static/index.html').read_text(encoding='utf-8')
        self.assertIn('id="recognition-engine" type="hidden" value="sface"',html)
        self.assertNotIn('<select id="recognition-engine"',html)
        self.assertIn('s.motor_seleccionado || "sface"', (root/'web_local/static/app.js').read_text(encoding='utf-8'))
        cmd=(root/'Iniciar_CLAUD_IA_Web.cmd').read_text()
        self.assertIn('"%~dp0.venv\\Scripts\\python.exe" -B "%~dp0web.py" --puerto 8765',cmd)
        self.assertNotIn('web_emociones_prueba.py',cmd)

    def test_selector_solo_detenido_y_motor_real(self):
        c = Control()
        self.assertEqual(c.ver()['motor_seleccionado'],'sface')
        self.assertIsNone(c.ver()['motor_activo'])
        with patch('vision.galeria_sface.cargar'), patch.object(c.ctx,'Process') as proceso, \
             patch('web_local.control.threading.Thread'):
            c.iniciar('sistema')
            self.assertEqual(proceso.call_args.kwargs['args'][1]['motor'],'sface')
            self.assertIsNone(c.ver()['motor_activo'])
            c.evento('motor_reconocimiento',{'motor':'sface','version':'v1'})
            self.assertEqual(c.ver()['motor_activo'],'sface')
            with self.assertRaises(ValueError):
                c.seleccionar_motor('lbph')
            c.detener()
            self.assertIsNone(c.ver()['motor_activo'])

    def test_error_preflight_no_crea_proceso_ni_fallback(self):
        c = Control(); c.seleccionar_motor('sface')
        with patch('vision.galeria_sface.cargar',side_effect=g.GaleriaInvalida('Galeria ausente')), \
             patch.object(c.ctx,'Process') as proceso, patch('cv2.VideoCapture') as camara, patch('serial.Serial') as arduino:
            with self.assertRaisesRegex(ValueError,'Galeria ausente'):
                c.iniciar('sistema')
            proceso.assert_not_called(); camara.assert_not_called(); arduino.assert_not_called()
        self.assertEqual(c.ver()['motor_seleccionado'],'sface')

    def test_limpieza_por_relevo_ausencia_desactivar_parar_error(self):
        c = Control(); c.seleccionar_motor('sface')
        c.estado.update(fase='activo',motor_activo='sface')
        def aceptar():
            c.evento('reconocimiento',decidir_sface('Alexa',.5,{'Alexa'}))
            self.assertEqual(c.ver()['nombre'],'Alexa')
            self.assertEqual(c.ver()['similitud_coseno'],.5)
            self.assertIsNone(c.ver()['similitud_estimada'])
        aceptar()
        c.evento('reconocimiento',decidir_sface('Alexa',.2,{'Alexa'}))
        self.assertEqual(c.ver()['nombre'],'Desconocido')
        self.assertIsNone(c.ver()['similitud_coseno'])
        c.evento('reconocimiento',decidir_sface('daniela',.6,{'daniela'}))
        self.assertEqual(c.ver()['nombre'],'daniela')
        for estado in ('varios_rostros','error','sin_rostro'):
            aceptar(); c.evento('reconocimiento',sin_resultado(estado))
            self.assertIsNone(c.ver()['nombre']); self.assertIsNone(c.ver()['similitud_coseno'])
        aceptar()
        c.evento('deteccion',{'rostros':0,'gesto':None,'deepface':{'ok':True}})
        self.assertIsNone(c.ver()['similitud_coseno'])
        aceptar(); c.opciones({'reconocimiento':False})
        self.assertIsNone(c.ver()['nombre']); self.assertIsNone(c.ver()['similitud_coseno'])
        c.opciones({'reconocimiento':True}); aceptar()
        c.evento('error','Fallo SFace')
        self.assertIsNone(c.ver()['nombre']); self.assertIsNone(c.ver()['similitud_coseno'])
        c.estado['fase']='activo'; aceptar()
        c.proceso=MagicMock(); c.parada=MagicMock(); c.detener()
        c.evento('reconocimiento',decidir_sface('Alexa',.5,{'Alexa'}))
        self.assertIsNone(c.ver()['nombre']); self.assertIsNone(c.ver()['similitud_coseno'])


class SistemaSFaceTests(unittest.TestCase):
    def test_mismo_motor_web_y_terminal_no_usa_recorte_lbph(self):
        from aplicacion import sistema
        resultados = []
        for web in (False,True):
            with ExitStack() as st:
                frame = np.zeros((240,320,3),np.uint8)
                cap = MagicMock(); cap.isOpened.return_value=True
                cap.get.side_effect=lambda p: 320 if p==3 else 240 if p==4 else 30
                cap.read.side_effect=[(True,frame.copy())]*6+[(False,None)]
                cara, mano = MagicMock(), MagicMock()
                caja = NS(origin_x=60,origin_y=50,width=100,height=100)
                cara.detect.side_effect=[NS(detections=[NS(bounding_box=caja)])]*5+[NS(detections=[])]
                mano.detect.return_value=NS(hand_landmarks=[])
                sf = MagicMock(version='test')
                sf.reconocer.side_effect=[decidir_sface('Alexa',.5,{'Alexa'}),
                    decidir_sface('Alexa',.2,{'Alexa'}),decidir_sface('daniela',.6,{'daniela'}),
                    RuntimeError('Fallo simulado'),sin_resultado('varios_rostros')]
                st.enter_context(patch.object(sistema,'MotorSFace',return_value=sf))
                recorte = st.enter_context(patch.object(sistema,'extraer_rostro',side_effect=AssertionError('No LBPH en SFace')))
                st.enter_context(patch.object(sistema,'entrenar_reconocedor',side_effect=AssertionError('No fallback')))
                st.enter_context(patch.object(sistema,'abrir_camara',return_value=cap))
                st.enter_context(patch('cv2.VideoCapture',side_effect=AssertionError('No hardware')))
                st.enter_context(patch('serial.Serial',side_effect=AssertionError('No hardware')))
                st.enter_context(patch.object(sistema,'Arduino',return_value=MagicMock(ser=None)))
                st.enter_context(patch.object(sistema,'Reproductor'))
                st.enter_context(patch.object(sistema,'descargar_modelos'))
                st.enter_context(patch.object(sistema.threading,'Thread'))
                st.enter_context(patch.object(sistema.time,'sleep'))
                st.enter_context(patch.object(sistema,'deepface_estado',return_value={'ok':False,'intentado':True}))
                st.enter_context(patch.object(sistema,'cargar_db',return_value={'nombres':['Alexa'],'imagenes':['a']}))
                st.enter_context(patch.object(sistema.mp_vision.FaceDetector,'create_from_options',return_value=cara))
                st.enter_context(patch.object(sistema.mp_vision.HandLandmarker,'create_from_options',return_value=mano))
                for nombre in ('namedWindow','setWindowProperty','imshow','destroyAllWindows'):
                    st.enter_context(patch.object(sistema.cv2,nombre))
                st.enter_context(patch.object(sistema.cv2,'waitKey',return_value=-1))
                put=st.enter_context(patch.object(sistema,'put'))
                puente=MagicMock(); puente.detener.return_value=False
                puente.opciones.return_value={'emociones':False,'gestos':False}
                impresiones=st.enter_context(patch('builtins.print'))
                sistema.ejecutar(demo=True,motor='sface',interfaz=puente if web else None)
                salida='\n'.join(str(c.args[0]) for c in impresiones.call_args_list if c.args)
                self.assertNotIn('[SFace]',salida)
                self.assertIn('SFace: Fallo simulado',salida)
                textos=[c.args[1] for c in put.call_args_list if c.args[1] in ('Sin lectura','Desconocido') or c.args[1].startswith(('Alexa -','daniela -'))]
                self.assertEqual(textos,['Alexa - Similitud 50,0 %','Desconocido','daniela - Similitud 60,0 %','Sin lectura','Sin lectura'])
                if web:
                    lecturas=[c.args[1] for c in puente.emitir.call_args_list if c.args[0]=='reconocimiento']
                    self.assertEqual([r['nombre'] for r in lecturas],['Alexa','Desconocido','daniela',None,None,None])
                recorte.assert_not_called()
                resultados.append(textos)
        self.assertEqual(*resultados)
