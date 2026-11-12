from contextlib import ExitStack
from pathlib import Path
import pickle
from queue import Queue
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from interfaz.evaluacion_sface import EvaluacionSFace, analizar_lecturas, validar_participante_sface
from interfaz.recorrido import Recorrido
from tests.test_recorrido import lecturas


def participante(tipo='registrada', identificador='Alison'):
    return dict(tipo=tipo, identificador=identificador, acepta_conservar=True, motor='sface',
                nombre_propuesto='Alison_Prueba' if tipo=='registrada' else '')


class OriginalesTests(unittest.TestCase):
    def test_candidata_sin_original_cancela_y_limpia_sin_captura_extra(self):
        g=Recorrido(0,conservar_originales=True)
        for k,o in enumerate(lecturas()[:6]):
            g.actualizar(o,(k+1)*.2)
        self.assertTrue(g.cancelado)
        self.assertFalse(g.completo)
        self.assertEqual(g.fotos,[])
        self.assertEqual(g.originales,[])
        self.assertEqual(g.ventana,[])

    def test_original_corresponde_a_candidata_mas_nitida_no_al_ultimo_frame(self):
        g = Recorrido(0,conservar_originales=True)
        observaciones = lecturas()[:6]
        for k,o in enumerate(observaciones):
            o.nitidez = 200 if k==2 else 100
            o.original_bgr = np.full((12,15,3),k,np.uint8)
            g.actualizar(o,(k+1)*.2)
            o.original_bgr[:]=99  # El historial debe tener su propia copia.
        self.assertEqual(len(g.fotos),1)
        self.assertEqual(len(g.originales),1)
        self.assertTrue((g.originales[0]==2).all())
        g.cancelar()
        self.assertEqual(g.originales,[])

    def test_recorrido_normal_no_retiene_color(self):
        g = Recorrido(0)
        for k,o in enumerate(lecturas()):
            o.original_bgr=np.zeros((12,15,3),np.uint8)
            g.actualizar(o,(k+1)*.2)
        self.assertTrue(g.completo)
        self.assertEqual(len(g.fotos),5)
        self.assertEqual(g.originales,[])


class EvaluacionTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db=self.root/'base.pkl'
        self.db.write_bytes(pickle.dumps(dict(nombres=['Alexa','Alison','daniela'],imagenes=['a','b','c'])))
        self.previo=self.db.read_bytes()
        self.destino=self.root/'evaluacion_duplicados_sface'
        self.galeria={n:[np.full((1,128),i,np.float32)] for i,n in enumerate(('Alexa','Alison','daniela'),1)}
        self.stack.enter_context(patch('interfaz.evaluacion_sface.cargar',return_value=(self.galeria,'base','galeria')))
        self.stack.enter_context(patch('interfaz.evaluacion_sface.inventario',return_value=([], 'base','galeria')))
        self.sf=self.stack.enter_context(patch('interfaz.evaluacion_sface.SFace')).return_value
        self.sf.extraer.return_value=(np.ones((1,128),np.float32),'')
        self.sf.reconocedor.match.side_effect=lambda f, ref, tipo: {1:.25,2:.65,3:.23}[int(ref[0,0])]
        self.stack.enter_context(patch('cv2.VideoCapture',side_effect=AssertionError('No camara real')))
        self.stack.enter_context(patch('serial.Serial',side_effect=AssertionError('No Arduino')))
        self.frames=[np.random.default_rng(k).integers(0,256,(160,200,3),np.uint8) for k in range(5)]

    def evaluador(self,datos=None):
        return EvaluacionSFace(datos or participante(),self.db,self.destino)

    def test_cero_y_varios_rostros_no_alinean_ni_fuerzan_identidad(self):
        from evaluar_sface import SFace
        for caras in ([],[np.zeros(15),np.zeros(15)]):
            with self.subTest(caras=len(caras)):
                modelo=object.__new__(SFace)
                modelo.detectar=MagicMock(return_value=caras)
                modelo.extraer_cara=MagicMock(side_effect=AssertionError('No alinear cara ambigua'))
                e=self.evaluador();e.modelo=modelo
                r=e.completar(self.frames,lambda:False)
                modelo.extraer_cara.assert_not_called()
                self.assertEqual(modelo.detectar.call_count,5)
                self.assertEqual(r['analisis']['validas'],0)
                self.assertEqual(len(r['analisis']['sin_lectura']),5)
                self.assertIsNone(r['analisis']['candidata_repetida'])
                self.assertIsNone(r['analisis']['decision_duplicado'])
                self.assertTrue(all(l['primera'] is None and str(len(caras))+' rostros' in l['error'] for l in r['lecturas']))
                self.assertEqual(self.db.read_bytes(),self.previo)

    def test_capturas_invalidas_y_cantidad_incompleta_no_guardan(self):
        invalidas=[None,np.zeros((0,200,3),np.uint8),np.zeros((200,200,4),np.uint8),
                   np.zeros((200,200,3),np.float32),np.zeros((200,200,3),np.uint8)]
        e=self.evaluador()
        for invalida in invalidas:
            with self.subTest(tipo=type(invalida)):
                with self.assertRaises(ValueError):
                    e.completar([invalida]+self.frames[1:],lambda:False)
                self.assertFalse(self.destino.exists())
        with self.assertRaisesRegex(ValueError,'cinco originales'):
            e.completar(self.frames[:4],lambda:False)

    def test_error_codificacion_y_reutilizacion_no_guardan(self):
        e=self.evaluador()
        with patch('interfaz.evaluacion_sface.cv2.imencode',return_value=(False,None)):
            with self.assertRaisesRegex(ValueError,'preparar'):
                e.completar(self.frames,lambda:False)
        with self.assertRaisesRegex(ValueError,'reutilizo'):
            e.completar([self.frames[0]]*5,lambda:False)
        from vision.galeria_sface import digest
        e.filas_referencia=[{'sha256':digest(cv2.imencode('.png',self.frames[0])[1].tobytes())}]
        with self.assertRaisesRegex(ValueError,'reutilizo una referencia'):
            e.completar(self.frames,lambda:False)
        self.assertFalse(self.destino.exists())

    def test_error_extraccion_es_sin_lectura_y_error_match_aborta_sin_guardar(self):
        self.sf.extraer.side_effect=[cv2.error('fallo extraccion')]+[(np.ones((1,128),np.float32),'')]*4
        r=self.evaluador().completar(self.frames,lambda:False)
        self.assertEqual(r['analisis']['validas'],4)
        self.assertIn('fallo extraccion',r['lecturas'][0]['error'])
        self.assertEqual(self.sf.extraer.call_count,5)
        e=self.evaluador();e.destino=self.root/'fallo_match'
        self.sf.extraer.side_effect=None
        self.sf.reconocedor.match.side_effect=cv2.error('fallo match')
        with self.assertRaises(cv2.error):
            e.completar(self.frames,lambda:False)
        self.assertFalse(e.destino.exists())
        self.assertEqual(self.db.read_bytes(),self.previo)

    def test_fallos_guardado_y_rename_limpian_solo_su_temporal(self):
        self.destino.mkdir()
        vecino=self.destino/'ensayo_previo';vecino.mkdir()
        (vecino/'conservar.txt').write_bytes(b'no tocar')
        abrir=Path.open
        e=self.evaluador()
        for fallo in ('png','csv','rename'):
            with self.subTest(fallo=fallo),ExitStack() as st:
                def abrir_seguro(p,*args,**kwargs):
                    if p.parent.name.startswith('.incompleto_recorrido_') and p.name==('derecha.png' if fallo=='png' else 'puntuaciones.csv'):
                        raise OSError('fallo guardado simulado')
                    return abrir(p,*args,**kwargs)
                if fallo=='rename':
                    st.enter_context(patch.object(Path,'rename',side_effect=OSError('fallo rename simulado')))
                else:
                    st.enter_context(patch.object(Path,'open',abrir_seguro))
                with self.assertRaises(OSError):
                    e.completar(self.frames,lambda:False)
                self.assertEqual(list(self.destino.iterdir()),[vecino])
                self.assertEqual((vecino/'conservar.txt').read_bytes(),b'no tocar')
                self.assertEqual(self.db.read_bytes(),self.previo)

    def test_trabajador_error_procesar_o_guardar_cierra_y_no_da_alta(self):
        from interfaz.trabajador import registrar
        for error in (cv2.error('fallo procesamiento'),OSError('fallo escritura')):
            with self.subTest(error=type(error)),ExitStack() as st:
                st.enter_context(patch('configuracion.config.DB_PATH',str(self.db)))
                st.enter_context(patch('configuracion.config.RAIZ_PROYECTO',str(self.root)))
                g=MagicMock(completo=True,cancelado=False,originales=self.frames)
                st.enter_context(patch('interfaz.recorrido.Recorrido',return_value=g))
                evaluador=MagicMock();evaluador.completar.side_effect=error
                st.enter_context(patch('interfaz.evaluacion_sface.EvaluacionSFace',return_value=evaluador))
                cap=MagicMock();cap.isOpened.return_value=True
                st.enter_context(patch('hardware.camara.abrir_camara',return_value=cap))
                estimador=MagicMock()
                st.enter_context(patch('vision.orientacion.EstimadorOrientacion',return_value=estimador))
                guardar=st.enter_context(patch('interfaz.registro.guardar_persona'))
                exigir=st.enter_context(patch('interfaz.duplicados.exigir_alta'))
                puente=MagicMock();puente.detener.return_value=False
                registrar(puente,'',0,evaluacion=participante())
                cap.read.assert_not_called()
                cap.release.assert_called_once();estimador.close.assert_called_once()
                g.cancelar.assert_called_once();g.limpiar.assert_called_once()
                guardar.assert_not_called();exigir.assert_not_called()
                self.assertFalse(any(c.args[0] in ('guardado','evaluacion') for c in puente.emitir.call_args_list))
                self.assertTrue(any(c.args[0]=='aviso_registro' and 'No se complet' in c.args[1] for c in puente.emitir.call_args_list))
                self.assertEqual(self.db.read_bytes(),self.previo)

    def test_alison_alias_mide_todos_sin_criterio_ni_alta(self):
        r=self.evaluador().completar(self.frames,lambda:False)
        self.assertEqual(r['analisis']['primeras_por_identidad'],{'Alison':5})
        self.assertEqual(r['participante_declarado']['nombre_propuesto'],'Alison_Prueba')
        self.assertEqual(r['personas'][1]['nombre'],'Alison')
        self.assertEqual(r['personas'][1]['minima'],.65)
        self.assertIsNone(r['analisis']['decision_duplicado'])
        self.assertFalse(r['analisis']['permitir_alta'])
        self.assertEqual(self.sf.reconocedor.match.call_count,15)
        self.sf.comparar.assert_not_called()  # No usa la decision del reconocimiento .363.
        carpeta=Path(r['carpeta'])
        self.assertEqual(len(list(carpeta.glob('*.png'))),5)
        self.assertFalse(list(carpeta.glob('*.json')))
        self.assertEqual(self.db.read_bytes(),self.previo)
        for postura,frame in zip(('frente','derecha','izquierda','arriba','abajo'),self.frames):
            np.testing.assert_array_equal(cv2.imread(str(carpeta/(postura+'.png'))),frame)

    def test_externo_no_es_bloqueado_por_primera_candidata(self):
        for maximo in (.25,.95):
            self.sf.reconocedor.match.side_effect=lambda f,ref,t: maximo if ref[0,0]==2 else maximo-.1
            r=self.evaluador(participante('no_registrada','EXTERNA_001')).completar(self.frames,lambda:False)
            self.assertEqual(r['analisis']['primeras_por_identidad'],{'Alison':5})
            self.assertIsNone(r['analisis']['decision_duplicado'])
            self.assertIn('no demuestra coincidencia',r['analisis']['descripcion'])
            self.assertEqual(r['analisis']['validas'],5)

    def test_sin_lectura_y_empates_visibles_sin_otra_captura(self):
        self.sf.extraer.side_effect=[(None,'YuNet: sin rostro')]+[(np.ones((1,128),np.float32),'')]*4
        self.sf.reconocedor.match.return_value=.5
        self.sf.reconocedor.match.side_effect=None
        r=self.evaluador().completar(self.frames,lambda:False)
        self.assertEqual(self.sf.extraer.call_count,5)
        self.assertEqual(r['analisis']['sin_lectura'],['frente'])
        self.assertIn('empates exactos',r['analisis']['descripcion'])
        self.assertIsNone(r['analisis']['candidata_repetida'])
        self.assertIsNone(r['personas'][0]['posturas']['frente'])

    def test_cancelacion_cambio_galeria_y_gris_no_guardan(self):
        self.assertIsNone(self.evaluador().completar(self.frames,lambda:True))
        self.assertFalse(self.destino.exists())
        with self.assertRaisesRegex(ValueError,'BGR'):
            self.evaluador().completar([np.zeros((200,200),np.uint8)]*5,lambda:False)
        with patch('interfaz.evaluacion_sface.cargar',side_effect=[(self.galeria,'base','galeria'),(self.galeria,'nueva','otra')]):
            with self.assertRaisesRegex(ValueError,'cambio'):
                self.evaluador().completar(self.frames,lambda:False)
        self.assertFalse(self.destino.exists())

    def test_alias_y_consentimiento_no_crean_altas(self):
        base=pickle.loads(self.previo)
        for datos in ({**participante(),'acepta_conservar':False},
                      {**participante(),'nombre_propuesto':'Alison'},
                      {**participante(),'nombre_propuesto':''}):
            with self.assertRaises(ValueError): validar_participante_sface(datos,base)

    def test_cancelacion_durante_guardado_elimina_solo_temporales(self):
        cancelar=iter([False]*7+[True]*8)
        r=self.evaluador().completar(self.frames,lambda:next(cancelar,True))
        self.assertIsNone(r)
        self.assertEqual(list(self.destino.iterdir()),[])
        self.assertEqual(self.db.read_bytes(),self.previo)

    def test_web_normal_recorrido_sface_puntajes_y_archivos_temporales(self):
        import json
        from types import SimpleNamespace
        from tests.test_web import HTTPTests
        from web_local.control import Control
        from interfaz.trabajador import registrar
        from evaluar_sface import csv_leer
        HTTPTests.setUpClass()
        web=HTTPTests(); web.setUp(); self.addCleanup(web.tearDown)
        estado=Control(db_path=self.db, fotos_dir=self.root/'fotos')
        estado.estado.update(actividad='registro', evaluacion_registro=True, evaluacion_motor='sface')
        web.control.ver.side_effect=estado.ver
        puente=MagicMock(); puente.comandos=Queue(); puente.detener.return_value=False
        puente.emitir.side_effect=estado.evento
        cap=MagicMock();cap.isOpened.return_value=True
        frames=[np.random.default_rng(k+900).integers(0,256,(160,200,3),np.uint8) for k in range(30)]
        cap.read.side_effect=[(True,f) for f in frames]
        estimador=MagicMock();estimador.analizar.side_effect=lecturas()
        self.stack.enter_context(patch('configuracion.config.DB_PATH',str(self.db)))
        self.stack.enter_context(patch('configuracion.config.RAIZ_PROYECTO',str(self.root)))
        self.stack.enter_context(patch('hardware.camara.abrir_camara',return_value=cap))
        self.stack.enter_context(patch('vision.orientacion.EstimadorOrientacion',return_value=estimador))
        reloj=iter(np.arange(0,100,.2))
        self.stack.enter_context(patch('interfaz.trabajador.time',SimpleNamespace(
            monotonic=lambda:next(reloj),sleep=lambda _:None)))
        guardar=self.stack.enter_context(patch('interfaz.registro.guardar_persona'))
        exigir=self.stack.enter_context(patch('interfaz.duplicados.exigir_alta'))
        def despachar(actividad, cam_idx, evaluacion):
            self.assertEqual(actividad,'registro')
            registrar(puente,'',cam_idx,evaluacion=evaluacion)
        web.control.iniciar.side_effect=despachar
        web.entrar()
        codigo,_,_=web.peticion('/api/evaluar-duplicados',dict(camara=0,evaluacion=participante()))
        self.assertEqual(codigo,200)
        codigo,_,cuerpo=web.peticion('/api/estado')
        self.assertEqual(codigo,200)
        resultado=json.loads(cuerpo)['estado']['resultado_evaluacion']
        self.assertEqual(resultado['motor'],'sface')
        self.assertEqual(resultado['analisis']['validas'],5)
        self.assertIsNone(resultado['analisis']['decision_duplicado'])
        self.assertFalse(resultado['analisis']['permitir_alta'])
        self.assertEqual(len(resultado['personas']),3)
        carpeta=Path(resultado['carpeta'])
        self.assertTrue(carpeta.is_relative_to(self.root))
        filas=csv_leer(carpeta/'puntuaciones.csv')
        self.assertEqual(len(filas),15)  # Cinco posturas por tres identidades.
        self.assertEqual({f['postura'] for f in filas},{'frente','derecha','izquierda','arriba','abajo'})
        self.assertIn('Sin criterio de duplicados validado',(carpeta/'resumen.md').read_text(encoding='utf-8'))
        self.assertEqual(len(list(carpeta.glob('*.png'))),5)
        self.assertEqual(cap.read.call_count,30)
        cap.release.assert_called_once(); estimador.close.assert_called_once()
        guardar.assert_not_called(); exigir.assert_not_called()
        self.assertEqual(self.db.read_bytes(),self.previo)

    def test_trabajador_reutiliza_cinco_seleccionadas_sin_guardar_registro(self):
        from interfaz.trabajador import registrar
        self.stack.enter_context(patch('configuracion.config.DB_PATH',str(self.db)))
        self.stack.enter_context(patch('configuracion.config.RAIZ_PROYECTO',str(self.root)))
        cap=MagicMock();cap.isOpened.return_value=True
        frames=[np.random.default_rng(k).integers(0,256,(160,200,3),np.uint8) for k in range(30)]
        cap.read.side_effect=[(True,f) for f in frames]
        self.stack.enter_context(patch('hardware.camara.abrir_camara',return_value=cap))
        estimador=MagicMock();estimador.analizar.side_effect=lecturas()
        self.stack.enter_context(patch('vision.orientacion.EstimadorOrientacion',return_value=estimador))
        guardar=self.stack.enter_context(patch('interfaz.registro.guardar_persona'))
        exigir=self.stack.enter_context(patch('interfaz.duplicados.exigir_alta'))
        self.stack.enter_context(patch('interfaz.trabajador.time.sleep'))
        reloj=iter(np.arange(0,100,.2))
        self.stack.enter_context(patch('interfaz.trabajador.time.monotonic',side_effect=lambda:next(reloj)))
        puente=MagicMock();puente.comandos=Queue();puente.detener.return_value=False
        registrar(puente,'',0,evaluacion=participante())
        self.assertEqual(cap.read.call_count,30)
        self.assertEqual(self.sf.extraer.call_count,5)
        # Nitidez creciente: elige 6,12,18,24,30, sin leer una captura adicional.
        for llamada,original in zip(self.sf.extraer.call_args_list,frames[5::6]):
            np.testing.assert_array_equal(llamada.args[0],original)
        guardar.assert_not_called();exigir.assert_not_called()
        cap.release.assert_called_once();estimador.close.assert_called_once()
        self.assertEqual(len(list(self.destino.rglob('*.png'))),5)
        self.assertFalse(any(c.args[0]=='guardado' for c in puente.emitir.call_args_list))
        self.assertEqual(self.db.read_bytes(),self.previo)


class ControlTests(unittest.TestCase):
    def test_preflight_sface_no_abre_hardware_si_no_hay_galeria(self):
        from web_local.control import Control
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'base.pkl';p.write_bytes(pickle.dumps(dict(nombres=['Alison'],imagenes=['foto'])))
            c=Control(p,Path(td));c.ctx=MagicMock()
            with patch('vision.galeria_sface.cargar',side_effect=ValueError('Galeria ausente')):
                with self.assertRaisesRegex(ValueError,'Galeria ausente'):
                    c.iniciar('registro',evaluacion=participante())
            c.ctx.Process.assert_not_called()

    def test_empate_de_votos_no_impone_identidad(self):
        filas=[dict(postura=str(i),primera=n,puntuaciones={n:.5}) for i,n in enumerate(['Alison','Alexa','Alison','Alexa'])]
        filas.append(dict(postura='4',primera=None,puntuaciones={}))
        r=analizar_lecturas(filas,participante())
        self.assertIsNone(r['candidata_repetida'])
        self.assertIn('Empate de votos',r['descripcion'])


if __name__=='__main__':
    unittest.main()
