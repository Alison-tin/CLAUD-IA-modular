from contextlib import ExitStack
import json
from pathlib import Path
import pickle
from queue import Queue
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from interfaz.evaluacion import Evaluacion, POSTURAS, validar_participante
from interfaz.registro import leer_base
from tests.test_recorrido import lecturas, imagen


def participante(tipo='registrada', identificador='Persona Exacta'):
    return dict(tipo=tipo, identificador=identificador, acepta_conservar=True)


class EvaluacionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Cámara real prohibida')))
        self.stack.enter_context(patch('serial.Serial', side_effect=AssertionError('Arduino prohibido')))
        self.raiz = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db = self.raiz/'base.pkl'
        self.fotos = self.raiz/'fotos'
        self.fotos.mkdir()
        nombres, rutas = [], []
        for i, nombre in enumerate(('Persona Exacta','Otra Persona')):
            p=self.fotos/f'{i}.jpg'
            p.write_bytes(cv2.imencode('.jpg', imagen(i))[1].tobytes())
            nombres.append(nombre); rutas.append(str(p))
        self.db.write_bytes(pickle.dumps(dict(nombres=nombres, imagenes=rutas)))
        self.previo = self.db.read_bytes()
        self.fotos_previas = {p.name:p.read_bytes() for p in self.fotos.iterdir()}
        self.destino = self.raiz/'evaluacion_duplicados'
        self.muestras = [cv2.imencode('.jpg',imagen(i))[1].tobytes() for i in range(5)]

    def assert_activos_intactos(self):
        self.assertEqual(self.db.read_bytes(),self.previo)
        self.assertEqual({p.name:p.read_bytes() for p in self.fotos.iterdir()},self.fotos_previas)
        self.assertFalse(Path(str(self.db)+'.duplicados.json').exists())

    def test_cinco_posturas_todas_las_identidades_y_archivos_separados(self):
        for datos in (participante(), participante('no_registrada','PRUEBA_001')):
            e=Evaluacion(datos,self.db,self.fotos,self.destino)
            resultado=e.completar(self.muestras, lambda:False)
            carpeta=Path(resultado['carpeta'])
            self.assertEqual(carpeta.parent,self.destino)
            self.assertEqual(len(list(carpeta.glob('*.jpg'))),5)
            for postura,muestra in zip(POSTURAS,self.muestras):
                self.assertEqual((carpeta/f'{postura}.jpg').read_bytes(),muestra)
            informe=json.loads((carpeta/'resumen.json').read_text(encoding='utf-8'))
            self.assertEqual(informe['participante_declarado'],datos)
            self.assertEqual({p['nombre'] for p in informe['personas']},{'Persona Exacta','Otra Persona'})
            for p in informe['personas']:
                self.assertEqual(set(p['posturas']),set(POSTURAS))
                self.assertTrue(all(np.isfinite(v) and v>=0 for v in p['posturas'].values()))
            self.assertTrue((carpeta/'distancias.csv').exists())
            self.assertNotIn('limites',informe)
        self.assert_activos_intactos()

    def test_participante_exacto_consentimiento_e_identificador_seguro(self):
        for datos in ({}, participante(identificador='persona exacta'), participante('no_registrada','../escape'),
                      participante('no_registrada','Persona Exacta'), {**participante(),'acepta_conservar':False}):
            with self.assertRaises(ValueError):
                validar_participante(datos,leer_base(self.db))

    def test_cancelacion_en_medicion_escritura_y_publicacion_limpia(self):
        # 5 mediciones + previo a escribir + cinco archivos + previo y posterior al renombrado.
        for corte in (1,4,6,8,12,13):
            e=Evaluacion(participante(),self.db,self.fotos,self.destino)
            contador=iter(range(1,30))
            r=e.completar(self.muestras,lambda:next(contador)>=corte)
            self.assertIsNone(r)
            self.assertFalse(self.destino.exists() and list(self.destino.iterdir()))
        self.assert_activos_intactos()

    def test_no_conserva_incompletas_cambio_base_o_error_escritura(self):
        e=Evaluacion(participante(),self.db,self.fotos,self.destino)
        with self.assertRaises(ValueError): e.completar(self.muestras[:4],lambda:False)
        with patch('interfaz.evaluacion.Path.write_bytes',side_effect=OSError('disco simulado')):
            with self.assertRaises(OSError): e.completar(self.muestras,lambda:False)
        self.assertEqual(list(self.destino.iterdir()),[])
        (self.fotos/'0.jpg').write_bytes(self.muestras[4])
        with self.assertRaisesRegex(ValueError,'cambió'): e.completar(self.muestras,lambda:False)
        self.assertEqual(list(self.destino.iterdir()),[])

    def ejecutar_simulado(self, cancelar=False):
        from interfaz.trabajador import registrar
        self.stack.enter_context(patch('configuracion.config.DB_PATH',str(self.db)))
        self.stack.enter_context(patch('configuracion.config.FOTOS_DIR',str(self.fotos)))
        self.stack.enter_context(patch('configuracion.config.RAIZ_PROYECTO',str(self.raiz)))
        cap=MagicMock(); cap.isOpened.return_value=True
        frame=np.zeros((400,400,3),np.uint8); frame[:,:100]=255
        cap.read.return_value=(True,frame)
        abrir=self.stack.enter_context(patch('hardware.camara.abrir_camara',return_value=cap))
        estimador=MagicMock(); estimador.analizar.side_effect=lecturas()
        self.stack.enter_context(patch('vision.orientacion.EstimadorOrientacion',return_value=estimador))
        guardar=self.stack.enter_context(patch('interfaz.registro.guardar_persona',side_effect=AssertionError('Alta prohibida')))
        self.stack.enter_context(patch('interfaz.trabajador.time.sleep'))
        reloj=iter(np.arange(0,100,.2))
        self.stack.enter_context(patch('interfaz.trabajador.time.monotonic',side_effect=lambda:next(reloj)))
        puente=MagicMock(); puente.comandos=Queue()
        puente.detener.return_value=False
        if cancelar: puente.detener.side_effect=[False]*9+[True]*30
        registrar(puente,'',0,evaluacion=participante())
        guardar.assert_not_called()
        abrir.assert_called_once_with(0)
        cap.release.assert_called_once(); estimador.close.assert_called_once()
        self.assert_activos_intactos()
        return puente, estimador, frame

    def test_trabajador_reutiliza_treinta_lecturas_cinco_fotos_sin_alta(self):
        puente,estimador,frame=self.ejecutar_simulado()
        self.assertEqual(estimador.analizar.call_count,30)
        np.testing.assert_array_equal(estimador.analizar.call_args.args[0],frame)
        np.testing.assert_array_equal(puente.fotograma.call_args.args[0],cv2.flip(frame,1))
        resultados=[c.args[1] for c in puente.emitir.call_args_list if c.args[0]=='evaluacion']
        self.assertEqual(len(resultados),1)
        self.assertEqual(len(list(self.destino.rglob('*.jpg'))),5)
        self.assertFalse(any(c.args[0]=='guardado' for c in puente.emitir.call_args_list))

    def test_trabajador_cancelado_no_conserva_muestras(self):
        puente,_,_=self.ejecutar_simulado(cancelar=True)
        self.assertFalse(self.destino.exists())
        self.assertFalse(any(c.args[0]=='evaluacion' for c in puente.emitir.call_args_list))

    def test_control_separa_evaluacion_del_alta_y_exige_consentimiento(self):
        from web_local.control import Control
        c=Control(self.db,self.fotos)
        c.ctx=MagicMock()
        with self.assertRaises(ValueError):
            c.iniciar('registro',evaluacion=dict(participante(),motor='lbph'))
        c.ctx.Process.assert_not_called()
        datos=dict(participante(),motor='sface',nombre_propuesto='Alias_Nuevo')
        with patch('vision.galeria_sface.cargar'), patch('web_local.control.threading.Thread'):
            with self.assertRaises(ValueError):
                c.iniciar('registro',evaluacion=dict(datos,acepta_conservar=False))
            c.iniciar('registro',evaluacion=datos)
        args=c.ctx.Process.call_args.kwargs['args']
        self.assertEqual(args[1]['evaluacion']['motor'],'sface')
        self.assertFalse(args[1]['prueba'])
        self.assertTrue(c.ver()['evaluacion_registro'])
        self.assertEqual(c.ver()['arduino'],'demo')

    def test_alta_sigue_bloqueada_sin_politica_ni_camara(self):
        from web_local.control import Control
        c=Control(self.db,self.fotos)
        c.ctx=MagicMock()
        with self.assertRaises(ValueError):
            c.iniciar('registro',nombre='Nueva Persona')
        c.ctx.Process.assert_not_called()
        self.assert_activos_intactos()
