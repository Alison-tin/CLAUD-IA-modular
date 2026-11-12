"""Transacción en carpetas temporales y scores simulados; no valida biometría."""
from contextlib import ExitStack
from pathlib import Path
import pickle
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import MagicMock,patch

import cv2
import numpy as np

from interfaz.alta_sface import AltaSFace,decidir
from vision import galeria_sface as g
from evaluar_sface import csv_guardar


class ReglaTests(unittest.TestCase):
    def test_votos_externa_alexa_no_son_duplicado_y_zona_gris(self):
        self.assertEqual(decidir([{'Alexa':.34,'Alison':.2}]*5,{'Alexa','Alison'}),('nueva',None))
        self.assertEqual(decidir([{'Alexa':.7,'Alison':.3}]*5,{'Alexa','Alison'}),('duplicado','Alexa'))
        for scores in ({'Alexa':.48,'Alison':.3},{'Alexa':.7,'Alison':.6}, {'Alexa':float('nan'),'Alison':.2}):
            self.assertEqual(decidir([scores]*5,{'Alexa','Alison'})[0],'dudoso')
        self.assertEqual(decidir([{'Alexa':.3}]*4,{'Alexa'})[0],'dudoso')


class TransaccionTests(unittest.TestCase):
    def setUp(self):
        self.s=ExitStack();self.addCleanup(self.s.close)
        self.root=Path(self.s.enter_context(tempfile.TemporaryDirectory()))
        self.db=self.root/'base.pkl';self.fotos=self.root/'fotos';self.e=self.root/'eval';self.gal=self.root/'galerias'
        self.db.write_bytes(pickle.dumps(dict(nombres=['Alexa','Alison'],imagenes=['legado1','legado2'])))
        self.s.enter_context(patch('cv2.VideoCapture',side_effect=AssertionError('Sin camara')))
        self.s.enter_context(patch('serial.Serial',side_effect=AssertionError('Sin Arduino')))
        self.s.enter_context(patch.object(g,'verificar_modelos'))
        self.motor=MagicMock()
        self.motor.extraer.side_effect=lambda f:(np.ones((1,128),np.float32)*(1+float(f[0,0,0])/255),'')
        self.motor.reconocedor.match.return_value=.2
        self.s.enter_context(patch.object(g,'SFace',return_value=self.motor))
        self.s.enter_context(patch('interfaz.alta_sface.SFace',return_value=self.motor))
        for k,n in enumerate(('Alexa','Alison')):
            carpeta=self.e/'referencias'/(n+'_origen');carpeta.mkdir(parents=True)
            b=cv2.imencode('.png',np.random.default_rng(k+60).integers(0,256,(140,150,3),np.uint8))[1].tobytes()
            (carpeta/'foto.png').write_bytes(b)
            csv_guardar(carpeta/'capturas.csv',[dict(persona=n,archivo='foto.png',tipo='registrada',uso='referencias',sesion=carpeta.name,sha256=g.digest(b))])
        g.preparar(self.db,self.e,self.gal)
        self.prev=self.db.read_bytes()
        self.antes={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.frames=[np.random.default_rng(k+100).integers(0,256,(160,180,3),np.uint8) for k in range(5)]
        self.jpegs=[cv2.imencode('.jpg',cv2.cvtColor(cv2.resize(f,(200,200)),cv2.COLOR_BGR2GRAY))[1].tobytes() for f in self.frames]

    def servicio(self):return AltaSFace(self.db,self.fotos,self.e,self.gal)

    def test_revision_manual_solo_dudosa_revalida_cinco_y_guarda(self):
        s=self.servicio()
        self.motor.reconocedor.match.return_value=.42
        r=s.completar('Nueva',self.frames,self.jpegs,lambda:False)
        self.assertTrue(r['revision_manual'])
        self.assertEqual(len(r['puntuaciones']),5)
        self.intacto()
        r=s.completar('Nueva',self.frames,self.jpegs,lambda:False,confirmacion_manual=True)
        self.assertEqual(r['estado'],'guardado')
        self.assertEqual(len(g.cargar(self.db,self.e,self.gal)[0]['Nueva']),5)

    def test_manual_no_supera_foto_invalida_ni_duplicado(self):
        s=self.servicio()
        frames=list(self.frames);frames[2]=None
        r=s.completar('Nueva',frames,self.jpegs,lambda:False,confirmacion_manual=True)
        self.assertEqual(r['repetir_postura'],3)
        self.assertNotIn('revision_manual',r)
        self.intacto()
        self.motor.reconocedor.match.side_effect=[.8,.2]*5
        r=s.completar('Nueva',self.frames,self.jpegs,lambda:False,confirmacion_manual=True)
        self.assertEqual(r['estado'],'duplicado')
        self.intacto()

    def test_jpeg_corrupto_pide_solo_su_postura(self):
        fotos=list(self.jpegs);fotos[3]=b''
        r=self.servicio().completar('Nueva',self.frames,fotos,lambda:False)
        self.assertEqual(r['repetir_postura'],4)
        self.intacto()

    def intacto(self):
        self.assertEqual({p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()},self.antes)

    def test_guardado_conjunto_persiste_tras_recrear_motor_y_base(self):
        s=self.servicio();r=s.completar('Nueva / Nombre',self.frames,self.jpegs,lambda:False)
        self.assertEqual(r['estado'],'guardado')
        base=pickle.loads(self.db.read_bytes())
        self.assertEqual(base['nombres'].count('Nueva / Nombre'),5)
        grupos,_,version=g.cargar(self.db,self.e,self.gal)
        self.assertEqual(len(grupos['Nueva / Nombre']),5)
        self.assertEqual(r['galeria'],version)
        self.assertEqual(len(list(self.fotos.rglob('*.jpg'))),5)
        # Un nuevo objeto después de reiniciar consulta lo publicado, no una caché.
        from vision.motor_sface import MotorSFace
        with patch('vision.motor_sface.cargar',side_effect=lambda **kw:g.cargar(self.db,self.e,self.gal)), \
             patch('vision.motor_sface.SFace',return_value=self.motor):
            nuevo=MotorSFace(db_path=self.db)
        self.assertIn('Nueva / Nombre',nuevo.galeria)
        from evaluar_sface import SFace
        motor=object.__new__(SFace)
        motor.detectar=MagicMock(return_value=[np.array([10,10,100,100]+[30,30]*5+[.99],np.float32)])
        feature=nuevo.galeria['Nueva / Nombre'][0].copy()
        motor.extraer_cara=MagicMock(return_value=(feature,''))
        motor.reconocedor=MagicMock()
        motor.reconocedor.match.side_effect=lambda a,b,t:.9 if np.array_equal(a,b) else .1
        nuevo.modelo=motor
        lectura=nuevo.reconocer(np.zeros((140,140,3),np.uint8),[(10,10,100,100)])
        self.assertEqual(lectura['nombre'],'Nueva / Nombre')
        from web_local.control import Control
        self.assertIn({'nombre':'Nueva / Nombre','fotos':5},Control(self.db,self.fotos).personas())
        codigo = """from pathlib import Path
import sys
from vision.galeria_sface import cargar
from web_local.control import Control
db,e,gal,fotos=map(Path,sys.argv[1:])
grupos,_,_=cargar(db,e,gal)
assert len(grupos['Nueva / Nombre'])==5
assert {'nombre':'Nueva / Nombre','fotos':5} in Control(db,fotos).personas()
print('Base y galeria persistentes en otro interprete')
"""
        proceso=subprocess.run([sys.executable,'-B','-c',codigo,str(self.db),str(self.e),str(self.gal),str(self.fotos)],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=20)
        self.assertEqual(proceso.returncode,0,proceso.stdout+proceso.stderr)
        self.assertTrue(all(p.read_bytes()==b for p,b in self.antes.items() if p!=self.db))

    def test_fallos_de_galeria_o_publicacion_no_dejan_base_ni_fotos(self):
        for donde in ('preparar','os.replace'):
            with self.subTest(donde=donde):
                s=self.servicio()
                with patch('interfaz.alta_sface.'+donde,side_effect=OSError('Fallo simulado')):
                    with self.assertRaises(OSError): s.completar('Nueva',self.frames,self.jpegs,lambda:False)
                self.intacto()

    def test_cancelar_antes_y_durante_guardado_no_registra(self):
        s=self.servicio()
        self.assertEqual(s.completar('Nueva',self.frames,self.jpegs,lambda:True)['estado'],'cancelado')
        self.intacto()
        # Cinco comprobaciones de lectura + antes de entrar a guardar + cinco escrituras + publicación.
        for limite in (2,8,12,13,14):
            contador=[0]
            def cancelar():
                contador[0]+=1
                return contador[0]>=limite
            try:r=s.completar('Nueva',self.frames,self.jpegs,cancelar)
            except InterruptedError:pass
            self.intacto()

    def test_fallo_despues_de_crear_galeria_revierte_todo(self):
        real=g.preparar
        def falla(*a,**kw):
            real(*a,**kw)
            raise OSError('Fallo despues de escribir galeria')
        s=self.servicio()
        with patch('interfaz.alta_sface.preparar',side_effect=falla):
            with self.assertRaises(OSError):s.completar('Nueva',self.frames,self.jpegs,lambda:False)
        self.intacto()

    def test_trabajador_reutiliza_las_mismas_cinco_y_anuncia_solo_despues(self):
        from queue import Queue
        from types import SimpleNamespace
        from tests.test_recorrido import lecturas
        from interfaz.trabajador import registrar
        servicio=self.servicio()
        completar=MagicMock(wraps=servicio.completar);servicio.completar=completar
        cap=MagicMock();cap.isOpened.return_value=True
        frames=[np.random.default_rng(k+900).integers(0,256,(160,180,3),np.uint8) for k in range(30)]
        cap.read.side_effect=[(True,f) for f in frames]
        estimador=MagicMock();estimador.analizar.side_effect=lecturas()
        from queue import Queue
        puente=MagicMock();puente.comandos=Queue();puente.detener.return_value=False
        reloj=iter(np.arange(0,100,.2))
        def evento(tipo,valor):
            if tipo=='guardado':
                self.assertEqual(len(g.cargar(self.db,self.e,self.gal)[0]['Nueva']),5)
        puente.emitir.side_effect=evento
        with patch('configuracion.config.DB_PATH',str(self.db)),patch('configuracion.config.FOTOS_DIR',str(self.fotos)), \
             patch('hardware.camara.abrir_camara',return_value=cap), \
             patch('vision.orientacion.EstimadorOrientacion',return_value=estimador), \
             patch('interfaz.alta_sface.AltaSFace',return_value=servicio), \
             patch('interfaz.trabajador.time',SimpleNamespace(monotonic=lambda:next(reloj),sleep=lambda _:None)):
            registrar(puente,'Nueva',0,alta_sface=True)
        self.assertEqual(cap.read.call_count,30)
        self.assertEqual(completar.call_count,1)
        for obtenido,esperado in zip(completar.call_args.args[1],frames[5::6]):
            np.testing.assert_array_equal(obtenido,esperado)
        puente.emitir.assert_any_call('guardado','Nueva')
        cap.release.assert_called_once();estimador.close.assert_called_once()

    def test_duplicado_y_dudoso_no_escriben(self):
        for valor,esperado in ((.7,'duplicado'),(.45,'dudoso')):
            s=self.servicio()
            self.motor.reconocedor.match.side_effect=lambda f,r,t:valor if float(r[0,0])>1.5 else .1
            # Fuerza puntajes por identidad de forma explícita, sin validar caras sintéticas.
            s.grupos={'Alexa':[np.full((1,128),2,np.float32)],'Alison':[np.ones((1,128),np.float32)]}
            r=s.completar('Nueva',self.frames,self.jpegs,lambda:False)
            self.assertEqual(r['estado'],esperado)
            self.intacto()

    def recorrido_resolucion(self, modo):
        from queue import Queue
        from types import SimpleNamespace
        from tests.test_recorrido import lecturas,imagen,ANGULOS
        from vision.orientacion import Observacion
        from interfaz.trabajador import registrar
        servicio=self.servicio();completar=MagicMock(wraps=servicio.completar);servicio.completar=completar
        extra=6 if modo=='repetir' else 0
        frames=[np.random.default_rng(k+900).integers(0,256,(160,180,3),np.uint8) for k in range(30+extra)]
        observaciones=lecturas()+[Observacion(1,ANGULOS[2],imagen(99),120+k) for k in range(extra)]
        cap=MagicMock();cap.isOpened.return_value=True;cap.read.side_effect=[(True,f) for f in frames]
        estimador=MagicMock();estimador.analizar.side_effect=observaciones
        puente=MagicMock();puente.comandos=Queue();cancelado=[False];puente.detener.side_effect=lambda:cancelado[0]
        if modo=='vencer':
            from queue import Empty
            puente.comandos.get=MagicMock(side_effect=Empty)
        if modo=='repetir':
            llamadas=[0]
            def extraer(f):
                llamadas[0]+=1
                return (None,'0 rostros') if llamadas[0]==3 else (np.ones((1,128),np.float32),'')
            self.motor.extraer.side_effect=extraer
        else:self.motor.reconocedor.match.return_value=.42
        def evento(tipo,valor):
            if tipo=='resultado_registro' and (valor.get('revision_manual') or valor.get('repetir_postura')):
                if modo=='cancelar':cancelado[0]=True
                elif modo!='vencer':puente.comandos.put(('repetir_postura' if modo=='repetir' else 'confirmar_nueva','sesion'))
        puente.emitir.side_effect=evento
        reloj=iter(np.arange(0,300,.2))
        with patch('configuracion.config.DB_PATH',str(self.db)),patch('configuracion.config.FOTOS_DIR',str(self.fotos)), \
             patch('hardware.camara.abrir_camara',return_value=cap), \
             patch('vision.orientacion.EstimadorOrientacion',return_value=estimador), \
             patch('interfaz.alta_sface.AltaSFace',return_value=servicio), \
             patch('interfaz.trabajador.time',SimpleNamespace(monotonic=lambda:next(reloj),sleep=lambda _:None)):
            registrar(puente,'Nueva',0,alta_sface=True,registro_id='sesion')
        self.assertEqual(cap.read.call_count,30+extra)
        cap.release.assert_called_once();estimador.close.assert_called_once()
        if modo in ('cancelar','vencer'):
            self.assertEqual(self.db.read_bytes(),self.prev)
            self.assertFalse(self.fotos.exists())
        else:
            self.assertEqual(completar.call_count,2)
            self.assertEqual(len(g.cargar(self.db,self.e,self.gal)[0]['Nueva']),5)

    def test_trabajador_revision_sin_nuevas_capturas(self):self.recorrido_resolucion('manual')
    def test_trabajador_solo_recaptura_postura_fallida(self):self.recorrido_resolucion('repetir')
    def test_trabajador_cancelar_revision_no_guarda(self):self.recorrido_resolucion('cancelar')
    def test_trabajador_vencer_revision_no_guarda(self):self.recorrido_resolucion('vencer')

    def test_motivo_extraccion_y_comparacion_distinguidos_sin_guardado(self):
        s=self.servicio();eventos=[]
        s.auditar=lambda etapa,**datos:eventos.append(dict(etapa=etapa,**datos))
        self.motor.extraer.side_effect=lambda f:(None,'YuNet: 0 rostros')
        r=s.completar('Nueva',self.frames,self.jpegs,lambda:False)
        self.assertEqual(r['motivo'],'extraccion_sin_lectura')
        self.assertIn('YuNet: 0 rostros',eventos[-1]['detalle'])
        self.assertEqual(eventos[-1]['etapa'],'extraccion')
        self.intacto()
        self.motor.extraer.side_effect=lambda f:(np.ones((1,128),np.float32),'')
        self.motor.reconocedor.match.return_value=.45
        r=s.completar('Nueva',self.frames,self.jpegs,lambda:False)
        self.assertEqual(r['motivo'],'comparacion_dudosa')
        self.assertTrue(any(e['etapa']=='comparacion' and len(e.get('puntuaciones',[]))==5 for e in eventos))
        self.intacto()

    def test_errores_identifican_galeria_y_publicacion_sin_dejar_alta(self):
        for objeto,etapa in [('preparar','galeria_preparacion'),('os.replace','publicacion')]:
            s=self.servicio()
            with patch('interfaz.alta_sface.'+objeto,side_effect=OSError('Fallo simulado')):
                with self.assertRaises(OSError):s.completar('Nueva',self.frames,self.jpegs,lambda:False)
            self.assertEqual(s.etapa,etapa)
            self.intacto()

    def test_traza_persistente_conserva_motivo_tecnico_sin_imagenes(self):
        import json
        from interfaz.registro_traza import registrar,error_registro
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/'base.pkl'
            registrar(db,'prueba','comparacion',puntuaciones=[{'Alexa':.2,'Alison':.3}])
            try:raise OSError('Fallo de disco simulado')
            except OSError as e:error_registro(db,'prueba','publicacion',e)
            filas=[json.loads(l) for l in (Path(td)/'registro_sface.log').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(filas[-1]['etapa'],'publicacion')
            self.assertIn('Fallo de disco simulado',filas[-1]['traza'])
            self.assertEqual(len(list(Path(td).iterdir())),1)
