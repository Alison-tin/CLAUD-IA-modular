"""Retiro, reinicio y primera alta: datos sintéticos temporales, sin hardware."""
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch,MagicMock

from tests import test_alta_sface as auxiliares
from interfaz.registro import leer_base
from web_local.control import Control
from vision import galeria_sface as g
from vision.motor_sface import MotorSFace


class RetiroSFaceTests(unittest.TestCase):
    servicio=auxiliares.TransaccionTests.servicio

    def setUp(self):
        auxiliares.TransaccionTests.setUp(self)
        self.fotos.mkdir()
        rutas=[]
        for i in range(2):
            p=self.fotos/f'original_{i}.jpg';p.write_bytes(self.jpegs[i]);rutas.append(str(p))
        self.db.write_bytes(pickle.dumps(dict(nombres=['Alexa','Alison'],imagenes=rutas)))
        g.preparar(self.db,self.e,self.gal)
        self.prev=self.db.read_bytes()
        self.originales={p:p.read_bytes() for p in self.fotos.iterdir()}
        preparar,cargar,inventario=g.preparar,g.cargar,g.inventario
        self.preparar=preparar;self.cargar=cargar
        # Las dependencias del controlador apuntan únicamente a estas carpetas temporales.
        self.s.enter_context(patch.object(g,'inventario',side_effect=lambda db_path, evaluacion=None:inventario(db_path,self.e)))
        self.s.enter_context(patch.object(g,'preparar',side_effect=lambda db_path, *a,**kw:preparar(db_path,self.e,self.gal)))
        self.s.enter_context(patch.object(g,'cargar',side_effect=lambda db_path=None,*a,**kw:cargar(db_path or self.db,self.e,self.gal)))
        self.s.enter_context(patch.object(g,'GALERIAS',self.gal))
        self.control=Control(self.db,self.fotos)

    def comprobar_inicio(self):
        with patch.object(self.control.ctx,'Process') as proceso,patch('web_local.control.threading.Thread'):
            self.control.iniciar('sistema',modo='demo')
            proceso.assert_called_once()
        self.control.proceso=None
        with patch('vision.motor_sface.cargar',side_effect=lambda **kw:self.cargar(self.db,self.e,self.gal)), \
             patch('vision.motor_sface.SFace',return_value=self.motor):
            return MotorSFace(db_path=self.db)

    def test_eliminar_y_reiniciar_excluye_retirada_con_respaldo(self):
        respaldo=self.control.eliminar('Alexa')
        self.assertEqual([p['nombre'] for p in self.control.personas()],['Alison'])
        self.assertEqual(set(self.comprobar_inicio().galeria),{'Alison'})
        from interfaz.bajas import carpeta_respaldos
        self.assertEqual((carpeta_respaldos(self.db)/respaldo/'base_antes.pkl').read_bytes(),self.prev)
        self.assertTrue(all(p.read_bytes()==b for p,b in self.originales.items()))
        self.assertFalse(list(self.db.parent.glob('*.tmp')))

    def test_fallo_galeria_o_verificacion_conserva_base_y_arranque(self):
        for fase in ('preparar','cargar'):
            with self.subTest(fase=fase),patch.object(g,fase,side_effect=OSError('Disco simulado')):
                with self.assertRaisesRegex(ValueError,'persona se conserva'):
                    self.control.eliminar('Alexa')
            self.assertEqual(self.db.read_bytes(),self.prev)
            self.assertEqual(set(self.comprobar_inicio().galeria),{'Alexa','Alison'})
            self.assertFalse(list(self.db.parent.glob('*.tmp')))

    def test_fallo_publicacion_no_elimina_y_permite_reintentar(self):
        with patch('interfaz.bajas.os.replace',side_effect=OSError('Fallo simulado')):
            with self.assertRaisesRegex(ValueError,'registro se conserva'):
                self.control.eliminar('Alexa')
        self.assertEqual(self.db.read_bytes(),self.prev)
        self.assertEqual(set(self.comprobar_inicio().galeria),{'Alexa','Alison'})
        self.control.eliminar('Alexa')
        self.assertEqual(set(self.comprobar_inicio().galeria),{'Alison'})

    def test_ultima_persona_arranca_vacio_y_primera_alta_persiste(self):
        self.control.eliminar('Alexa');self.control.eliminar('Alison')
        self.assertEqual(self.control.personas(),[])
        motor=self.comprobar_inicio()
        self.assertEqual(motor.galeria,{})
        self.assertEqual(motor.reconocer(self.frames[0],[(0,0,150,150)])['estado_reconocimiento'],'sin_registros')
        with patch.object(self.control.ctx,'Process') as proceso,patch('web_local.control.threading.Thread'):
            self.control.iniciar_recorrido('Primera nueva')
            proceso.assert_called_once()
        self.control.proceso=None
        r=self.servicio().completar('Primera nueva',self.frames,self.jpegs,lambda:False)
        self.assertEqual(r['estado'],'guardado')
        nuevo=Control(self.db,self.fotos)
        self.assertEqual(nuevo.personas(),[dict(nombre='Primera nueva',fotos=5)])
        self.assertEqual(set(self.comprobar_inicio().galeria),{'Primera nueva'})

    def test_repara_version_ausente_tras_retiro_anterior(self):
        # Reproduce el fallo anterior: base modificada sin la galería correspondiente.
        base=leer_base(self.db)
        base['nombres']=base['nombres'][1:];base['imagenes']=base['imagenes'][1:]
        self.db.write_bytes(pickle.dumps(base))
        self.assertEqual(set(self.comprobar_inicio().galeria),{'Alison'})
        self.assertEqual(leer_base(self.db),base)

    def test_web_elimina_actualiza_personas_y_arranca_demo_sin_scripts(self):
        import http.client,json,secrets,threading,time
        from web_local.servidor import Servidor,crear_acceso
        from tests.test_web import trabajador_simulado
        self.control.trabajador=trabajador_simulado
        clave=secrets.token_urlsafe(24)
        server=Servidor(('127.0.0.1',0),self.control,crear_acceso(clave))
        hilo=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01});hilo.start()
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
        cookie=csrf=''
        def pedir(ruta,datos=None):
            headers={'Origin':f'http://127.0.0.1:{server.server_port}','Cookie':cookie,'X-CSRF-Token':csrf}
            if datos is not None:headers['Content-Type']='application/json'
            conn.request('POST' if datos is not None else 'GET',ruta,
                json.dumps(datos) if datos is not None else None,headers)
            respuesta=conn.getresponse();cuerpo=json.loads(respuesta.read())
            return respuesta,cuerpo
        try:
            respuesta,datos=pedir('/api/entrar',{'clave':clave})
            self.assertEqual(respuesta.status,200)
            cookie=respuesta.getheader('Set-Cookie').split(';',1)[0];csrf=datos['csrf']
            for nombre,restantes in [('Alexa',['Alison']),('Alison',[])]:
                respuesta,_=pedir('/api/eliminar',{'nombre':nombre,'confirmacion':nombre})
                self.assertEqual(respuesta.status,200)
                _,datos=pedir('/api/personas')
                self.assertEqual([p['nombre'] for p in datos['personas']],restantes)
                respuesta,_=pedir('/api/iniciar',{'modo':'demo','camara':0})
                self.assertEqual(respuesta.status,200)
                respuesta,_=pedir('/api/detener',{})
                self.assertEqual(respuesta.status,200)
                for _ in range(100):
                    if self.control.proceso is None:break
                    time.sleep(.02)
                self.assertIsNone(self.control.proceso)
        finally:
            self.control.detener()
            conn.close();server.shutdown();server.server_close();hilo.join()
