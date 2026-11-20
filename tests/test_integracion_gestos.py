import json
import os
from pathlib import Path
from queue import Queue
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import numpy as np
from hardware.arduino import Arduino
from interfaz.trabajador import Puente
from secuencias.reproductor import Reproductor, cargar, resolver_json
from propuestas_movimientos.preparar import validar
from vision.manos import GESTO_ACCION
from web_local.control import Control


class Rapido(threading.Event):
    def wait(self, timeout=None):
        return self.is_set()


class IntegracionTests(unittest.TestCase):
    def test_guia_usa_archivos_reales_y_marca_asociacion_incompleta(self):
        from vision.manos import guia_gestos
        guia={g['gesto']:g for g in guia_gestos()}
        self.assertEqual(len(guia),9)
        for gesto,(archivo,_) in GESTO_ACCION.items():
            self.assertEqual(guia[gesto]['archivo'],archivo)
            self.assertTrue(guia[gesto]['coincide_con_codigo'])
        self.assertIn('dos veces',guia['indice']['movimiento'])
        self.assertIn('tres veces',guia['ok']['movimiento'])
        self.assertIn('confirmar',guia['mano_abierta']['nota'])
        real=cargar
        def cambiado(nombre):
            bloques=real(nombre)
            return bloques[:-1] if nombre=='sorpresa.json' else bloques
        with patch('secuencias.reproductor.cargar',side_effect=cambiado):
            cambio={g['gesto']:g for g in guia_gestos()}['pulgar_arriba']
            self.assertFalse(cambio['coincide_con_codigo'])
            self.assertIn('incompleta',cambio['nota'])
        with patch('secuencias.reproductor.cargar',side_effect=ValueError('Falta JSON')):
            self.assertTrue(all(not g['disponible'] for g in guia_gestos()))

    def test_guia_sigue_una_reasignacion_del_selector(self):
        from vision.manos import guia_gestos
        with patch.dict(GESTO_ACCION,{'pulgar_arriba':('risa.json','Risa')}):
            g=next(g for g in guia_gestos() if g['gesto']=='pulgar_arriba')
            self.assertEqual(g['archivo'],'risa.json')
            self.assertIn('tres veces',g['movimiento'])

    def test_saludo_cancelable_no_reanuda_tras_otra_accion(self):
        from movimientos.expresiones import expresion_saludo
        with patch('serial.Serial',side_effect=AssertionError('Sin hardware')):
            a=Arduino(demo=True);a.demo=False;a.ser=MagicMock(is_open=True)
            pausa=threading.Event()
            real=a.enviar
            def enviar(*args):
                real(*args)
                if args[:2]==(0,12):pausa.set()
            with patch.object(a,'enviar',side_effect=enviar):
                rep=Reproductor(a)
                self.assertTrue(rep.accion('saludo',expresion_saludo))
                self.assertTrue(pausa.wait(2))
                self.assertFalse(rep.reproducir('sorpresa.json'))
                a.enviar(2,1,464,3)  # seguimiento descartado durante saludo
                rep.detener()
                n=a.ser.write.call_count
                self.assertFalse(rep._t.is_alive())
                self.assertFalse(rep.accion('saludo',expresion_saludo))
                self.assertFalse(rep.reproducir('risa.json'))
                self.assertEqual(a.ser.write.call_count,n)
                self.assertIn(b'PCA,2,4,410,3\n',[c.args[0] for c in a.ser.write.call_args_list])
                self.assertNotIn(b'PCA,2,1,464,3\n',[c.args[0] for c in a.ser.write.call_args_list])

    def test_limites_de_todas_las_acciones_sin_cambiar_calibracion(self):
        from movimientos import expresiones,gestos
        from configuracion.config import PCA,PCA0,SERVO,SERVO_CUELLO
        with patch('serial.Serial',side_effect=AssertionError('Sin hardware')),patch('time.sleep'):
            a=Arduino(demo=True);a.demo=False;a.ser=MagicMock(is_open=True)
            a.ir_a_centro();a.parpado(True);a.parpado(False)
            for funcion in [expresiones.expresion_saludo,expresiones.expresion_neutro,
                            expresiones.expresion_reconocido,expresiones.mover_boca]+gestos.GESTOS_VIDA:
                funcion(a)
            for llamada in a.ser.write.call_args_list:
                _,p,c,pos,v=llamada.args[0].decode().strip().split(',')
                tabla=SERVO if int(p)==PCA else SERVO_CUELLO
                self.assertLessEqual(tabla[int(c)][0],int(pos))
                self.assertLessEqual(int(pos),tabla[int(c)][1])
            n=a.ser.write.call_count
            for orden in [(2,4,400,3),(0,12,651,3),(9,1,400,3),(0,13,300,3),
                          (2,4,float('nan'),3),(2,4,525,0)]:
                with self.assertRaises(ValueError):a.enviar(*orden)
            self.assertEqual(n,a.ser.write.call_count)

    def test_emociones_mapeo_inferencias_distintas_y_cancelacion(self):
        from secuencias.reproductor import EmocionesMovimiento
        rep=MagicMock();rep.reproducir.return_value=True;rep.accion.return_value=True
        for nombre,archivo in EmocionesMovimiento.MAPA.items():
            control=EmocionesMovimiento(rep)
            r=dict(nombre=nombre,sesion='sesion',id=1)
            self.assertFalse(control.actualizar(r))
            self.assertFalse(control.actualizar(r))  # misma inferencia no confirma
            self.assertTrue(control.actualizar(dict(r,id=2)))
            rep.reproducir.assert_called_with(archivo,origen='emocion')
            self.assertFalse(control.actualizar(dict(r,id=3)))  # pausa entre acciones
            control.actualizar(None)
            rep.cancelar_emocion.assert_called()
        control=EmocionesMovimiento(rep)
        for i,n in enumerate(['Feliz','Triste','Feliz','Asustado','Asustado','Disgustado','Disgustado']):
            self.assertFalse(control.actualizar(dict(nombre=n,sesion='s',id=i)))
        a=Arduino(demo=True);a.demo=False;a.ser=MagicMock(is_open=True)
        enviada=threading.Event()
        a.ser.write.side_effect=lambda _:enviada.set()
        real=Reproductor(a)
        try:
            self.assertTrue(real.reproducir('risa.json',origen='emocion'))
            self.assertTrue(enviada.wait(2))
            real.cancelar_emocion()
            real._t.join(2)
            self.assertFalse(real._t.is_alive())
            n=a.ser.write.call_count
            real.detener()
            self.assertEqual(n,a.ser.write.call_count)
        finally:
            real.detener()

    def test_web_fisica_emocion_proceso_a_secuencia_arduino_simulado(self):
        from aplicacion import sistema
        import itertools
        eventos=Queue();puente=Puente(eventos,Queue(),Queue(),threading.Event(),)
        backend=MagicMock()
        backend.estado.return_value=dict(ok=True,intentado=True,externo=True,detalle='listo')
        ids=itertools.count(1)
        backend.actualizar.side_effect=lambda *a,**k:dict(nombre='Feliz',confianza=.8,sesion='s',id=next(ids))
        cara=MagicMock();cara.detect.return_value=NS(detections=[NS(bounding_box=NS(origin_x=220,origin_y=100,width=180,height=200))])
        mano=MagicMock();mano.detect.return_value=NS(hand_landmarks=[])
        cap=MagicMock();cap.isOpened.return_value=True
        cap.get.side_effect=lambda prop:640 if prop==3 else 480 if prop==4 else 30
        reps=[];contador=[0]
        def leer():
            if reps and reps[0]._t:reps[0]._t.join(2)
            contador[0]+=1
            return (True,np.zeros((480,640,3),np.uint8)) if contador[0]<=4 else (False,None)
        cap.read.side_effect=leer
        def crear(a,avisar):
            rep=Reproductor(a,avisar);rep._cancelar=Rapido()
            rep.accion=MagicMock(return_value=False)  # aislar disparador de emociones del saludo
            reps.append(rep);return rep
        serial=MagicMock(is_open=True);serial.port='SIMULADO'
        with ExitStack() as st:
            st.enter_context(patch('serial.Serial',return_value=serial))
            st.enter_context(patch('cv2.VideoCapture',side_effect=AssertionError('Sin camara')))
            for nombre,valor in [('abrir_camara',cap),('descargar_modelos',None),('cargar_db',{'imagenes':[],'nombres':[]})]:
                st.enter_context(patch.object(sistema,nombre,return_value=valor))
            st.enter_context(patch.object(sistema,'_cargar_deepface',side_effect=AssertionError('No importar DeepFace en principal')))
            st.enter_context(patch.object(sistema.mp_vision.FaceDetector,'create_from_options',return_value=cara))
            st.enter_context(patch.object(sistema.mp_vision.HandLandmarker,'create_from_options',return_value=mano))
            st.enter_context(patch.object(sistema,'Reproductor',side_effect=crear))
            st.enter_context(patch.object(sistema.time,'sleep'))
            sistema.ejecutar(demo=False,port='SIMULADO',interfaz=puente,emociones_externas=backend)
        enviados=[c.args[0].decode() for c in serial.write.call_args_list]
        esperados=[f'PCA,{p},{c},{pos},{v}\n' for p,c,t,d,pos,v in cargar('alegria.json')]
        self.assertTrue(any(enviados[i:i+len(esperados)]==esperados for i in range(len(enviados))))
        self.assertTrue(any(t=='expresion' and v['nombre']=='Feliz' for t,v in eventos.queue))
        backend.cerrar.assert_called_once();serial.close.assert_called_once()
        self.assertFalse(reps[0]._t.is_alive())

    def test_confirmacion_web_advierte_propuestas(self):
        html = (Path(__file__).resolve().parents[1]/'web_local/static/index.html').read_text(encoding='utf-8')
        dialogo = html.split('id="physical-dialog"', 1)[1].split('</dialog>', 1)[0]
        self.assertIn('Los gestos de mano activan los movimientos validados.', dialogo)

    def test_camara_simulada_nueve_gestos_demo_y_web_fisica_simulada(self):
        from aplicacion import sistema
        for web in (False, True):
            for gesto, (nombre, _) in GESTO_ACCION.items():
                with self.subTest(web=web, gesto=gesto), ExitStack() as st:
                    cap = MagicMock()
                    cap.isOpened.return_value = True
                    cap.get.side_effect = lambda prop: 640 if prop == 3 else 480 if prop == 4 else 30
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cap.read.side_effect = [(True, frame)] * 24 + [(False, None)]
                    cara, mano = MagicMock(), MagicMock()
                    cara.detect.return_value = NS(detections=[])
                    mano.detect.return_value = NS(hand_landmarks=[[NS(x=.5, y=.5)] * 21], handedness=[])
                    serial = MagicMock()
                    serial.is_open = True
                    serial.port = 'SIMULADO'
                    serial.write.side_effect = lambda b: len(b)
                    puerto = st.enter_context(patch('serial.Serial', return_value=serial))
                    st.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camara real prohibida')))
                    st.enter_context(patch.object(sistema, 'abrir_camara', return_value=cap))
                    st.enter_context(patch.object(sistema, 'descargar_modelos'))
                    st.enter_context(patch.object(sistema, '_cargar_deepface'))
                    st.enter_context(patch.object(sistema, 'deepface_estado', return_value={'ok':False, 'intentado':True}))
                    st.enter_context(patch.object(sistema, 'cargar_db', return_value={'imagenes':[], 'nombres':[]}))
                    st.enter_context(patch.object(sistema.mp_vision.FaceDetector, 'create_from_options', return_value=cara))
                    st.enter_context(patch.object(sistema.mp_vision.HandLandmarker, 'create_from_options', return_value=mano))
                    st.enter_context(patch.object(sistema, 'clasificar_gesto', return_value=gesto))
                    st.enter_context(patch.object(sistema, 'dibujar_mano', return_value=(0,0,10,10)))
                    st.enter_context(patch.object(sistema, 'dedos_arriba', return_value=[True]*5))
                    hablar = st.enter_context(patch.object(sistema, 'hablar', side_effect=AssertionError('Boca duplicada')))
                    st.enter_context(patch.object(sistema.time, 'sleep'))
                    reps, ordenes = [], []
                    class Registrado(Arduino):
                        def _enviar(self, pca, canal, pos, vel=5):
                            if self._secuencia == threading.get_ident():
                                ordenes.append((pca, canal, pos, vel))
                            super()._enviar(pca, canal, pos, vel)
                    st.enter_context(patch.object(sistema, 'Arduino', Registrado))
                    def crear(arduino, informar):
                        rep = Reproductor(arduino, informar)
                        rep._cancelar = Rapido()
                        reps.append(rep)
                        return rep
                    st.enter_context(patch.object(sistema, 'Reproductor', side_effect=crear))
                    puente = Puente(Queue(), Queue(), Queue(), threading.Event())
                    puente.video_limpio = web
                    def cuadro(frame):
                        # Los primeros siete fotogramas no confirman el gesto.
                        if cap.read.call_count < 8:
                            self.assertFalse(reps[0]._t)
                        if reps and reps[0]._t:
                            reps[0]._t.join(2)
                    puente.fotograma = cuadro
                    sistema.ejecutar(port='SIMULADO', demo=not web, interfaz=puente)
                    self.assertEqual(ordenes, [(b[0],b[1],b[4],b[5]) for b in cargar(nombre)])
                    eventos = list(puente.eventos.queue)
                    sec = [v for t,v in eventos if t == 'secuencia']
                    self.assertEqual(len(sec), 1)
                    self.assertEqual(sec[0]['archivo'], nombre)
                    self.assertTrue(sec[0]['iniciada'])
                    control = Control()
                    for t,v in eventos:
                        if t in ('secuencia', 'aviso_secuencia'):
                            control.evento(t,v)
                    self.assertTrue(control.ver()['aviso'].endswith(('iniciada','iniciado')))
                    if not web:
                        puerto.assert_not_called()
                        serial.write.assert_not_called()
                    hablar.assert_not_called()
                    cap.release.assert_called_once()

    def test_exclusion_entre_hilos_y_reproducciones(self):
        with patch('serial.Serial') as puerto:
            a = Arduino(demo=True)
            a.demo = False
            a.ser = MagicMock(is_open=True)
            with a.secuencia_exclusiva():
                t = threading.Thread(target=lambda: a.enviar(2,1,464,3))
                t.start(); t.join()
                a.ser.write.assert_not_called()
                a.enviar(2,1,464,3)
                a.ser.write.assert_called_once()
            a.enviar(2,1,464,3)
            self.assertEqual(a.ser.write.call_count, 2)
            rep = Reproductor(a)
            self.assertTrue(rep.reproducir('risa.json'))
            self.assertFalse(rep.reproducir('alegria.json'))
            rep.detener()
            puerto.assert_not_called()

    def test_invalidos_sin_envios_y_aviso_web(self):
        ejemplos = [[], {}, [[True,12,0,0,300,3]], [[0,12,False,0,300,3]],
                    [[0,12,float('nan'),0,300,3]], [[0,12,float('inf'),0,300,3]],
                    [[0,12,'0',0,300,3]], [[0,12,0,-1,300,3]],
                    [[0,12,0,0,900,3]], [[0,13,0,0,300,3]],
                    [[0,12,0,0,300,0]], [[0,12,0,0,300,True]],
                    [[0,12,10**400,0,300,3]], [[0,12,0,0,300,3,9]],
                    [[0,12,0,0,300,3],[0,12,0,0,300,3]]]
        a = Arduino(demo=True)
        control = Control()
        rep = Reproductor(a, lambda m: control.evento('aviso_secuencia', m))
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d)/'invalido.json'
            for datos in ejemplos:
                with self.subTest(datos=datos):
                    with self.assertRaises(ValueError):
                        validar(datos)
                    ruta.write_text(json.dumps(datos), encoding='utf-8')
                    with patch('secuencias.reproductor.resolver_json', return_value=str(ruta)), patch.object(a, 'enviar') as enviar:
                        self.assertFalse(rep.reproducir('risa.json'))
                        enviar.assert_not_called()
                        self.assertIn('invalida risa.json', control.ver()['aviso'])

    def test_rutas_desde_otro_directorio(self):
        anterior = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                for nombre,_ in GESTO_ACCION.values():
                    self.assertTrue(Path(resolver_json(nombre)).is_absolute())
                    self.assertTrue(cargar(nombre))
            finally:
                os.chdir(anterior)
