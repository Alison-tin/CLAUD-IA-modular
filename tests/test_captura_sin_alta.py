"""Inicio de captura independiente de aprobación y guardado, sin hardware."""
from contextlib import ExitStack
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from web_local.control import Control


class CapturaTests(unittest.TestCase):
    def test_revision_exige_id_confirmacion_y_resultado_actual_valido(self):
        from queue import Queue
        c=Control(self.db,self.root/'fotos');c.proceso=MagicMock();c.comandos=Queue()
        c.estado.update(actividad='registro',alta_sface=True,registro_id='actual',
            resultado_registro={'revision_manual':True},recorrido={'completo':True})
        for id,conf in [('anterior',True),('actual',False),('actual','true')]:
            with self.assertRaises(ValueError):c.resolver_registro(id,manual=True,confirmacion=conf)
        c.resolver_registro('actual',manual=True,confirmacion=True)
        self.assertEqual(c.comandos.get_nowait(),('confirmar_nueva','actual'))
        self.assertIsNone(c.estado['resultado_registro'])
        with self.assertRaises(ValueError):c.resolver_registro('actual',manual=True,confirmacion=True)
        c.estado['resultado_registro']={'estado':'duplicado','persona':'Alison'}
        with self.assertRaises(ValueError):c.resolver_registro('actual',manual=True,confirmacion=True)
        c.estado['resultado_registro']={'repetir_postura':2}
        c.resolver_registro('actual')
        self.assertEqual(c.comandos.get_nowait(),('repetir_postura','actual'))
        self.assertEqual(self.db.read_bytes(),self.prev)

    def setUp(self):
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db=self.root/'base.pkl'
        self.db.write_bytes(pickle.dumps({'nombres':[],'imagenes':[]}))
        self.prev=self.db.read_bytes()
        self.stack.enter_context(patch('cv2.VideoCapture',side_effect=AssertionError('Sin camara')))
        self.stack.enter_context(patch('serial.Serial',side_effect=AssertionError('Sin Arduino')))

    def test_nombre_valido_inicia_solo_captura_sin_preparacion(self):
        c=Control(self.db,self.root/'fotos')
        with patch.object(c.ctx,'Process') as proceso, patch('web_local.control.threading.Thread'), patch('vision.galeria_sface.cargar'), \
             patch.object(c,'preparacion_registro',side_effect=AssertionError('Sin boton previo')):
            c.iniciar_recorrido('  Persona Nueva  ',cam_idx=0)
        datos=proceso.call_args.kwargs['args'][1]
        self.assertFalse(datos['prueba'])
        self.assertTrue(datos['alta_sface'])
        self.assertEqual(datos['nombre'],'Persona Nueva')
        self.assertEqual(c.ver()['registro_nombre'],'Persona Nueva')
        self.assertEqual(self.db.read_bytes(),self.prev)

    def test_nombre_invalido_y_alta_sin_aprobacion_no_crean_proceso(self):
        for n in ('','   ','a\n',None,'x'*61):
            # Un salto de línea final se elimina al normalizar: probar control interno.
            if n=='a\n': n='a\nb'
            c=Control(self.db,self.root/'fotos')
            with patch.object(c.ctx,'Process') as proceso:
                with self.assertRaises(ValueError): c.iniciar_recorrido(n)
                proceso.assert_not_called()
        c=Control(self.db,self.root/'fotos')
        with patch.object(c,'preparacion_registro',return_value={'permitido':True}), \
             patch.object(c.ctx,'Process') as proceso:
            with self.assertRaisesRegex(ValueError,'Comenzar registro'):
                c.iniciar('registro',nombre='Persona Nueva')
            proceso.assert_not_called()

    def test_camara_no_disponible_se_valida_dentro_y_cierra_recursos(self):
        from interfaz.trabajador import registrar
        cap=MagicMock();cap.isOpened.return_value=False
        estimador=MagicMock()
        with patch('configuracion.config.DB_PATH',str(self.db)), \
             patch('hardware.camara.abrir_camara',return_value=cap), \
             patch('vision.orientacion.EstimadorOrientacion',return_value=estimador), \
             patch('interfaz.registro.guardar_persona') as guardar:
            with self.assertRaisesRegex(RuntimeError,'camara'):
                registrar(MagicMock(),'Persona Nueva',0,prueba=True)
            guardar.assert_not_called()
        cap.release.assert_called_once(); estimador.close.assert_called_once()
        self.assertEqual(self.db.read_bytes(),self.prev)

    def test_motor_web_unico_en_ambos_flujos(self):
        c=Control(self.db,self.root/'fotos')
        with self.assertRaises(ValueError): c.seleccionar_motor('lbph')
        with self.assertRaises(ValueError):
            c.iniciar('registro',evaluacion={'motor':'lbph'})

    def test_resultado_identifica_intento_y_nombre_solicitado_sin_cambiar_nombre_guardado(self):
        c=Control(self.db,self.root/'fotos')
        with patch.object(c.ctx,'Process'),patch('web_local.control.threading.Thread'),patch('vision.galeria_sface.cargar'):
            c.iniciar_recorrido('Alison Tinoco')
        intento=c.ver()['registro_id']
        self.assertEqual(len(intento),32)
        c.evento('resultado_registro',dict(estado='duplicado',persona='Alison',mensaje='Duplicado'))
        r=c.ver()['resultado_registro']
        self.assertEqual(r['registro_id'],intento)
        self.assertEqual(r['nombre_solicitado'],'Alison Tinoco')
        self.assertEqual(r['persona'],'Alison')
        self.assertEqual(self.db.read_bytes(),self.prev)
