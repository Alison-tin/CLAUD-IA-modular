"""Instrumentación opcional con puntuaciones simuladas, sin imágenes reales."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from evaluar_sface import SFace
from vision.motor_sface import MotorSFace, decidir_sface,ConfirmacionSFace
from vision.galeria_sface import digest


class DiagnosticoTests(unittest.TestCase):
    def test_consecutivas_no_retienen_nombre_y_rechazo_se_limpia(self):
        filtro=ConfirmacionSFace();caja=(0,0,100,100)
        def lectura(n,valor=.6):return decidir_sface(n,valor,{'Registrada','Otra'})
        for esperado in ('Desconocido','Desconocido','Registrada'):
            self.assertEqual(filtro.aplicar(lectura('Registrada'),caja)['nombre'],esperado)
        for n in ('Otra','Registrada','Otra'):
            self.assertEqual(filtro.aplicar(lectura(n),caja)['nombre'],'Desconocido')
        self.assertEqual(filtro.aplicar(lectura('Otra',.1),caja)['nombre'],'Desconocido')
        self.assertEqual(filtro.cuenta,0)
        for _ in range(3):r=filtro.aplicar(lectura('Registrada'),caja)
        self.assertEqual(r['nombre'],'Registrada')
        self.assertEqual(filtro.aplicar(lectura('Registrada'),(500,0,100,100))['nombre'],'Desconocido')
        filtro.aplicar({'estado_reconocimiento':'sin_rostro','nombre':None},None)
        self.assertEqual(filtro.cuenta,0)

    def test_empate_e_inconsistencia_no_asignan_identidad(self):
        self.assertEqual(decidir_sface('A',.7,{'A','B'},{'A':.7,'B':.7})['nombre'],'Desconocido')
        self.assertNotEqual(decidir_sface('A',.7,{'A','B'},{'A':.7,'B':.8})['estado_reconocimiento'],'aceptado')
        self.assertNotEqual(decidir_sface('A',.7,{'A','B'},{'A':.7})['estado_reconocimiento'],'aceptado')

    def test_maximo_por_identidad_y_segunda_distinta_no_segundo_template(self):
        m=object.__new__(SFace)
        m.reconocedor=MagicMock()
        m.reconocedor.match.side_effect=[.61,.59,.58,.2]
        datos={}
        r=m.comparar(np.ones((1,128),np.float32),
            {'Alexa':[np.ones((1,128))]*2,'Alison':[np.ones((1,128))],'daniela':[np.ones((1,128))]},datos)
        self.assertEqual(r,('Alexa',.61,'Alexa'))
        self.assertEqual(datos['puntuaciones'],{'Alexa':.61,'Alison':.58,'daniela':.2})

    def test_registro_mismo_resultado_frame_margen_y_rechazo(self):
        with tempfile.TemporaryDirectory() as td:
            m=object.__new__(MotorSFace)
            m.db_path=Path(td)/'base.pkl'; m.db_path.write_bytes(b'base')
            m.base_sha=digest(b'base'); m.version='version-prueba'; m.sesion_diagnostico='sesion-prueba'
            m.galeria={'Alexa':[np.ones((1,128))],'Alison':[np.ones((1,128))]}
            m.modelo=object.__new__(SFace)
            m.modelo.reconocedor=MagicMock()
            m.modelo.detectar=MagicMock(return_value=[np.array([10,10,80,80]+[30,30]*5+[.99],np.float32)])
            m.modelo.extraer_cara=MagicMock(return_value=(np.ones((1,128),np.float32),''))
            frame=np.zeros((100,100,3),np.uint8)
            for scores in ([.48,.47],[.2,.19]):
                m.diagnostico_activo=False
                m.modelo.reconocedor.match.side_effect=scores
                salida=io.StringIO()
                with contextlib.redirect_stdout(salida):
                    normal=m.reconocer(frame,[(10,10,80,80)])
                self.assertEqual(salida.getvalue(),'')
                m.diagnostico_activo=True
                m.modelo.reconocedor.match.side_effect=scores
                with contextlib.redirect_stdout(salida):
                    diag=m.reconocer(frame,[(10,10,80,80)],fotograma_id=123)
                self.assertEqual(normal,diag)
                d=json.loads(salida.getvalue().split('[SFace diagnostico] ',1)[1])
                self.assertEqual(d['fotograma'],123)
                self.assertEqual(d['sesion'],'sesion-prueba')
                self.assertEqual((d['primera'],d['segunda']),('Alexa','Alison'))
                self.assertAlmostEqual(d['margen'],.01)
                self.assertEqual(d['coseno_primera'],scores[0])
                self.assertEqual(d['decision'],normal['estado_reconocimiento'])
                self.assertNotIn('imagen',d)
            salida=io.StringIO()
            with contextlib.redirect_stdout(salida):
                r=m.reconocer(frame,[],fotograma_id=124)
            d=json.loads(salida.getvalue().split('[SFace diagnostico] ',1)[1])
            self.assertEqual(r['estado_reconocimiento'],'sin_rostro')
            self.assertIsNone(d['margen'])
            self.assertIsNone(d['primera'])
            salida=io.StringIO()
            with contextlib.redirect_stdout(salida):
                m.registrar_sin_comparacion(125,'desactivado')
            d=json.loads(salida.getvalue().split('[SFace diagnostico] ',1)[1])
            self.assertEqual(d['decision'],'desactivado')
            self.assertEqual(d['fotograma'],125)

    def test_sin_retencion_alternancia_y_umbral_conservado(self):
        r=[decidir_sface(n,c,{'Alexa','Alison'}) for n,c in
           [('Alison',.58),('Alexa',.48),('Alison',.49),('Alexa',.2)]]
        self.assertEqual([x['nombre'] for x in r],['Alison','Alexa','Alison','Desconocido'])
        self.assertIsNone(r[-1]['similitud_coseno'])

    def test_opcion_diagnostica_no_habilitada_por_defecto(self):
        import os
        with patch.dict(os.environ,{},clear=True), patch('vision.motor_sface.cargar',return_value=({},'sha','v')), \
             patch('vision.motor_sface.SFace'):
            self.assertFalse(MotorSFace().diagnostico_activo)
