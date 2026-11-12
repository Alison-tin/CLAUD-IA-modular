import os
from pathlib import Path
import unittest
from unittest.mock import patch
from web_local.control import Control
from tests.test_web import HTTPTests

class PresentacionHTTPTests(HTTPTests):
    def test_png_originales_servidos(self):
        import hashlib
        originales = {'frente': '157d6dafc337b15a7ae94d951e6f8204eac98d85abea2983c76bdd4ab69cf123', 'derecha': '5c0a716e5a2173a539287d0f56459648616d24788fb1b60d6d171ffe498df04a', 'izquierda': '7cc72202edc6a20b511df9db56be45ab03abd0faea4cca3d89fe511382601638', 'arriba': '941d2d3d74e0eaad833fffb5f9590d2ebafd124e0e1b32fff8db07145fd69608', 'abajo': '202485105bbd09d87d7fc5533e47f057620bd7df162d24bdfcabfbbd5ec99ea6'}
        for n, huella in originales.items():
            codigo, headers, cuerpo = self.peticion(f'/imagenes/posturas/{n}.png')
            self.assertEqual(codigo, 200)
            self.assertEqual(headers['Content-Type'], 'image/png')
            self.assertEqual(hashlib.sha256(cuerpo).hexdigest(), huella)

class FiltroTests(unittest.TestCase):
    def test_normal_filtra_sin_mutar_estado_interno(self):
        with patch.dict(os.environ, {}, clear=True):
            c=Control()
        c.estado['recorrido']={'diagnostico':{'angulos': [15,0,0]}, 'analizados':42,
            'lecturas_estables':3, 'lecturas_necesarias':6, 'instruccion':'Mejora la luz'}
        e=c.ver()['recorrido']
        self.assertEqual(e, {'progreso_postura':.5,'instruccion':'Mejora la luz'})
        self.assertEqual(c.estado['recorrido']['lecturas_estables'],3)
        self.assertIn('diagnostico',c.estado['recorrido'])
    def test_desarrollo_explicito(self):
        with patch.dict(os.environ, {'CLAUD_IA_DIAGNOSTICO_REGISTRO':'1'}):
            c=Control()
        c.estado['recorrido']={'diagnostico':{'bloqueo':'luz'}}
        self.assertIn('diagnostico',c.ver()['recorrido'])
