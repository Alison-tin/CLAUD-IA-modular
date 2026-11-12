"""Carreras de entrega al terminar el trabajador; sin sensores ni altas reales."""
from pathlib import Path
from queue import Queue
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

from interfaz.trabajador import Puente
from web_local.control import Control


class Cola(Queue):
    def close(self):pass
    def cancel_join_thread(self):pass


class RespuestaTests(unittest.TestCase):
    def test_dos_fallos_iguales_se_entregan_en_ambas_recapturas(self):
        eventos=Queue();puente=Puente(eventos,Queue(),Queue(),threading.Event())
        for _ in range(2):
            self.assertTrue(puente.emitir('resultado_registro',{'estado':'dudoso','repetir_postura':3}))
            self.assertEqual(eventos.get_nowait()[0],'resultado_registro')

    def control(self,db):
        c=Control(db,Path(db).parent/'fotos')
        c.estado.update(fase='activo',actividad='registro',alta_sface=True,registro_id='intento',
                        registro_nombre='Nueva',recorrido={'completo':True})
        c.eventos=Cola();c.cuadros=Cola();c.comandos=Cola()
        c.proceso=MagicMock(exitcode=0)
        return c

    def test_resultado_enviado_justo_antes_de_terminar_no_se_pierde(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.control(Path(td)/'base.pkl')
            def finalizar():
                c.eventos.put(('resultado_registro',dict(estado='guardado',persona='Nueva',galeria='v1',mensaje='Registro guardado correctamente')))
                return False
            c.proceso.is_alive.side_effect=finalizar
            c._leer()
            self.assertEqual(c.ver()['resultado_registro']['estado'],'guardado')
            self.assertEqual(c.ver()['resultado_registro']['nombre_solicitado'],'Nueva')

    def test_fin_sin_respuesta_no_afirma_que_no_existe_un_alta(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.control(Path(td)/'base.pkl');c.proceso.is_alive.return_value=False
            c._leer()
            r=c.ver()['resultado_registro']
            self.assertEqual(r['motivo'],'respuesta_no_confirmada')
            self.assertIn('Revisa Personas',r['indicacion'])
            self.assertIn('sin_resultado',(Path(td)/'registro_sface.log').read_text(encoding='utf-8'))

    def test_resultado_critico_reintenta_si_la_cola_estaba_llena(self):
        eventos=Queue(maxsize=1);eventos.put(('recorrido',{}))
        puente=Puente(eventos,Queue(),Queue(),threading.Event())
        def consumir():
            time.sleep(.03);eventos.get()
        t=threading.Thread(target=consumir);t.start()
        try:
            self.assertTrue(puente.emitir('resultado_registro',{'estado':'dudoso'}))
            self.assertEqual(eventos.get_nowait(),('resultado_registro',{'estado':'dudoso'}))
        finally:t.join()
