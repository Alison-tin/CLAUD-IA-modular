"""Huecos del flujo de alta: respuestas simuladas, sin validez biometrica."""
from contextlib import ExitStack
from pathlib import Path
import pickle
from queue import Queue
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from interfaz.registro import guardar_persona, leer_base
from interfaz.trabajador import registrar
from interfaz.recorrido import PROTOCOLO
from tests.test_recorrido import lecturas


class RegistroAisladoTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.db=self.root/'base.pkl';self.fotos=self.root/'fotos';self.fotos.mkdir()
        self.jpegs=[cv2.imencode('.jpg',np.random.default_rng(k).integers(0,256,(200,200),np.uint8))[1].tobytes() for k in range(5)]
        p=self.fotos/'original_sintetico.jpg';p.write_bytes(self.jpegs[0])
        self.db.write_bytes(pickle.dumps(dict(nombres=['Sintetica'],imagenes=[str(p)])))
        self.previo=self.snapshot()
        self.stack.enter_context(patch('cv2.VideoCapture',side_effect=AssertionError('No camara real')))
        self.stack.enter_context(patch('serial.Serial',side_effect=AssertionError('No Arduino real')))

    def snapshot(self):
        return {str(p.relative_to(self.root)):p.read_bytes() if p.is_file() else None for p in self.root.rglob('*')}

    def comprobador(self, resultado=('sin_coincidencia',None)):
        v=MagicMock();v.vacia=False;v.limites=object();v.protocolo=PROTOCOLO
        v.comprobar.return_value=resultado
        return v

    def trabajador(self,st,cap=None,estimador=None,final=None):
        cap=cap if cap is not None else MagicMock()
        estimador=estimador if estimador is not None else MagicMock()
        cap.isOpened.return_value=True
        if cap.read.side_effect is None and not isinstance(cap.read.return_value,tuple):
            cap.read.return_value=(True,np.zeros((400,400,3),np.uint8))
        if estimador.analizar.side_effect is None:
            estimador.analizar.side_effect=lecturas()
        st.enter_context(patch('configuracion.config.DB_PATH',str(self.db)))
        st.enter_context(patch('configuracion.config.FOTOS_DIR',str(self.fotos)))
        st.enter_context(patch('hardware.camara.abrir_camara',return_value=cap))
        st.enter_context(patch('vision.orientacion.EstimadorOrientacion',return_value=estimador))
        st.enter_context(patch('interfaz.trabajador.time.sleep'))
        st.enter_context(patch('interfaz.trabajador.time.monotonic',side_effect=iter(np.arange(0,100,.2))))
        primero=self.comprobador();ultimo=final or self.comprobador()
        st.enter_context(patch('interfaz.duplicados.VerificadorDuplicados',side_effect=[primero,ultimo]))
        puente=MagicMock();puente.detener.return_value=False;puente.comandos=Queue()
        return puente,cap,estimador,primero,ultimo

    def test_fallos_captura_y_procesamiento_no_escriben_y_cierran(self):
        for fase in ('lectura_false','lectura_error','procesamiento'):
            with self.subTest(fase=fase),ExitStack() as st:
                cap=MagicMock();estimador=MagicMock()
                if fase=='lectura_false': cap.read.return_value=(False,None)
                elif fase=='lectura_error': cap.read.side_effect=OSError('captura simulada')
                else: estimador.analizar.side_effect=cv2.error('procesamiento simulado')
                puente,cap,estimador,primero,_=self.trabajador(st,cap,estimador)
                with self.assertRaises((RuntimeError,OSError,cv2.error)):
                    registrar(puente,'AliasSintetico',0)
                primero.comprobar.assert_not_called()
                cap.release.assert_called_once();estimador.close.assert_called_once()
                self.assertEqual(self.snapshot(),self.previo)
                self.assertFalse(any(c.args[0]=='guardado' for c in puente.emitir.call_args_list))

    def test_comprobacion_final_dudosa_no_guarda_aunque_previa_permita(self):
        with ExitStack() as st:
            puente,cap,estimador,primero,ultimo=self.trabajador(st,final=self.comprobador(('dudoso',None)))
            registrar(puente,'AliasSintetico',0)
            primero.comprobar.assert_called_once();ultimo.comprobar.assert_called_once()
            cap.release.assert_called_once();estimador.close.assert_called_once()
        self.assertEqual(self.snapshot(),self.previo)
        self.assertFalse(any(c.args[0]=='guardado' for c in puente.emitir.call_args_list))

    def test_error_comprobador_final_no_autoriza_ni_deja_archivos(self):
        ultimo=self.comprobador();ultimo.comprobar.side_effect=RuntimeError('comprobador simulado')
        with ExitStack() as st:
            puente,cap,estimador,_,_=self.trabajador(st,final=ultimo)
            with self.assertRaisesRegex(RuntimeError,'comprobador simulado'):
                registrar(puente,'AliasSintetico',0)
            cap.release.assert_called_once();estimador.close.assert_called_once()
        self.assertEqual(self.snapshot(),self.previo)

    def test_fallos_guardado_transaccional_no_modifican_base_ni_original(self):
        abrir=Path.open
        for fase in ('segunda_foto','respaldo','serializacion','fsync','interrupcion'):
            with self.subTest(fase=fase),ExitStack() as st:
                st.enter_context(patch('interfaz.duplicados.VerificadorDuplicados',return_value=self.comprobador()))
                if fase=='segunda_foto':
                    def abrir_seguro(p,*args,**kwargs):
                        if p.name=='foto_002.jpg':raise OSError('escritura de foto simulada')
                        return abrir(p,*args,**kwargs)
                    st.enter_context(patch.object(Path,'open',abrir_seguro))
                elif fase=='respaldo':
                    st.enter_context(patch('interfaz.registro.shutil.copyfileobj',side_effect=OSError('respaldo simulado')))
                elif fase in ('serializacion','interrupcion'):
                    error=KeyboardInterrupt() if fase=='interrupcion' else OSError('serializacion simulada')
                    st.enter_context(patch('interfaz.registro.pickle.dump',side_effect=error))
                else:
                    st.enter_context(patch('interfaz.registro.os.fsync',side_effect=OSError('fsync simulado')))
                with self.assertRaises((OSError,KeyboardInterrupt)):
                    guardar_persona('AliasSintetico',self.jpegs,self.db,self.fotos)
                self.assertEqual(self.snapshot(),self.previo)

    def test_verificacion_del_temporal_incorrecta_revierte_todo(self):
        leer=leer_base
        def lectura(p):
            return {'nombres':[],'imagenes':[]} if Path(p).suffix=='.tmp' else leer(p)
        with patch('interfaz.duplicados.VerificadorDuplicados',return_value=self.comprobador()), \
             patch('interfaz.registro.leer_base',side_effect=lectura):
            with self.assertRaisesRegex(ValueError,'verificar'):
                guardar_persona('AliasSintetico',self.jpegs,self.db,self.fotos)
        self.assertEqual(self.snapshot(),self.previo)


if __name__=='__main__':
    unittest.main()
