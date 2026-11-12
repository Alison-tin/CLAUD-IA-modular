import hashlib
import json
from pathlib import Path
import pickle
from unittest.mock import patch

from tests.test_bajas import BajasTests
from interfaz.bajas import retirar_persona, carpeta_respaldos, escribir_nuevo
from interfaz.registro import leer_base


class RutasRespaldoTests(BajasTests):
    def test_rutas_trasladadas_nombre_repetido_y_otro_registro_intacto(self):
        nombres=['daniela','daniela perez','daniela']
        referencias=[]
        originales=[]
        for i in range(3):
            fuente=self.fotos/f'sesion_{i}'/'foto_001.jpg'
            fuente.parent.mkdir()
            fuente.write_bytes(self.fotos_previas[self.imagenes[i]])
            originales.append(fuente)
            referencias.append(r'C:\ubicacion_anterior\configuracion\rostros_fotos'+f'\\sesion_{i}\\foto_001.jpg')
        # Una foto homónima en la raíz nunca debe sustituir una foto de sesión.
        (self.fotos/'foto_001.jpg').write_bytes(b'foto equivocada de la raiz')
        base={'nombres':nombres,'imagenes':referencias,'metadato':'conservar'}
        for seleccionado in ('daniela','daniela perez'):
            with self.subTest(seleccionado=seleccionado):
                previa=pickle.dumps(base);self.db.write_bytes(previa)
                identificador=retirar_persona(seleccionado,self.db,self.fotos)
                respaldo=carpeta_respaldos(self.db)/identificador
                informe=json.loads((respaldo/'registro.json').read_text(encoding='utf-8'))
                indices=[i for i,n in enumerate(nombres) if n==seleccionado]
                self.assertEqual(informe['cantidad'],len(indices))
                self.assertEqual(hashlib.sha256((respaldo/'base_antes.pkl').read_bytes()).hexdigest(),informe['base_sha256'])
                for foto,i in zip(informe['fotos'],indices):
                    contenido=originales[i].read_bytes()
                    self.assertEqual((respaldo/foto['archivo']).read_bytes(),contenido)
                    self.assertEqual(hashlib.sha256(contenido).hexdigest(),foto['sha256'])
                    self.assertEqual(foto['referencia_original'],referencias[i])
                    self.assertEqual(foto['origen'],str(originales[i].resolve()))
                    self.assertEqual(foto['indice_base'],i)
                restantes=[i for i,n in enumerate(nombres) if n!=seleccionado]
                self.assertEqual(leer_base(self.db),{'nombres':[nombres[i] for i in restantes],
                    'imagenes':[referencias[i] for i in restantes],'metadato':'conservar'})
                for i,p in enumerate(originales):
                    self.assertEqual(p.read_bytes(),self.fotos_previas[self.imagenes[i]])
                self.assertEqual((self.fotos/'foto_001.jpg').read_bytes(),b'foto equivocada de la raiz')

    def test_foto_de_sesion_ausente_no_se_sustituye_por_homonima(self):
        base=dict(self.base)
        referencia=r'C:\anterior\rostros_fotos\sesion_ausente\foto_001.jpg'
        base['imagenes']=[referencia,*self.imagenes[1:]]
        (self.fotos/'foto_001.jpg').write_bytes(b'no es la foto referenciada')
        self.db.write_bytes(pickle.dumps(base));previa=self.db.read_bytes()
        with self.assertRaisesRegex(ValueError,'sesion_ausente'):
            retirar_persona('Persona prueba',self.db,self.fotos)
        self.assertEqual(self.db.read_bytes(),previa)
        self.assertFalse(carpeta_respaldos(self.db).exists())

    def test_sha256_final_detecta_copia_alterada_y_conserva_registro(self):
        for objetivo in ('foto','base','manifiesto'):
            with self.subTest(objetivo=objetivo):
                def escribir_y_alterar(ruta,datos):
                    escribir_nuevo(ruta,datos)
                    if ruta.name=='registro.json':
                        if objetivo=='foto':
                            destino=next((ruta.parent/'fotos').iterdir())
                            destino.write_bytes(b'copia corrupta simulada')
                        elif objetivo=='base':
                            (ruta.parent/'base_antes.pkl').write_bytes(b'base corrupta simulada')
                        else:
                            ruta.write_text('{}',encoding='utf-8')
                with patch('interfaz.bajas.escribir_nuevo',side_effect=escribir_y_alterar):
                    with self.assertRaisesRegex(ValueError,'El registro se conserva'):
                        retirar_persona('Persona prueba',self.db,self.fotos)
                self.assertEqual(self.db.read_bytes(),self.previa)
                for p,contenido in self.fotos_previas.items():
                    self.assertEqual(Path(p).read_bytes(),contenido)

    def test_fallo_segunda_foto_no_permite_actualizacion(self):
        def escribir_fallando(ruta,datos):
            if ruta.parent.name=='fotos' and ruta.name.startswith('foto_002'):
                raise OSError('fallo de disco simulado en la segunda foto')
            escribir_nuevo(ruta,datos)
        with patch('interfaz.bajas.escribir_nuevo',side_effect=escribir_fallando), patch('interfaz.bajas.actualizar') as actualizar:
            with self.assertRaises(OSError): retirar_persona('Persona prueba',self.db,self.fotos)
        actualizar.assert_not_called()
        self.assertEqual(self.db.read_bytes(),self.previa)
        for p,contenido in self.fotos_previas.items():
            self.assertEqual(Path(p).read_bytes(),contenido)
