import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from interfaz.bajas import retirar_persona, recuperar_persona, carpeta_respaldos
from interfaz.registro import leer_base
from hardware.exclusion import Exclusivo
from web_local.control import Control


class BajasTests(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.raiz = Path(self.temporal.name)
        self.db = self.raiz / "configuracion" / "base.pkl"
        self.fotos = self.db.parent / "fotos"
        self.fotos.mkdir(parents=True)
        import cv2
        import numpy as np
        self.imagenes = []
        for i in range(3):
            ruta = self.fotos / f"foto_{i}.jpg"
            ok, jpeg = cv2.imencode(".jpg", np.random.default_rng(i).integers(0,256,(200,200),dtype=np.uint8))
            self.assertTrue(ok)
            ruta.write_bytes(jpeg.tobytes())
            self.imagenes.append(str(ruta))
        self.base = {"nombres": ["Persona prueba", "Otra persona", "Persona prueba"],
                     "imagenes": self.imagenes, "metadato": "conservar"}
        self.db.write_bytes(pickle.dumps(self.base))
        self.previa = self.db.read_bytes()
        self.fotos_previas = {p: Path(p).read_bytes() for p in self.imagenes}
        self.guardias = [patch("cv2.VideoCapture", side_effect=AssertionError("Camara prohibida")),
                         patch("serial.Serial", side_effect=AssertionError("Arduino prohibido"))]
        for guardia in self.guardias:
            guardia.start()

    def tearDown(self):
        for guardia in self.guardias:
            guardia.stop()
        self.temporal.cleanup()

    def test_retiro_respaldado_sin_borrar_fotos_ni_otra_persona(self):
        identificador = retirar_persona("Persona prueba", self.db, self.fotos)
        respaldo = carpeta_respaldos(self.db) / identificador
        self.assertEqual((respaldo/"base_antes.pkl").read_bytes(), self.previa)
        manifiesto = json.loads((respaldo/"registro.json").read_text(encoding="utf-8"))
        self.assertEqual(manifiesto["cantidad"], 2)
        for entrada, i in zip(manifiesto["fotos"], (0, 2)):
            self.assertEqual((respaldo/entrada["archivo"]).read_bytes(), self.fotos_previas[self.imagenes[i]])
        for p, contenido in self.fotos_previas.items():
            self.assertEqual(Path(p).read_bytes(), contenido)
        self.assertEqual(leer_base(self.db), {"nombres": ["Otra persona"], "imagenes": [self.imagenes[1]], "metadato": "conservar"})

    def test_recupera_solo_persona_sin_sobrescribir_y_rechaza_duplicado(self):
        identificador = retirar_persona("Persona prueba", self.db, self.fotos)
        actual = leer_base(self.db)
        actual["nombres"].append("Alta posterior")
        actual["imagenes"].append(self.imagenes[1])
        self.db.write_bytes(pickle.dumps(actual))
        recuperar_persona(identificador, self.db, self.fotos)
        nueva = leer_base(self.db)
        self.assertEqual(nueva["nombres"], ["Otra persona", "Alta posterior", "Persona prueba", "Persona prueba"])
        self.assertTrue(all(Path(p).is_file() for p in nueva["imagenes"]))
        self.assertNotIn(nueva["imagenes"][-1], self.imagenes)
        self.assertEqual(Path(nueva["imagenes"][-1]).read_bytes(), self.fotos_previas[self.imagenes[2]])
        previa = self.db.read_bytes()
        with self.assertRaises(ValueError):
            recuperar_persona(identificador, self.db, self.fotos)
        self.assertEqual(self.db.read_bytes(), previa)

    def test_foto_ausente_impide_retiro(self):
        base = dict(self.base)
        base["imagenes"] = ["ausente.jpg", self.imagenes[1], self.imagenes[2]]
        self.db.write_bytes(pickle.dumps(base))
        previa = self.db.read_bytes()
        with self.assertRaises(ValueError):
            retirar_persona("Persona prueba", self.db, self.fotos)
        self.assertEqual(self.db.read_bytes(), previa)

    def test_copia_o_actualizacion_fallida_no_retira_registro(self):
        with patch("interfaz.bajas.escribir_nuevo", side_effect=OSError("disco simulado")):
            with self.assertRaises(OSError):
                retirar_persona("Persona prueba", self.db, self.fotos)
        self.assertEqual(self.db.read_bytes(), self.previa)
        with patch("interfaz.bajas.os.replace", side_effect=OSError("reemplazo simulado")):
            with self.assertRaises(OSError):
                retirar_persona("Persona prueba", self.db, self.fotos)
        self.assertEqual(self.db.read_bytes(), self.previa)

    def test_respaldo_alterado_no_recupera(self):
        identificador = retirar_persona("Persona prueba", self.db, self.fotos)
        respaldo = carpeta_respaldos(self.db)/identificador
        foto = next((respaldo/"fotos").iterdir())
        foto.write_bytes(b"alteracion de prueba")
        previa = self.db.read_bytes()
        with self.assertRaises(ValueError):
            recuperar_persona(identificador, self.db, self.fotos)
        self.assertEqual(self.db.read_bytes(), previa)

    def test_no_retiro_con_trabajador_o_recurso_ocupado(self):
        control = Control(self.db, self.fotos)
        control.proceso = MagicMock()
        with self.assertRaises(ValueError):
            control.eliminar("Persona prueba")
        control.proceso = None
        with Exclusivo("escritorio"):
            with self.assertRaises(ValueError):
                control.eliminar("Persona prueba")
        self.assertEqual(self.db.read_bytes(), self.previa)

    def test_control_actualiza_lista_y_avisa_reentrenamiento(self):
        control = Control(self.db, self.fotos)
        control.estado["nombre"] = "Persona prueba"
        with patch.object(control,'_preparar_galeria_retiro') as preparar:
            control.eliminar("Persona prueba")
        preparar.assert_called_once()
        self.assertEqual(control.personas(), [{"nombre": "Otra persona", "fotos": 1}])
        self.assertIsNone(control.ver()["nombre"])
        self.assertIn("galería actualizada", control.ver()["aviso"])

    def test_entrenamiento_tras_retiro_excluye_persona(self):
        from datos.personas import entrenar_reconocedor
        modelo, nombres = entrenar_reconocedor(db=leer_base(self.db), fotos_dir=str(self.fotos))
        self.assertEqual(len(modelo.getLabels()), 3)
        self.assertIn("Persona prueba", nombres.values())
        retirar_persona("Persona prueba", self.db, self.fotos)
        modelo, nombres = entrenar_reconocedor(db=leer_base(self.db), fotos_dir=str(self.fotos))
        self.assertEqual(len(modelo.getLabels()), 1)
        self.assertEqual(list(nombres.values()), ["Otra persona"])


if __name__ == "__main__":
    unittest.main()
