import ast
import hashlib
import pickle
from pathlib import Path
from queue import Queue
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from interfaz.registro import leer_base, validar_nombre, guardar_persona
from interfaz.trabajador import Puente, registrar
from hardware.exclusion import Exclusivo

RAIZ = Path(__file__).resolve().parents[1]


def setUpModule():
    global guardia_camara, guardia_serial
    guardia_camara = patch("cv2.VideoCapture", side_effect=AssertionError("Camara real prohibida en pruebas"))
    guardia_serial = patch("serial.Serial", side_effect=AssertionError("Arduino prohibido en pruebas"))
    guardia_camara.start()
    guardia_serial.start()


def tearDownModule():
    guardia_camara.stop()
    guardia_serial.stop()


class RegistroTests(unittest.TestCase):
    def test_rechaza_nombre_existente_y_base_inconsistente(self):
        base = {"nombres": ["Alison"], "imagenes": ["original.jpg"]}
        with self.assertRaises(ValueError):
            validar_nombre(" ALISON ", base)
        with self.assertRaises(ValueError):
            validar_nombre("\n", base)
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "base.pkl"
            ruta.write_bytes(pickle.dumps({"nombres": ["Alison"], "imagenes": []}))
            with self.assertRaises(ValueError):
                leer_base(ruta)

    def test_alta_5_sin_modificar_fotos_anteriores_y_con_respaldo(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = Path(d)
            fotos = raiz / "fotos"
            fotos.mkdir()
            foto = fotos / "Alison_001.jpg"
            foto.write_bytes(b"original intocable")
            db = raiz / "base.pkl"
            anterior = {"nombres": ["Alison"], "imagenes": [str(foto)]}
            db.write_bytes(pickle.dumps(anterior))
            contenido = db.read_bytes()
            # Esta prueba verifica persistencia, no la decision biometrica.
            with patch("interfaz.duplicados.VerificadorDuplicados") as verificador:
                verificador.return_value.comprobar.return_value = ("sin_coincidencia", None)
                guardar_persona("Beatriz", [b"jpeg prueba"] * 5, db, fotos)
            nueva = leer_base(db)
            self.assertEqual(nueva["nombres"], ["Alison"] + ["Beatriz"] * 5)
            self.assertEqual(nueva["imagenes"][0], str(foto))
            self.assertEqual(foto.read_bytes(), b"original intocable")
            self.assertEqual(len(set(nueva["imagenes"])), 6)
            self.assertTrue(all(Path(p).is_file() for p in nueva["imagenes"]))
            self.assertEqual(next(raiz.glob("*.antes_registro_*")).read_bytes(), contenido)
            despues = db.read_bytes()
            with self.assertRaises(ValueError):
                guardar_persona("BEATRIZ", [b"otra"] * 5, db, fotos)
            self.assertEqual(db.read_bytes(), despues)

    def test_incompleta_no_escribe(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                guardar_persona("Nueva", [b"jpeg"], Path(d)/"base", Path(d)/"fotos")
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_cancelar_registro_no_guarda_y_libera_camara_simulada(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        puente = MagicMock()
        puente.detener.side_effect = [False, True, True]
        puente.comandos = Queue()
        with patch("interfaz.registro.leer_base", return_value={"nombres": [], "imagenes": []}), \
             patch("hardware.camara.abrir_camara", return_value=cap), \
             patch("datos.personas.detectar_rostros", return_value=[]), \
             patch("interfaz.registro.guardar_persona") as guardar, \
             patch("vision.orientacion.EstimadorOrientacion") as orientacion:

            from vision.orientacion import Observacion
            orientacion.return_value.analizar.return_value = Observacion(0)
            registrar(puente, "Nueva", 0)
        guardar.assert_not_called()
        cap.release.assert_called_once()

    def test_recorrido_completo_simulado_incorpora_5_fotos(self):
        from vision.orientacion import Observacion
        angulos = [(0,0,0), (15,0,0), (-15,0,0), (0,15,0), (0,-15,0)]
        observaciones = [Observacion(1, a, np.random.default_rng(i).integers(0,256,(200,200),dtype=np.uint8), 100)
                         for i, a in enumerate(angulos) for _ in range(6)]
        cap, puente, estimador = MagicMock(), MagicMock(), MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = True, np.zeros((400,400,3), np.uint8)
        puente.detener.return_value = False
        puente.comandos = Queue()
        estimador.analizar.side_effect = observaciones
        with tempfile.TemporaryDirectory() as d:
            db, fotos = Path(d)/"base.pkl", Path(d)/"fotos"
            with patch("configuracion.config.DB_PATH", str(db)), patch("configuracion.config.FOTOS_DIR", str(fotos)), \
                 patch("hardware.camara.abrir_camara", return_value=cap), \
                 patch("vision.orientacion.EstimadorOrientacion", return_value=estimador), \
                 patch("interfaz.trabajador.time.monotonic", side_effect=iter(np.arange(0,100,.2))), \
                 patch("interfaz.trabajador.time.sleep"):
                registrar(puente, "Nueva", 0)
            base = leer_base(db)
            self.assertEqual(base["nombres"], ["Nueva"]*5)
            self.assertEqual(len(list(fotos.rglob('*.jpg'))), 5)
            puente.emitir.assert_any_call("guardado", "Nueva")
        cap.release.assert_called_once()
        estimador.close.assert_called_once()


class RecursosTests(unittest.TestCase):
    def test_exclusion_y_liberacion(self):
        nombre = "test_" + next(tempfile._get_candidate_names())
        a, b = Exclusivo(nombre), Exclusivo(nombre)
        try:
            a.adquirir()
            with self.assertRaises(RuntimeError):
                b.adquirir()
            a.liberar()
            b.adquirir()
        finally:
            a.liberar()
            b.liberar()

    def test_camara_no_abre_dos_veces_y_puede_reabrir(self):
        from hardware.camara import abrir_camara
        cap = MagicMock()
        cap.isOpened.return_value = True
        with patch("hardware.camara._abrir_camara", return_value=cap) as abrir:
            primera = abrir_camara()
            try:
                with self.assertRaises(RuntimeError):
                    abrir_camara()
                self.assertEqual(abrir.call_count, 1)
            finally:
                primera.release()
            abrir_camara().release()

    def test_puente_opciones_parada_y_cola_acotada(self):
        eventos, cuadros, comandos = Queue(10), Queue(1), Queue()
        parada = threading.Event()
        puente = Puente(eventos, cuadros, comandos, parada)
        comandos.put(("opciones", {"gestos": False}))
        self.assertFalse(puente.opciones()["gestos"])
        puente.emitir("camara", "Activa")
        puente.emitir("camara", "Activa")
        self.assertEqual(eventos.qsize(), 1)
        puente.fotograma(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(cuadros.qsize(), 1)
        parada.set()
        self.assertTrue(puente.detener())

    def test_limpieza_del_sistema_ante_error_sin_movimientos(self):
        from aplicacion.sistema import ejecutar
        cap, cara, mano, serial = [MagicMock() for _ in range(4)]
        def fallo(cam_idx, port, demo, comparar_rostro, interfaz, recursos, *opciones):
            recursos.update(cap=cap, cara=cara, mano=mano, serial=serial)
            raise RuntimeError("fallo simulado")
        with patch("aplicacion.sistema._ejecutar", side_effect=fallo):
            with self.assertRaises(RuntimeError):
                ejecutar(interfaz=MagicMock())
        cap.release.assert_called_once()
        cara.close.assert_called_once()
        mano.close.assert_called_once()
        serial.close.assert_called_once()


class ProteccionMovimientosTests(unittest.TestCase):
    def test_configuracion_y_modulos_intactos(self):
        esperados = {
            "configuracion/config.py": "eda158302eb84aacb8a9067735c466f5e4536633d48d47272e7d45355fdb170f",
            # Cambios autorizados: saludo dentro de limites, esperas cancelables y arbitraje.
            "hardware/arduino.py": "02b05a921e2cdfb9f49720dfd67adc81318895310c58e64a75c983f5b4a11d9f",
            "movimientos/expresiones.py": "db02e0cc491ee74f4e0577eebc95f3d749d720c0b4f5a46f400e2d6339b4f51f",
            "movimientos/gestos.py": "1b6762a2493cd688f447bb045ac740a993c089575b50a3123ea5167f683afb3b",
            "control/pid.py": "ec1db78c75b814fa17bdbcc19e1720e80af486dfb01fd4763c05df79dc802083",
        }
        for nombre, huella in esperados.items():
            self.assertEqual(hashlib.sha256((RAIZ/nombre).read_bytes()).hexdigest(), huella, nombre)

    def test_mismas_ordenes_arduino_en_sistema(self):
        def llamadas(nombre):
            arbol = ast.parse((RAIZ/nombre).read_text(encoding="utf-8-sig"))
            resultado=[]
            for n in ast.walk(arbol):
                if (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                        and isinstance(n.func.value,ast.Name) and n.func.value.id in ('arduino','a')
                        and n.func.attr != 'esperar'):
                    n.func.value.id='arduino'  # proxy cancelable conserva las ordenes
                    resultado.append(ast.dump(n))
            return resultado
        self.assertEqual(llamadas("inmoov_vision8.py"), llamadas("aplicacion/sistema.py"))


if __name__ == "__main__":
    unittest.main()
