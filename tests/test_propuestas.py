import copy
import json
import unittest
from unittest.mock import patch

from propuestas_movimientos.preparar import propuestas, validar, CARPETA
from propuestas_movimientos.ensayo import simular, main
from secuencias.reproductor import resolver_json
from vision.manos import GESTO_ACCION


class PropuestasTests(unittest.TestCase):
    def test_nueve_archivos_coinciden_con_gestos_y_ruta_absoluta(self):
        self.assertEqual(set(propuestas()), {nombre for nombre, _ in GESTO_ACCION.values()})
        total = 0
        with patch("serial.Serial", side_effect=AssertionError("Hardware prohibido")), \
             patch("cv2.VideoCapture", side_effect=AssertionError("Camara prohibida")):
            for nombre in propuestas():
                ruta = CARPETA/nombre
                datos = json.loads(ruta.read_text(encoding="utf-8"))
                self.assertEqual(datos, sorted(propuestas()[nombre], key=lambda b: b[2]))
                validar(datos)
                total += simular(ruta)
                self.assertEqual(resolver_json(nombre), str(ruta.resolve()))
        self.assertEqual(total, 97)

    def test_rechaza_posicion_fuera_de_rango_y_canal_desconocido(self):
        for bloque in ([2, 4, 0, 0, 400, 3], [9, 12, 0, 0, 300, 3], [0, 1, 0, 0, 500, 3]):
            with self.assertRaises(ValueError):
                validar([bloque])

    def test_rechaza_colisiones_tiempo_y_formato(self):
        for bloques in ([[2, 12, -1, 0, 420, 3]], [[2, 12, 0, 420, 3]],
                         [[2, 12, 0, 0, 420, 3], [2, 12, 0, 0, 600, 3]]):
            with self.assertRaises(ValueError):
                validar(bloques)

    def test_cancelacion_fisica_no_abre_arduino(self):
        with patch("sys.argv", ["ensayo.py", "parpadeo_doble.json", "--fisico", "--puerto", "COM7"]), \
             patch("builtins.input", return_value="CANCELAR"), \
             patch("serial.Serial", side_effect=AssertionError("Hardware prohibido")) as puerto:
            main()
        puerto.assert_not_called()


if __name__ == "__main__":
    unittest.main()
