"""Orientacion del rostro para altas; independiente del reconocedor y Arduino.

Analisis SIEMPRE sin espejo. Matriz canonica MediaPipe, ejes x derecha de
imagen, y arriba, z hacia observador. Yaw positivo = derecha de la persona;
pitch positivo = arriba. La vista previa se refleja despues del analisis.
Los limites de calidad son parametros iniciales de captura, no de identidad.
"""
from dataclasses import dataclass, field
from pathlib import Path
import math

import cv2
import numpy as np

MODELO = Path(__file__).resolve().parents[1]/'face_landmarker.task'


@dataclass
class Observacion:
    rostros: int
    angulos: tuple = None
    imagen: object = None
    nitidez: float = 0
    motivo: str = ''
    diagnostico: dict = field(default_factory=dict)


def angulos_persona(matriz):
    m = np.asarray(matriz, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError('Matriz facial no valida')
    a = m[:3, :3]
    u, s, vt = np.linalg.svd(a)
    if min(s) <= 0 or max(s)/min(s) > 1.3 or np.linalg.det(a) <= 0:
        raise ValueError('Transformacion facial no valida')
    r = u @ vt  # Retirar escala; conservar la rotacion rigida.
    x = math.atan2(r[2, 1], r[2, 2])
    y = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    z = math.atan2(r[1, 0], r[0, 0])
    return tuple(math.degrees(v) for v in (-y, -x, z))


class EstimadorOrientacion:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        if not MODELO.is_file():
            raise RuntimeError('Falta face_landmarker.task. Coloca el modelo en la carpeta principal de CLAUD-IA.')
        self.mp = mp
        self.modelo = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODELO)),
            running_mode=vision.RunningMode.IMAGE, num_faces=2,
            min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
            output_facial_transformation_matrixes=True))

    def analizar(self, frame):
        from datos.personas import detectar_rostros, extraer_rostro
        cajas = detectar_rostros(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = self.modelo.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        puntos = resultado.face_landmarks
        cantidad = max(len(cajas), len(puntos))
        d = {'detector_rostros': len(cajas), 'landmarker_rostros': len(puntos),
             'analisis_en_espejo': False, 'vista_previa_en_espejo': True,
             'limites_calidad': {'lado_minimo_px': 120, 'luz_media': [45, 210],
                                 'contraste_minimo': 18, 'nitidez_minima': 65}}
        def rechazar(codigo, mensaje, angulos=None, nitidez=0):
            d['bloqueo'] = codigo
            return Observacion(cantidad, angulos, None, nitidez, mensaje, d)
        if len(cajas) != 1 or len(puntos) != 1:
            mensaje = ('Hay varias personas: deja un solo rostro en la imagen.' if cantidad > 1 else
                       'No se confirma un rostro con ambos detectores. Mira hacia la camara con el rostro completo.')
            return rechazar('rostros', mensaje)
        if len(puntos[0]) < 468 or len(resultado.facial_transformation_matrixes) != 1:
            return rechazar('puntos', 'No se pudo estimar la orientacion. Mantén el rostro visible.')
        try:
            angulos = angulos_persona(resultado.facial_transformation_matrixes[0])
        except (ValueError, np.linalg.LinAlgError):
            return rechazar('matriz', 'No se pudo estimar una orientacion valida.')
        d['angulos_grados'] = dict(zip(('derecha', 'arriba', 'inclinacion_lateral'), map(lambda x: round(x, 2), angulos)))
        # Comprobacion visual auxiliar del sentido del giro; no decide la postura.
        ojo_a, ojo_b, nariz = puntos[0][33], puntos[0][263], puntos[0][1]
        distancia_ojos = math.hypot(ojo_b.x-ojo_a.x, ojo_b.y-ojo_a.y)
        if distancia_ojos > 0 and all(math.isfinite(v) for v in (nariz.x, ojo_a.x, ojo_b.x)):
            d['nariz_x_respecto_ojos_sin_espejo'] = round((nariz.x-(ojo_a.x+ojo_b.x)/2)/distancia_ojos, 3)
        h, w = frame.shape[:2]
        x, y, ancho, alto = cajas[0]
        d['tamano_rostro_px'] = [ancho, alto]
        if min(ancho, alto) < 120:
            return rechazar('tamano', 'Acércate un poco: el rostro es demasiado pequeno.', angulos)
        if (x < 2 or y < 2 or x+ancho >= w-2 or y+alto >= h-2
                or any(not (0 < p.x < 1 and 0 < p.y < 1) for p in puntos[0])):
            return rechazar('encuadre', 'Centra el rostro completo, sin acercarlo al borde de la imagen.', angulos)
        gris = cv2.cvtColor(frame[y:y+alto, x:x+ancho], cv2.COLOR_BGR2GRAY)
        gris = cv2.resize(gris, (200, 200))
        nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
        luz, contraste = float(gris.mean()), float(gris.std())
        d.update(luz_media=round(luz, 2), contraste=round(contraste, 2), nitidez=round(nitidez, 2))
        d['filtros'] = {'luz': 45 <= luz <= 210, 'contraste': contraste >= 18, 'nitidez': nitidez >= 65}
        if luz < 45:
            return rechazar('luz_baja', 'Falta luz en el rostro. Busca luz uniforme de frente.', angulos, nitidez)
        if luz > 210:
            return rechazar('luz_alta', 'Hay demasiada luz. Evita luz directa o reflejos fuertes.', angulos, nitidez)
        if contraste < 18:
            return rechazar('contraste', 'El rostro tiene poco contraste. Ajusta la iluminacion frontal.', angulos, nitidez)
        if nitidez < 65:
            return rechazar('nitidez', 'La imagen no es nitida. Quédate quieto y espera el enfoque; puedes mantener tus gafas.', angulos, nitidez)
        imagen = extraer_rostro(frame, cajas[0])
        d['bloqueo'] = '' if imagen is not None else 'recorte'
        return Observacion(1, angulos, imagen, nitidez,
                           '' if imagen is not None else 'No se pudo extraer el rostro. Reencuadra ligeramente.', d)

    def close(self):
        self.modelo.close()
