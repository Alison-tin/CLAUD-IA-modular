import os
import cv2
import pickle
import ntpath
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from configuracion.config import *
from hardware.camara import abrir_camara



# DETECTOR FACIAL MEDIAPIPE


_detector_cara = None


def obtener_detector_cara():
    global _detector_cara

    if _detector_cara is not None:
        return _detector_cara

    if not os.path.exists(MODELO_CARA_PATH):
        print(f" Modelo facial no encontrado: {MODELO_CARA_PATH}")
        return None

    try:
        base_options = mp_python.BaseOptions(
            model_asset_path=MODELO_CARA_PATH
        )

        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5
        )

        _detector_cara = mp_vision.FaceDetector.create_from_options(
            options
        )

        print(" Detector facial MediaPipe listo")

        return _detector_cara

    except Exception as e:
        print(f" Error creando detector facial: {e}")
        return None



# DETECTAR ROSTROS


def detectar_rostros(frame):
    """
    Detecta rostros utilizando MediaPipe.

    Retorna:
        [(x, y, w, h), ...]
    """

    detector = obtener_detector_cara()

    if detector is None:
        return []

    try:
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagen_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        resultado = detector.detect(
            imagen_mp
        )

        rostros = []

        h_img, w_img = frame.shape[:2]

        for deteccion in resultado.detections:

            bb = deteccion.bounding_box

            x = max(
                0,
                int(bb.origin_x)
            )

            y = max(
                0,
                int(bb.origin_y)
            )

            w = int(bb.width)
            h = int(bb.height)

            if x + w > w_img:
                w = w_img - x

            if y + h > h_img:
                h = h_img - y

            if w > 0 and h > 0:
                rostros.append(
                    (x, y, w, h)
                )

        return rostros

    except Exception as e:
        print(f" Error detectando rostro: {e}")
        return []



# EXTRAER ROSTRO


def extraer_rostro(frame, rostro, diagnostico=None):
    """
    Extrae el rostro y lo convierte a escala de grises.
    """

    x, y, w, h = rostro

    # El detector del registro limita la caja antes de extraer. Aplicar
    # el mismo contrato a las cajas crudas del detector en vivo.
    x, y = max(0, int(x)), max(0, int(y))
    w = min(int(w), frame.shape[1] - x)
    h = min(int(h), frame.shape[0] - y)
    if w <= 0 or h <= 0:
        return None

    margen = int(
        min(w, h) * 0.15
    )

    x1 = max(
        0,
        x - margen
    )

    y1 = max(
        0,
        y - margen
    )

    x2 = min(
        frame.shape[1],
        x + w + margen
    )

    y2 = min(
        frame.shape[0],
        y + h + margen
    )

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return None

    gris = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gris = cv2.resize(
        gris,
        (200, 200)
    )

    if diagnostico is not None:
        diagnostico.update(recorte=(x1, y1, x2, y2),
                           gris_media_antes_eq=float(gris.mean()),
                           gris_std_antes_eq=float(gris.std()),
                           laplaciano_antes_eq=float(cv2.Laplacian(gris, cv2.CV_64F).var()))

    
    gris = cv2.equalizeHist(
        gris
    )

    return gris



# BASE DE DATOS


def cargar_db():

    if os.path.exists(DB_PATH):

        try:

            with open(
                DB_PATH,
                "rb"
            ) as f:

                db = pickle.load(f)

            if "nombres" not in db:
                db["nombres"] = []

            if "imagenes" not in db:
                db["imagenes"] = []

            return db

        except Exception as e:

            print(
                f" Error cargando base de datos: {e}"
            )

    return {
        "nombres": [],
        "imagenes": []
    }


def guardar_db(db):

    carpeta = os.path.dirname(
        DB_PATH
    )

    if carpeta:
        os.makedirs(
            carpeta,
            exist_ok=True
        )

    with open(
        DB_PATH,
        "wb"
    ) as f:

        pickle.dump(
            db,
            f
        )

    personas = len(
        set(db["nombres"])
    )

    print(
        f" {personas} personas guardadas."
    )


def listar_personas(db):

    personas = sorted(
        set(db["nombres"])
    )

    if personas:

        print(
            f"\n Registradas ({len(personas)}): "
            + ", ".join(personas)
        )

    else:

        print(
            "\n  Sin personas registradas"
        )

    return personas



# REGISTRO DE PERSONA


def registrar_persona(nombre, cam_idx=0):
    """Entrada historica: delega en el registro protegido compartido con la web."""
    from interfaz.terminal import registrar_terminal
    return registrar_terminal(nombre, cam_idx)


# MENSAJE DE CARGA


print(
    " Módulo de personas cargado correctamente"
)


# ENTRENAR RECONOCIMIENTO FACIAL


def entrenar_reconocedor(muestras_debug=None, db=None, fotos_dir=None, persona_comparacion="Alexa",
                        entradas_debug=None):
    from interfaz.duplicados import resolver_referencia

    if db is None:
        db = cargar_db()
    if fotos_dir is None:
        fotos_dir = FOTOS_DIR

    if len(db["nombres"]) != len(db["imagenes"]):
        raise ValueError("Base inconsistente: nombres e imagenes tienen distinta longitud")

    if not db["imagenes"]:
        print(" No hay imágenes para entrenar.")
        return None, {}

    if not hasattr(cv2, "face"):
        print(" OpenCV Face no está disponible.")
        return None, {}

    if not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print(" LBPH no está disponible.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    rostros = []
    etiquetas = []

    nombres_por_id = {}
    ids_por_nombre = {}

    siguiente_id = 0

    for nombre, ruta in zip(
        db["nombres"],
        db["imagenes"]
    ):

        try:
            ruta = str(resolver_referencia(ruta, fotos_dir))
        except (OSError, ValueError) as error:
            print(f" No se pudo localizar muestra de {nombre}: {error}")
            continue

        imagen = cv2.imread(
            ruta,
            cv2.IMREAD_GRAYSCALE
        )

        if imagen is None:
            continue

        imagen = cv2.resize(
            imagen,
            (200, 200)
        )

        # Las fotos persistidas son la salida gris/ecualizada de extraer_rostro.
        # No volver a ecualizar el JPEG: cambia la distribucion frente al vivo.

        if nombre not in ids_por_nombre:

            ids_por_nombre[nombre] = siguiente_id
            nombres_por_id[siguiente_id] = nombre

            siguiente_id += 1

        etiqueta = ids_por_nombre[nombre]

        rostros.append(imagen)
        etiquetas.append(etiqueta)

        if entradas_debug is not None:
            entradas_debug.append({'nombre': nombre, 'ruta': ruta,
                                   'etiqueta': etiqueta, 'imagen': imagen.copy()})

        # Copia para la comparacion temporal; no modifica la muestra entrenada.
        if muestras_debug is not None and nombre.casefold() == persona_comparacion.casefold():
            if "imagen" not in muestras_debug:
                muestras_debug["imagen"] = imagen.copy()
                muestras_debug["ruta"] = ruta
                muestras_debug["nombre"] = nombre

    if not rostros:

        print(
            " No se encontraron imágenes válidas."
        )

        return None, {}

    etiquetas = np.array(
        etiquetas,
        dtype=np.int32
    )

    print(
        f" Entrenando con {len(rostros)} muestras..."
    )

    recognizer.train(
        rostros,
        etiquetas
    )

    print(
        f" Reconocedor listo: "
        f"{len(nombres_por_id)} personas"
    )

    return recognizer, nombres_por_id
