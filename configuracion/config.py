import os

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================

PCA = 2
BAUD = 115200

RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
SECUENCIAS_DIR = RAIZ_PROYECTO

# Captura comun para registro y reconocimiento. Nombres de constantes OpenCV.
CAMARA_BACKEND = "CAP_DSHOW"
# "predeterminada": no imponer controles de imagen ni FPS.
# "anterior": recuperar exactamente las solicitudes usadas anteriormente.
CAMARA_PERFIL = "predeterminada"
CAMARA_AJUSTES_PREDETERMINADOS = (
    ("CAP_PROP_FRAME_WIDTH", 1280),
    ("CAP_PROP_FRAME_HEIGHT", 720),
)
CAMARA_AJUSTES_ANTERIORES = (
    ("CAP_PROP_FRAME_WIDTH", 1280),
    ("CAP_PROP_FRAME_HEIGHT", 720),
    ("CAP_PROP_FPS", 30),
    ("CAP_PROP_BUFFERSIZE", 1),
    ("CAP_PROP_AUTOFOCUS", 1),
    ("CAP_PROP_AUTO_EXPOSURE", 1),
    ("CAP_PROP_BRIGHTNESS", 160),
    ("CAP_PROP_CONTRAST", 128),
    ("CAP_PROP_SATURATION", 128),
    ("CAP_PROP_SHARPNESS", 150),
    ("CAP_PROP_GAIN", 50),
    ("CAP_PROP_EXPOSURE", -3),
)
CAMARA_PERFILES = {
    "predeterminada": CAMARA_AJUSTES_PREDETERMINADOS,
    "anterior": CAMARA_AJUSTES_ANTERIORES,
}
CAMARA_AJUSTES = CAMARA_PERFILES[CAMARA_PERFIL]


# ======================================================
# SERVOMOTORES - PCA 2
# ======================================================

SERVO = {
    # canal: (mínimo, máximo, neutro)
    0: (340, 590, 465),
    1: (418, 510, 464),
    2: (440, 700, 570),
    3: (200, 450, 325),
    4: (410, 640, 525),
    5: (435, 665, 550),
    6: (470, 565, 517),
    7: (525, 625, 575),
    8: (370, 620, 495),
    9: (450, 660, 555),
    10: (570, 700, 635),
    11: (570, 700, 635),
    12: (420, 600, 510),
    13: (520, 655, 587),
    14: (435, 580, 507),
    15: (480, 620, 550),
}


# ======================================================
# CUELLO Y MAXILAR - PCA 0
# ======================================================

PCA0 = 0

SERVO_CUELLO = {
    # canal: (mínimo, máximo, neutro, velocidad, nombre, dirección mínima, dirección máxima)
    15: (300, 800, 530, 1, "rotacion_cabeza", "IZQ", "DER"),
    14: (250, 750, 550, 4, "inclinacion_cabeza", "DER", "IZQ"),
    12: (300, 650, 300, 5, "maxilar_inf_pca0", "sube", "baja"),
}


# ======================================================
# MAXILAR
# ======================================================

MAXILAR_PCA = 0
MAXILAR_CANAL = 12

MAXILAR_CIERRA = 300
MAXILAR_ABRE = 650


# ======================================================
# PÁRPADOS
# ======================================================

PARPADOS_ABIERTOS = {
    12: 600,
    13: 520,
    14: 580,
    15: 480
}

PARPADOS_CERRADOS = {
    12: 420,
    13: 655,
    14: 435,
    15: 620
}


# ======================================================
# PARÁMETROS DE MOVIMIENTO Y SEGUIMIENTO
# ======================================================

Kp, Ki, Kd = 0.20, 0.002, 0.08

ZONA_MUERTA_X = 30
ZONA_MUERTA_Y = 25

VEL_SEG = 4
VEL_EXP = 3

TIMEOUT_ROSTRO = 2.0

PARPADEO_INT = (3.0, 6.0)

TOLERANCIA_REC = 58.0

FOTOS_POR_PER = 30

RECONOCER_CADA = 8

COOLDOWN_GESTO = 2.5

FRAMES_CONF = 8


# ======================================================
# MODELOS DE MEDIAPIPE
# ======================================================

MODELO_CARA_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)

MODELO_CARA_PATH = os.path.join(RAIZ_PROYECTO, "blaze_face_short_range.tflite")

MODELO_MANO_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

MODELO_MANO_PATH = os.path.join(RAIZ_PROYECTO, "hand_landmarker.task")


# ======================================================
# BASE DE DATOS Y FOTOS
# ======================================================

DB_PATH = os.path.join(
    CONFIG_DIR,
    "rostros_conocidos.pkl"
)

FOTOS_DIR = os.path.join(
    CONFIG_DIR,
    "rostros_fotos"
)


# ======================================================
# EMOCIONES
# ======================================================

EMOCIONES_ES = {
    "happy": "Feliz",
    "sad": "Triste",
    "angry": "Enojado",
    "surprise": "Sorprendido",
    "fear": "Asustado",
    "disgust": "Disgustado",
    "neutral": "Neutro",
}


# ======================================================
# COLOR POR EMOCIÓN (BGR)
# ======================================================

COLOR_EMOCION = {
    "Feliz": (0, 220, 0),
    "Triste": (200, 100, 0),
    "Enojado": (0, 0, 220),
    "Sorprendido": (0, 200, 255),
    "Asustado": (180, 0, 255),
    "Disgustado": (0, 150, 150),
    "Neutro": (180, 180, 180),
}
