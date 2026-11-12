import logging
import numpy as np
import cv2

from configuracion.config import EMOCIONES_ES



# ESTADO DE DEEPFACE


_df_state = {
    "ok": False,
    "intentado": False
}



# CARGAR DEEPFACE


def _cargar_deepface():
    try:
        import absl.logging, logging

        absl.logging.set_verbosity(
            absl.logging.ERROR
        )

        logging.getLogger(
            "tensorflow"
        ).setLevel(
            logging.ERROR
        )

        from deepface import DeepFace

        # Test rápido para confirmar que funciona
        # y que los modelos están disponibles
        img_test = (
            np.zeros(
                (100, 100, 3),
                dtype=np.uint8
            ) + 128
        )

        DeepFace.analyze(
            img_test,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        _df_state["ok"] = True

        print(
            " DeepFace listo y funcional"
        )

    except ImportError:
        print(
            "  DeepFace no instalado  "
            "pip install deepface"
        )

    except Exception as e:
        print(
            f"  DeepFace error al cargar: {e}"
        )

    finally:
        _df_state["intentado"] = True



# ANÁLISIS DE EMOCIÓN


def analizar_emocion(
    frame_rgb,
    x1,
    y1,
    x2,
    y2
):
    """
    Devuelve (emocion_es, confianza)
    o (None, 0) si falla.
    """

    if not _df_state["ok"]:
        return None, 0

    try:

        from deepface import DeepFace

        roi = frame_rgb[
            max(0, y1):y2,
            max(0, x1):x2
        ]

        if roi.size == 0:
            return None, 0

        roi_bgr = cv2.cvtColor(
            roi,
            cv2.COLOR_RGB2BGR
        )

        res = DeepFace.analyze(
            roi_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        if isinstance(res, list):
            res = res[0]

        emo_en = res["dominant_emotion"]

        conf = (
            res["emotion"][emo_en]
            / 100.0
        )

        emo_es = EMOCIONES_ES.get(
            emo_en,
            emo_en
        )

        return emo_es, conf

    except Exception:
        return None, 0

def deepface_estado():
    return _df_state