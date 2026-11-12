"""Contrato de entrada para los detectores; no abre dispositivos."""
import cv2
import numpy as np


def validar_bgr(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3
            or min(frame.shape[:2]) <= 0):
        raise ValueError('Se esperaba uint8 BGR HxWx3 con dimensiones positivas')


def imagen_mediapipe(frame):
    import mediapipe as mp
    validar_bgr(frame)
    # Copia propia por llamada: admite vistas, espejo y strides negativos.
    rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                   dtype=np.uint8, order='C', copy=True)
    if rgb.strides != (rgb.shape[1] * 3, 3, 1):
        raise ValueError(f'Strides RGB inesperados: {rgb.strides}')
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
