"""Decision por lectura, sin memoria temporal ni acceso a dispositivos."""
import math


def sin_lectura():
    return {'nombre': None, 'distancia': None, 'umbral': None,
            'similitud_estimada': None}


def decidir_reconocimiento(etiqueta, distancia, nombres, umbral):
    distancia = float(distancia)
    candidato = nombres.get(int(etiqueta))
    aceptado = (candidato is not None and math.isfinite(distancia)
                and 0 <= distancia < umbral)
    return {'nombre': candidato if aceptado else 'Desconocido',
            'distancia': distancia if math.isfinite(distancia) else None,
            'umbral': umbral,
            'similitud_estimada': max(0.0, min(1.0, 1.0-distancia/100.0))
            if aceptado else None}


def distancias_por_identidad(modelo, imagen, nombres):
    """Diagnostico del MISMO modelo y array, sin otro preprocesamiento."""
    import cv2
    colector = cv2.face.StandardCollector_create()
    modelo.predict_collect(imagen, colector)
    distancias = {}
    for etiqueta, distancia in colector.getResults():
        nombre = nombres[int(etiqueta)]
        distancias[nombre] = min(distancias.get(nombre, float('inf')), float(distancia))
    return distancias
