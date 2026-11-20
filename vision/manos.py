import math
import cv2



# CONEXIONES DE LA MANO


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indice
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # menique
    (0, 17),                                  # palma base
]



# DIBUJAR MANO


def dibujar_mano(
    frame,
    lms,
    W,
    H,
    color_linea=(255, 255, 255),
    color_punto=(0, 0, 255)
):
    """Dibuja el esqueleto completo de la mano estilo cvzone."""

    pts = [
        (int(lm.x * W), int(lm.y * H))
        for lm in lms
    ]

    # Lineas de los huesos
    for a, b in HAND_CONNECTIONS:
        cv2.line(
            frame,
            pts[a],
            pts[b],
            color_linea,
            2,
            cv2.LINE_AA
        )

    # Puntos de las articulaciones
    for i, (px, py) in enumerate(pts):

        cv2.circle(
            frame,
            (px, py),
            5,
            color_punto,
            cv2.FILLED
        )

        cv2.circle(
            frame,
            (px, py),
            5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Bounding box
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1 = min(xs) - 20
    y1 = min(ys) - 20
    x2 = max(xs) + 20
    y2 = max(ys) + 20

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 255),
        2
    )

    return (x1, y1, x2, y2)



# DEDOS ARRIBA


def dedos_arriba(lm):
    """
    Retorna lista:
    [pulgar, indice, medio, anular, menique]

    1 = arriba
    0 = abajo
    """

    dedos = []

    pd = lm[5].x < lm[17].x

    dedos.append(
        1 if (
            (lm[4].x < lm[3].x)
            if pd
            else (lm[4].x > lm[3].x)
        )
        else 0
    )

    for punta, pip in [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]:

        dedos.append(
            1 if lm[punta].y < lm[pip].y
            else 0
        )

    return dedos



# DEDO EXTENDIDO


def dedo_ext(lm, punta, pip):
    return lm[punta].y < lm[pip].y



# DISTANCIA ENTRE LANDMARKS


def dist_lm(lm, a, b):
    return math.sqrt(
        (lm[a].x - lm[b].x) ** 2
        + (lm[a].y - lm[b].y) ** 2
    )



# CLASIFICAR GESTO


def clasificar_gesto(lm):

    ind = dedo_ext(lm, 8, 6)
    med = dedo_ext(lm, 12, 10)
    anu = dedo_ext(lm, 16, 14)
    men = dedo_ext(lm, 20, 18)

    pd = lm[5].x < lm[17].x

    pul = (
        (lm[4].x < lm[3].x)
        if pd
        else (lm[4].x > lm[3].x)
    )

    n = sum([
        ind,
        med,
        anu,
        men
    ])

    if (
        dist_lm(lm, 4, 8) < 0.07
        and med
        and anu
        and men
    ):
        return "ok"

    if n == 4 and pul:
        return "mano_abierta"

    if n == 0 and not pul:
        return "puno"

    if n == 0 and pul:

        if lm[4].y < lm[2].y:
            return "pulgar_arriba"

        if lm[4].y > lm[2].y + 0.05:
            return "pulgar_abajo"

    if ind and not med and not anu and not men:
        return "indice"

    if ind and med and not anu and not men:
        return "dos_dedos"

    if ind and med and anu and not men:
        return "tres_dedos"

    if n == 4 and not pul:
        return "cuatro_dedos"

    return None



# ACCIONES ASOCIADAS A CADA GESTO


GESTO_ACCION = {
    "mano_abierta": (
        "alegria.json",
        "Alegria"
    ),

    "puno": (
        "enojo.json",
        "Enojo"
    ),

    "pulgar_arriba": (
        "sorpresa.json",
        "Sorpresa"
    ),

    "pulgar_abajo": (
        "tristeza.json",
        "Tristeza"
    ),

    "indice": (
        "habla_media.json",
        "Habla"
    ),

    "dos_dedos": (
        "guino_derecho.json",
        "Guino"
    ),

    "tres_dedos": (
        "parpadeo_doble.json",
        "Parpadeo"
    ),

    "cuatro_dedos": (
        "barrido_horizontal.json",
        "Barrido"
    ),

    "ok": (
        "risa.json",
        "Risa"
    ),
}



# GESTOS QUE ANIMAN LA BOCA


GESTOS_HABLA = {
    "indice",
    "ok",
    "mano_abierta"
}


# Descripciones del programa, no confirmaciones de movimiento mecánico.
GUIA_GESTOS = {
    'mano_abierta': ('Mano abierta','Extiende los cinco dedos','Mueve parte de la cara y vuelve al reposo.','Sonrisa por confirmar: esos canales no tienen descripción anatómica.'),
    'puno': ('Puño','Recoge todos los dedos','Mueve los ojos hacia un lado y hacia el otro.',''),
    'pulgar_arriba': ('Pulgar arriba','Pulgar extendido hacia arriba','Abre los párpados; abre parcialmente y cierra la boca.',''),
    'pulgar_abajo': ('Pulgar abajo','Pulgar extendido hacia abajo','Mueve los ojos en vertical; abre parcialmente y cierra la boca.',''),
    'indice': ('Índice','Extiende solo el índice','Abre parcialmente y cierra la boca dos veces.',''),
    'dos_dedos': ('Dos dedos','Índice y medio extendidos','Cierra y abre un par de párpados para hacer un guiño.','Falta confirmar el lado del guiño en el cableado.'),
    'tres_dedos': ('Tres dedos','Índice, medio y anular extendidos','Parpadea dos veces.',''),
    'cuatro_dedos': ('Cuatro dedos','Recoge el pulgar; extiende los demás','Mueve los ojos hacia un lado y hacia el otro y vuelve al centro.',''),
    'ok': ('OK','Une pulgar e índice; extiende los demás','Abre parcialmente y cierra la boca tres veces y mueve parte de la cara.','La sonrisa de la expresión Risa no está documentada por canal.'),
}


def guia_gestos():
    """La guía comparte el selector del sistema y verifica sus nueve JSON actuales."""
    from propuestas_movimientos.preparar import propuestas
    from secuencias.reproductor import cargar
    esperadas=propuestas()
    # Descripciones ligadas al archivo, para seguir una reasignación de gesto.
    archivos={'mano_abierta':'alegria.json','puno':'enojo.json','pulgar_arriba':'sorpresa.json',
              'pulgar_abajo':'tristeza.json','indice':'habla_media.json','dos_dedos':'guino_derecho.json',
              'tres_dedos':'parpadeo_doble.json','cuatro_dedos':'barrido_horizontal.json','ok':'risa.json'}
    descripciones={archivo:GUIA_GESTOS[gesto][2:] for gesto,archivo in archivos.items()}
    resultado=[]
    for gesto,(archivo,expresion) in GESTO_ACCION.items():
        nombre,instruccion=GUIA_GESTOS.get(gesto,(gesto,'Gesto pendiente de describir'))[:2]
        movimiento,nota=descripciones.get(archivo,('Movimiento pendiente de revisar.','Asociación incompleta.'))
        disponible=coincide=False
        try:
            bloques=cargar(archivo)
            disponible=True
            coincide=archivo in descripciones and bloques==sorted(esperadas.get(archivo,[]),key=lambda b:b[2])
        except (ValueError,OSError):
            pass
        if not disponible:
            movimiento='Movimiento no disponible.'
            nota='Asociación incompleta: falta una secuencia válida.'
        elif not coincide:
            movimiento='Movimiento pendiente de revisar.'
            nota='Asociación incompleta: la secuencia cambió y su descripción debe revisarse.'
        resultado.append(dict(gesto=gesto,nombre=nombre,instruccion=instruccion,
            movimiento=movimiento,nota=nota,imagen=gesto.replace('_','-'),
            archivo=archivo,expresion=expresion,disponible=disponible,coincide_con_codigo=coincide))
    return resultado



# COLORES DE LOS GESTOS


COLORES_G = {
    "mano_abierta": (0, 220, 0),
    "puno": (0, 0, 220),
    "pulgar_arriba": (0, 200, 255),
    "pulgar_abajo": (200, 100, 0),
    "indice": (255, 200, 0),
    "dos_dedos": (0, 180, 255),
    "tres_dedos": (180, 0, 255),
    "cuatro_dedos": (0, 255, 200),
    "ok": (0, 255, 128),
}
