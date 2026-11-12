"""Genera y valida propuestas pendientes de validacion fisica. Nunca abre hardware."""
import json
import math
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from configuracion.config import PCA, PCA0, SERVO, SERVO_CUELLO

CARPETA = Path(__file__).resolve().parent


def grupo(pca, canales, posiciones, tiempo):
    return [[pca, c, tiempo, 0, p, 3] for c, p in zip(canales, posiciones)]


def facial(canales, posiciones, tiempo):
    return grupo(PCA, canales, posiciones, tiempo)


def neutros(canales, tiempo):
    return facial(canales, [SERVO[c][2] for c in canales], tiempo)


def boca(tiempos, posiciones):
    return [
        [PCA0, 12, t, 0, p, 3] for t, p in zip(tiempos, posiciones)]


def propuestas():
    cara = [3, 5, 8, 9, 10, 11]
    alegria = neutros(cara, 0) + facial(cara, [385, 605, 535, 570, 603, 603], .5) + neutros(cara, 1.5)
    # Coordinacion de ojos calculada con las formulas existentes, truncando
    # a entero como Arduino.enviar(); no se invocan funciones de movimiento.
    def ojos_v(pos, t):
        pareja = int(SERVO[2][1] - (pos-SERVO[0][0])/(SERVO[0][1]-SERVO[0][0])*(SERVO[2][1]-SERVO[2][0]))
        return facial([0, 2], [pos, pareja], t)
    def ojos_h(pos, t):
        pareja = int(SERVO[6][0] + (pos-SERVO[1][0])/(SERVO[1][1]-SERVO[1][0])*(SERVO[6][1]-SERVO[6][0]))
        return facial([1, 6], [pos, pareja], t)
    parpados = [12, 13, 14, 15]
    abiertos, cerrados = [600, 520, 580, 480], [420, 655, 435, 620]
    doble = []
    for t, posiciones in [(0, abiertos), (.35, cerrados), (.7, abiertos), (1.05, cerrados), (1.4, abiertos)]:
        doble += facial(parpados, posiciones, t)
    risa = neutros([3, 5, 8], 0) + facial([3, 5, 8], [385, 605, 535], .5)
    risa += boca([0, .5, .85, 1.2, 1.55, 1.9, 2.25], [300, 475, 300, 475, 300, 475, 300])
    risa += neutros([3, 5, 8], 2.6)
    return {
        "alegria.json": alegria,
        "enojo.json": ojos_h(464, 0)+ojos_h(441, .5)+ojos_h(487, 1)+ojos_h(464, 1.5),
        "sorpresa.json": facial(parpados, abiertos, 0)+boca([0, .5, 1.5], [300, 475, 300]),
        "tristeza.json": ojos_v(465, 0)+ojos_v(515, .5)+boca([0, .5, 1.1], [300, 475, 300])+ojos_v(465, 1.6),
        "habla_media.json": boca([0, .5, .85, 1.2, 1.55], [300, 475, 300, 475, 300]),
        "guino_derecho.json": facial([12, 13], abiertos[:2], 0)+facial([12, 13], cerrados[:2], .5)+facial([12, 13], abiertos[:2], 1),
        "parpadeo_doble.json": doble,
        "barrido_horizontal.json": ojos_h(464, 0)+ojos_h(441, .6)+ojos_h(487, 1.4)+ojos_h(464, 2.2),
        "risa.json": risa,
    }


def validar(bloques):
    if not isinstance(bloques, list) or not bloques:
        raise ValueError("La secuencia debe ser una lista no vacia")
    vistos = set()
    for bloque in bloques:
        if not isinstance(bloque, list) or len(bloque) != 6:
            raise ValueError("Bloque con formato incorrecto")
        pca, canal, inicio, duracion, posicion, velocidad = bloque
        if any(type(v) not in (int, float) or (type(v) is float and not math.isfinite(v)) for v in bloque):
            raise ValueError(f"Valores numericos finitos requeridos: {bloque}")
        if type(pca) is not int or type(canal) is not int:
            raise ValueError(f"PCA y canal deben ser enteros: {bloque}")
        tabla = SERVO if pca == PCA else SERVO_CUELLO if pca == PCA0 else {}
        if canal not in tabla or type(posicion) is not int or not tabla[canal][0] <= posicion <= tabla[canal][1]:
            raise ValueError(f"Canal/posicion fuera de rango: {bloque}")
        if not 0 <= inicio <= 60 or duracion != 0 or velocidad != 3:
            raise ValueError(f"Tiempo/velocidad inesperados: {bloque}")
        identificador = (pca, canal, inicio)
        if identificador in vistos:
            raise ValueError(f"Dos ordenes simultaneas al mismo servo: {bloque}")
        vistos.add(identificador)
    return {"bloques": len(bloques), "ultimo_envio_s": max(b[2] for b in bloques), "rangos": "VALIDOS"}


def main():
    informe = {}
    for nombre, bloques in propuestas().items():
        bloques.sort(key=lambda b: b[2])
        informe[nombre] = validar(bloques)
        destino = CARPETA/nombre
        texto = json.dumps(bloques, indent=2)+"\n"
        if destino.exists():
            if destino.read_text(encoding="utf-8") != texto:
                raise ValueError(f"No se sobrescribe una propuesta diferente: {nombre}")
        else:
            with destino.open("x", encoding="utf-8") as archivo:
                archivo.write(texto)
        validar(json.loads(destino.read_text(encoding="utf-8")))
    print(json.dumps(informe, indent=2))
    print("Propuestas nuevas; no originales. Conectadas a gestos; pendientes de validacion fisica.")


if __name__ == "__main__":
    main()
