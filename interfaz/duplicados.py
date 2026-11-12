"""Control de altas independiente del reconocimiento y de su escala visual.

Sin una politica validada para la base activa, el resultado es dudoso.
No se deducen limites biometricos de las propias fotos de entrenamiento.
"""
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

CAPTURAS_COMPROBACION = 5
REPETIR = "Resultado dudoso. Repite la captura con buena luz y una sola persona."
FALTA_CONFIGURAR = ("Registro bloqueado: falta configurar la comprobacion de duplicados para la base activa, "
                    "o su configuracion ya no es valida. No se iniciaran capturas. "
                    "Se requieren datos de evaluacion independientes para validar los limites.")


class ComprobacionFacial(ValueError):
    """Mensaje seguro que puede mostrarse en la web."""


def resolver_referencia(ruta, fotos_dir):
    """Reubicar una raiz conservando la ruta COMPLETA de la sesion.

    Nunca buscar por basename: foto_001.jpg se repite entre identidades.
    Las rutas Windows historicas tambien se interpretan en pruebas POSIX.
    """
    original = Path(ruta)
    if original.is_file():
        return original.resolve()
    raiz = Path(fotos_dir).resolve()
    partes = PureWindowsPath(str(ruta)).parts
    marcas = [i for i, p in enumerate(partes)
              if p.casefold() in {raiz.name.casefold(), 'rostros_fotos'}]
    if len(marcas) == 1:
        cola = partes[marcas[0]+1:]
    elif not marcas and not original.is_absolute() and not PureWindowsPath(str(ruta)).drive:
        cola = partes
    else:
        raise FileNotFoundError(f'Referencia no localizable sin adivinar su identidad: {ruta}')
    if not cola or any(p in ('..', '/', '\\') for p in cola):
        raise ValueError(f'Referencia insegura o incompleta: {ruta}')
    candidato = raiz.joinpath(*cola).resolve()
    if not candidato.is_relative_to(raiz):
        raise ValueError(f'Referencia fuera de la carpeta de fotos: {ruta}')
    if not candidato.is_file():
        raise FileNotFoundError(f'No se encontro la foto. Referencia: {ruta}; ruta esperada: {candidato}')
    return candidato


def cargar_referencias(base, fotos_dir):
    """Lectura estricta compartida por altas y calibracion, sin escribir la base."""
    grupos = {}
    errores = []
    huella = hashlib.sha256(b"LBPH-1-8-8-8-gray200-equalize-v1")
    for indice, (nombre, ruta) in enumerate(zip(base["nombres"], base["imagenes"]), 1):
        try:
            p = resolver_referencia(ruta, fotos_dir)
            contenido = p.read_bytes()
            imagen = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_GRAYSCALE)
            if imagen is None:
                raise ValueError('La imagen no se puede decodificar.')
            imagen = cv2.equalizeHist(cv2.resize(imagen, (200, 200)))
        except (OSError, ValueError, cv2.error) as error:
            errores.append(f'Muestra {indice}, registro {nombre}: {error}')
            continue
        huella.update(json.dumps([nombre, hashlib.sha256(contenido).hexdigest()], ensure_ascii=True).encode())
        grupos.setdefault(nombre, []).append(imagen)
    if errores:
        raise ValueError('No se pudieron verificar todas las fotos activas:\n' + '\n'.join(errores))
    return grupos, huella.hexdigest()


def crear_modelos(grupos):
    modelos = {}
    for nombre, imagenes in grupos.items():
        modelo = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        modelo.train(imagenes, np.zeros(len(imagenes), dtype=np.int32))
        modelos[nombre] = modelo
    return modelos


def medir_muestra(jpeg, modelos):
    imagen = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if imagen is None or imagen.shape != (200, 200):
        raise ValueError("Se necesita un recorte JPEG valido de 200x200.")
    imagen = cv2.equalizeHist(imagen)
    distancias = {n: float(m.predict(imagen)[1]) for n, m in modelos.items()}
    if any(not math.isfinite(d) or d < 0 for d in distancias.values()):
        raise ValueError("LBPH no devolvio distancias validas.")
    return distancias


def preparacion_registro(db_path, fotos_dir):
    from interfaz.registro import leer_base
    from interfaz.recorrido import PROTOCOLO
    try:
        verificador = VerificadorDuplicados(leer_base(db_path), db_path, fotos_dir)
        permitido = verificador.vacia or (verificador.limites is not None and verificador.protocolo == PROTOCOLO)
    except (OSError, ValueError):
        permitido = False
    return {"permitido": permitido,
            "mensaje": "Comprobacion disponible. Puedes iniciar el registro." if permitido else FALTA_CONFIGURAR}


def decidir(lecturas, limites):
    """Cada lectura contiene la distancia a TODAS las identidades activas.

    Unanimidad de al menos cinco capturas: misma identidad bajo el limite
    de coincidencia y todas las otras sobre el de exclusion, o todas las
    identidades sobre exclusion. Cualquier mezcla requiere repetir.
    """
    if len(lecturas) < CAPTURAS_COMPROBACION or not limites:
        return "dudoso", None
    nombres = set(limites)
    votos = []
    for lectura in lecturas:
        if set(lectura) != nombres or any(not math.isfinite(d) or d < 0 for d in lectura.values()):
            return "dudoso", None
        cercanos = [n for n in nombres if lectura[n] <= limites[n][0]]
        if len(cercanos) == 1 and all(lectura[n] >= limites[n][1] for n in nombres - set(cercanos)):
            votos.append(cercanos[0])
        elif all(lectura[n] >= limites[n][1] for n in nombres):
            votos.append(None)
        else:
            return "dudoso", None
    if len(set(votos)) != 1:
        return "dudoso", None
    return ("sin_coincidencia", None) if votos[0] is None else ("duplicado", votos[0])


class VerificadorDuplicados:
    def __init__(self, base, db_path, fotos_dir):
        self.modelos = {}
        self.limites = None
        self.protocolo = None
        self.vacia = not base["nombres"]
        if self.vacia:
            return
        # La politica queda vinculada al contenido de TODAS las fotos activas.
        # Una muestra ausente/invalida nunca se omite silenciosamente.
        try:
            grupos, self.huella = cargar_referencias(base, fotos_dir)
            politica = json.loads(Path(str(db_path) + ".duplicados.json").read_text(encoding="utf-8"))
            if (politica.get("version") != 1 or politica.get("base_sha256") != self.huella
                    or not politica.get("validacion")):
                return
            limites = politica["limites"]
            if set(limites) != set(grupos):
                return
            for par in limites.values():
                if (len(par) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in par)
                        or not 0 <= par[0] < par[1]):
                    return
            self.modelos = crear_modelos(grupos)
            self.limites = limites
            self.protocolo = politica.get('protocolo')
        except (OSError, ValueError, KeyError, TypeError, AttributeError, cv2.error):
            self.limites = None

    def comprobar(self, muestras):
        # Una politica de 30 capturas no valida automaticamente el nuevo recorrido.
        from interfaz.recorrido import PROTOCOLO
        if not self.vacia and len(muestras) == 5 and self.protocolo != PROTOCOLO:
            return "dudoso", None
        if len(muestras) < CAPTURAS_COMPROBACION:
            return "dudoso", None
        lecturas = []
        try:
            for jpeg in muestras:
                imagen = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
                if imagen is None or imagen.shape != (200, 200):
                    return "dudoso", None
                if self.vacia:
                    continue
                if self.limites is None:
                    return "dudoso", None
                lecturas.append(medir_muestra(jpeg, self.modelos))
        except (ValueError, TypeError, cv2.error):
            return "dudoso", None
        if self.vacia:
            return "sin_coincidencia", None
        return decidir(lecturas, self.limites)


def exigir_alta(resultado):
    estado, nombre = resultado
    if estado == "duplicado":
        raise ComprobacionFacial(f"Este rostro coincide con un registro existente: {nombre}")
    if estado != "sin_coincidencia":
        raise ComprobacionFacial(REPETIR)
