"""Galerias derivadas/versionadas. Solo preparar() escribe; nunca en la base activa."""
import configparser
import hashlib
from pathlib import Path

import cv2
import numpy as np

from evaluar_sface import ROOT, MODELOS_SHA, SFace, csv_leer, csv_guardar, verificar_modelos
from configuracion.config import DB_PATH, RAIZ_PROYECTO
from interfaz.registro import leer_base

GALERIAS = Path(RAIZ_PROYECTO)/'datos'/'galerias_sface'
PROTOCOLO = 'sface-v1-yunet2023-letterbox320-alignCrop-maxcos-0.363'


class GaleriaInvalida(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archivo_seguro(manifest, fila):
    archivo = (manifest.parent/fila['archivo']).resolve()
    if archivo.parent != manifest.parent.resolve() or archivo.suffix != '.png':
        raise GaleriaInvalida('Referencia fuera de su sesion PNG')
    contenido = archivo.read_bytes()
    if digest(contenido) != fila['sha256']:
        raise GaleriaInvalida(f'Hash de captura incorrecto: {archivo}')
    return archivo, contenido


def inventario(db_path=DB_PATH, evaluacion=ROOT):
    """Verifica todas las referencias activas y disyuncion con consultas reales."""
    db_path, evaluacion = Path(db_path), Path(evaluacion)
    base_bytes = db_path.read_bytes()
    base = leer_base(db_path)
    if db_path.read_bytes() != base_bytes:
        raise GaleriaInvalida('La base cambio durante la verificacion')
    activos = set(base['nombres'])
    hashes_prueba = set()
    for m in sorted((evaluacion/'sesiones').glob('*/capturas.csv')):
        for fila in csv_leer(m):
            if fila['uso'] != 'evaluacion' or fila['sesion'] != m.parent.name:
                raise GaleriaInvalida('Procedencia de prueba inconsistente')
            _, contenido = archivo_seguro(m, fila)
            hashes_prueba.add(digest(contenido))
    filas, vistos = [], set()
    for m in sorted((evaluacion/'referencias').glob('*/capturas.csv')):
        for f in csv_leer(m):
            alta = (f.get('origen') == 'alta-sface-v1' and m.parent.name.startswith('alta_')
                    and len(m.parent.name)==37 and all(c in '0123456789abcdef' for c in m.parent.name[5:]))
            if (f['uso'] != 'referencias' or f['tipo'] != 'registrada'
                    or f['sesion'] != m.parent.name or not (m.parent.name.startswith(f['persona']+'_') or alta)):
                raise GaleriaInvalida('Manifiesto de referencias con procedencia invalida')
            if f['persona'] not in activos:
                continue  # Retiradas: no se incorporan; originales conservados.
            archivo, contenido = archivo_seguro(m, f)
            h = digest(contenido)
            if h in hashes_prueba:
                raise GaleriaInvalida('Una referencia tambien es captura de prueba')
            if h in vistos:
                raise GaleriaInvalida('Referencia repetida entre sesiones')
            vistos.add(h)
            filas.append(dict(persona=f['persona'], sha256=h,
                              archivo=archivo.relative_to(evaluacion.resolve()).as_posix(),
                              sesion=f['sesion'], manifiesto_sha256=digest(m.read_bytes())))
    if set(f['persona'] for f in filas) != activos:
        raise GaleriaInvalida('Faltan referencias nuevas validas para alguna identidad activa')
    base_sha = digest(base_bytes)
    version = digest(repr((PROTOCOLO, base_sha, sorted(MODELOS_SHA.items()), filas)).encode('utf-8'))
    return filas, base_sha, version


def preparar(db_path=DB_PATH, evaluacion=ROOT, destino=GALERIAS):
    verificar_modelos()
    filas, base_sha, version = inventario(db_path, evaluacion)
    carpeta = Path(destino)/('v1_'+version)
    if carpeta.exists():
        cargar(db_path, evaluacion, destino)
        return carpeta
    motor = SFace() if filas else None
    embeddings, nombres = [], []
    for fila in filas:
        contenido = (Path(evaluacion)/fila['archivo']).read_bytes()
        if digest(contenido) != fila['sha256']:
            raise GaleriaInvalida('Referencia modificada durante preparacion')
        frame = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_COLOR)
        feature, error = motor.extraer(frame)
        if feature is None:
            raise GaleriaInvalida(f"Referencia SFace invalida: {fila['archivo']}: {error}")
        embeddings.append(feature.ravel())
        nombres.append(fila['persona'])
    # No publicar una version que ya quedo desactualizada durante la inferencia.
    if inventario(db_path, evaluacion)[2] != version:
        raise GaleriaInvalida('La base o las referencias cambiaron durante preparacion')
    carpeta.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(carpeta/'embeddings.npz', features=np.asarray(embeddings, np.float32).reshape(-1,128),
                        nombres=np.asarray(nombres, dtype=str))
    csv_guardar(carpeta/'referencias.csv', filas)
    if not filas:
        (carpeta/'referencias.csv').write_text('persona,sha256,archivo,sesion,manifiesto_sha256\n',encoding='utf-8')
    meta = configparser.ConfigParser()
    meta['galeria'] = dict(protocolo=PROTOCOLO, version=version, base_sha256=base_sha,
                          embeddings_sha256=digest((carpeta/'embeddings.npz').read_bytes()),
                          referencias_sha256=digest((carpeta/'referencias.csv').read_bytes()))
    with (carpeta/'metadatos.ini').open('x', encoding='utf-8') as f:
        meta.write(f)
    return carpeta


def cargar(db_path=DB_PATH, evaluacion=ROOT, destino=GALERIAS):
    try:
        verificar_modelos()
        filas, base_sha, version = inventario(db_path, evaluacion)
        if not filas:
            return {}, base_sha, version  # Base vacía válida; ninguna identidad que comparar.
        carpeta = Path(destino)/('v1_'+version)
        meta = configparser.ConfigParser()
        if not meta.read(carpeta/'metadatos.ini', encoding='utf-8'):
            raise GaleriaInvalida('Galeria SFace ausente o invalidada. Ejecuta preparar_galeria_sface.py con el sistema detenido.')
        m = meta['galeria']
        if (m['protocolo'] != PROTOCOLO or m['version'] != version or m['base_sha256'] != base_sha
                or digest((carpeta/'embeddings.npz').read_bytes()) != m['embeddings_sha256']
                or digest((carpeta/'referencias.csv').read_bytes()) != m['referencias_sha256']
                or csv_leer(carpeta/'referencias.csv') != filas):
            raise GaleriaInvalida('Integridad de galeria SFace incorrecta')
        with np.load(carpeta/'embeddings.npz', allow_pickle=False) as datos:
            features, nombres = datos['features'].copy(), datos['nombres'].tolist()
        if (features.dtype != np.float32 or features.shape != (len(filas), 128) or not np.isfinite(features).all()
                or np.any(np.linalg.norm(features, axis=1) < 1e-8)
                or nombres != [f['persona'] for f in filas]):
            raise GaleriaInvalida('Embeddings o etiquetas SFace inconsistentes')
        grupos = {}
        for nombre, feature in zip(nombres, features):
            grupos.setdefault(nombre, []).append(feature.reshape(1,128).copy())
        return grupos, base_sha, version
    except (OSError, ValueError, KeyError, configparser.Error) as e:
        raise GaleriaInvalida(str(e)) from e


def asegurar(db_path=DB_PATH, evaluacion=ROOT, destino=GALERIAS):
    """Reparar solo una versión ausente, nunca ocultar corrupción o referencias inválidas."""
    try:
        return cargar(db_path,evaluacion,destino)
    except GaleriaInvalida as error:
        if not str(error).startswith('Galeria SFace ausente o invalidada.'):
            raise
    from hardware.exclusion import Exclusivo
    with Exclusivo('base_personas'):
        preparar(db_path,evaluacion,destino)
        return cargar(db_path,evaluacion,destino)
