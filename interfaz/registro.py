"""Alta de personas sin reemplazar nombres ni archivos previos."""
import os
from pathlib import Path
import pickle
import shutil
import unicodedata
import uuid

from hardware.exclusion import Exclusivo


def leer_base(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        return {"nombres": [], "imagenes": []}
    with ruta.open("rb") as archivo:
        base = pickle.load(archivo)
    if not isinstance(base, dict) or not isinstance(base.get("nombres"), list) or not isinstance(base.get("imagenes"), list):
        raise ValueError("Base de personas invalida; no se modifico.")
    if len(base["nombres"]) != len(base["imagenes"]):
        raise ValueError("Base de personas inconsistente; no se modifico.")
    return base


def inicializar_base(ruta):
    """Primer arranque sin datos personales; nunca sustituye una base existente."""
    ruta=Path(ruta)
    with Exclusivo('base_personas'):
        if ruta.exists():
            leer_base(ruta)
            return False
        ruta.parent.mkdir(parents=True,exist_ok=True)
        temporal=ruta.with_name(ruta.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            vacia={'nombres':[],'imagenes':[]}
            with temporal.open('xb') as f:
                pickle.dump(vacia,f);f.flush();os.fsync(f.fileno())
            if leer_base(temporal)!=vacia:
                raise ValueError('No se pudo verificar la base vacía')
            os.replace(temporal,ruta)
        finally:
            temporal.unlink(missing_ok=True)
        return True


def clave(nombre):
    return unicodedata.normalize("NFKC", nombre).strip().casefold()


def validar_nombre(nombre, base):
    nombre = unicodedata.normalize("NFKC", nombre).strip()
    if not nombre or len(nombre) > 60 or any(unicodedata.category(c).startswith("C") for c in nombre):
        raise ValueError("Escribe un nombre de 1 a 60 caracteres, sin caracteres de control.")
    if any(clave(n) == clave(nombre) for n in base["nombres"]):
        raise ValueError("Esta persona ya esta registrada. El registro existente se conserva.")
    return nombre


def guardar_persona(nombre, muestras_jpeg, db_path, fotos_dir, cantidad=5, cancelar=None):
    if len(muestras_jpeg) != cantidad:
        raise ValueError(f"Se necesitan {cantidad} fotos; no se guardo el registro.")
    db_path, fotos_dir = Path(db_path), Path(fotos_dir)
    with Exclusivo("base_personas"):
        contenido_previo = db_path.read_bytes() if db_path.exists() else None
        base = leer_base(db_path)
        nombre = validar_nombre(nombre, base)
        from interfaz.duplicados import VerificadorDuplicados, exigir_alta
        # Releer y comparar bajo el bloqueo, antes de crear cualquier archivo.
        if cancelar is not None and cancelar():
            from interfaz.duplicados import ComprobacionFacial
            raise ComprobacionFacial("Registro cancelado: no se guardaron fotos ni cambios en la base")
        exigir_alta(VerificadorDuplicados(base, db_path, fotos_dir).comprobar(muestras_jpeg))
        identificador = uuid.uuid4().hex
        destino = fotos_dir / ("registro_" + identificador)
        temporal = db_path.with_name(db_path.name + "." + identificador + ".tmp")
        copia = db_path.with_name(db_path.name + ".antes_registro_" + identificador)
        destino_creado = temporal_creado = copia_creada = False
        try:
            destino.mkdir(parents=True, exist_ok=False)
            destino_creado = True
            rutas = []
            for i, jpeg in enumerate(muestras_jpeg, 1):
                ruta = destino / f"foto_{i:03d}.jpg"
                with ruta.open("xb") as archivo:
                    archivo.write(jpeg)
                rutas.append(str(ruta.resolve()))
            # Copia de seguridad exclusiva antes de actualizar la base.
            if db_path.exists():
                copia = db_path.with_name(db_path.name + ".antes_registro_" + identificador)
                with db_path.open("rb") as fuente, copia.open("xb") as salida:
                    copia_creada = True
                    shutil.copyfileobj(fuente, salida)
            nueva = dict(base)
            nueva["nombres"] = base["nombres"] + [nombre] * cantidad
            nueva["imagenes"] = base["imagenes"] + rutas
            db_path.parent.mkdir(parents=True, exist_ok=True)
            temporal = db_path.with_name(db_path.name + "." + identificador + ".tmp")
            with temporal.open("xb") as archivo:
                temporal_creado = True
                pickle.dump(nueva, archivo)
                archivo.flush()
                os.fsync(archivo.fileno())
            if leer_base(temporal) != nueva:
                raise ValueError("No se pudo verificar la nueva base.")
            contenido_actual = db_path.read_bytes() if db_path.exists() else None
            if contenido_actual != contenido_previo:
                raise ValueError("La base cambio desde otra aplicacion. No se reemplazo; vuelve a consultar las personas.")
            if cancelar is not None and cancelar():
                from interfaz.duplicados import ComprobacionFacial
                raise ComprobacionFacial("Registro cancelado: no se guardaron fotos ni cambios en la base")
            os.replace(temporal, db_path)
        except BaseException:
            if temporal_creado:
                temporal.unlink(missing_ok=True)
            if destino_creado:
                shutil.rmtree(destino)
            if copia_creada:
                copia.unlink(missing_ok=True)
            raise
        return nombre
