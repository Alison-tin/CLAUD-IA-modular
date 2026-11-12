"""Retiro y recuperacion de registros, con copias verificadas y sin borrar fotos."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import uuid

from hardware.exclusion import Exclusivo
from interfaz.registro import leer_base, validar_nombre
from interfaz.duplicados import resolver_referencia


def huella(datos):
    return hashlib.sha256(datos).hexdigest()


def carpeta_respaldos(db_path):
    return Path(db_path).resolve().parent.parent / "respaldos" / "registros_retirados"


def escribir_nuevo(ruta, datos):
    with ruta.open("xb") as archivo:
        archivo.write(datos)
        archivo.flush()
        os.fsync(archivo.fileno())
    if ruta.read_bytes() != datos:
        raise ValueError("No se pudo verificar el respaldo. La base no se modificó.")


def actualizar(db_path, previa, nueva, antes_publicar=None):
    temporal = db_path.with_name(db_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        escribir_nuevo(temporal, pickle.dumps(nueva))
        if leer_base(temporal) != nueva:
            raise ValueError("No se pudo verificar la base preparada.")
        if antes_publicar is not None:
            antes_publicar(temporal)
        if db_path.read_bytes() != previa:
            raise ValueError("La base cambió desde otra aplicación. No se reemplazó.")
        os.replace(temporal, db_path)
    finally:
        temporal.unlink(missing_ok=True)


def verificar_respaldo(respaldo, manifiesto):
    """Releer lo publicado y verificar SHA-256 antes de permitir el retiro."""
    ruta = respaldo / 'registro.json'
    try:
        if json.loads(ruta.read_text(encoding='utf-8')) != manifiesto:
            raise ValueError('El manifiesto no coincide con el respaldo previsto.')
        ruta = respaldo / 'base_antes.pkl'
        if huella(ruta.read_bytes()) != manifiesto['base_sha256']:
            raise ValueError('SHA-256 de la base respaldada no coincide.')
        for foto in manifiesto['fotos']:
            ruta = respaldo / foto['archivo']
            if huella(ruta.read_bytes()) != foto['sha256']:
                raise ValueError('SHA-256 de la foto respaldada no coincide.')
            ruta = Path(foto['origen'])
            if huella(ruta.read_bytes()) != foto['sha256']:
                raise ValueError('La foto de origen cambió durante el respaldo.')
    except (OSError, ValueError) as error:
        raise ValueError(f'No se pudo verificar {ruta}: {error}. El registro se conserva.') from error


def retirar_persona(nombre, db_path, fotos_dir, antes_publicar=None):
    db_path, fotos_dir = Path(db_path), Path(fotos_dir)
    # El trabajador web conserva el bloqueo historico "escritorio";
    # camara. No retirar mientras alguno pueda seguir usando un modelo en memoria.
    with Exclusivo("escritorio"), Exclusivo("camara"), Exclusivo("base_personas"):
        previa = db_path.read_bytes()
        base = leer_base(db_path)
        indices = [i for i, n in enumerate(base["nombres"]) if n == nombre]
        if not indices:
            raise ValueError("La persona ya no está registrada. Actualiza la lista.")
        muestras = []
        for i in indices:
            referencia = base["imagenes"][i]
            try:
                fuente = resolver_referencia(referencia, fotos_dir)
                contenido = fuente.read_bytes()
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"No se pudieron respaldar todas las fotos. El registro se conserva. "
                    f"Muestra {i+1}, referencia {referencia}: {error}"
                ) from error
            muestras.append((str(fuente), contenido, str(referencia), i))
        respaldo = carpeta_respaldos(db_path) / ("retiro_" + uuid.uuid4().hex)
        respaldo.mkdir(parents=True, exist_ok=False)
        (respaldo / "fotos").mkdir(exist_ok=False)
        escribir_nuevo(respaldo / "base_antes.pkl", previa)
        fotos = []
        for numero, (fuente, contenido, referencia, indice) in enumerate(muestras, 1):
            # Nombres exclusivos aunque las fuentes compartan nombre de archivo.
            relativa = f"fotos/foto_{numero:03d}{Path(fuente).suffix or '.jpg'}"
            escribir_nuevo(respaldo / relativa, contenido)
            fotos.append({"archivo": relativa, "sha256": huella(contenido), "origen": fuente,
                          "referencia_original": referencia, "indice_base": indice})
        manifiesto = {"version": 1, "nombre": nombre, "cantidad": len(fotos),
                      "base_sha256": huella(previa), "fotos": fotos}
        escribir_nuevo(respaldo / "registro.json", json.dumps(manifiesto, ensure_ascii=False, indent=2).encode("utf-8"))
        verificar_respaldo(respaldo, manifiesto)
        eliminar = set(indices)
        nueva = dict(base)
        nueva["nombres"] = [n for i, n in enumerate(base["nombres"]) if i not in eliminar]
        nueva["imagenes"] = [p for i, p in enumerate(base["imagenes"]) if i not in eliminar]
        actualizar(db_path, previa, nueva, antes_publicar=antes_publicar)
        return respaldo.name


def recuperar_persona(identificador, db_path, fotos_dir):
    if not isinstance(identificador, str) or not identificador.startswith("retiro_") or any(c not in "retiro_0123456789abcdef" for c in identificador):
        raise ValueError("Identificador de respaldo inválido.")
    db_path, fotos_dir = Path(db_path), Path(fotos_dir)
    respaldo = carpeta_respaldos(db_path) / identificador
    with Exclusivo("escritorio"), Exclusivo("camara"), Exclusivo("base_personas"):
        previa = db_path.read_bytes()
        base = leer_base(db_path)
        manifiesto = json.loads((respaldo / "registro.json").read_text(encoding="utf-8"))
        nombre = validar_nombre(manifiesto["nombre"], base)
        if manifiesto.get("version") != 1 or not manifiesto["fotos"] or manifiesto["cantidad"] != len(manifiesto["fotos"]):
            raise ValueError("Respaldo incompleto; no se modificó la base.")
        contenidos = []
        for foto in manifiesto["fotos"]:
            fuente = (respaldo / foto["archivo"]).resolve()
            if fuente.parent != (respaldo / "fotos").resolve():
                raise ValueError("Ruta de respaldo inválida.")
            contenido = fuente.read_bytes()
            if huella(contenido) != foto["sha256"]:
                raise ValueError("Una foto del respaldo cambió. No se recuperó el registro.")
            contenidos.append(contenido)
        destino = fotos_dir / ("recuperado_" + uuid.uuid4().hex)
        destino.mkdir(parents=True, exist_ok=False)
        rutas = []
        for numero, contenido in enumerate(contenidos, 1):
            ruta = destino / f"foto_{numero:03d}.jpg"
            escribir_nuevo(ruta, contenido)
            rutas.append(str(ruta.resolve()))
        escribir_nuevo(respaldo / ("base_antes_recuperacion_" + uuid.uuid4().hex + ".pkl"), previa)
        nueva = dict(base)
        nueva["nombres"] = base["nombres"] + [nombre] * len(rutas)
        nueva["imagenes"] = base["imagenes"] + rutas
        actualizar(db_path, previa, nueva)
        return nombre
