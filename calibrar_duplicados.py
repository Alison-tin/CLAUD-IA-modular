"""Recogida guiada y medicion LBPH aisladas. Nunca crea altas ni politicas.

Solo el subcomando capturar abre camara, cuando el operador lo ejecuta.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import uuid

import cv2

from configuracion.config import DB_PATH, FOTOS_DIR
from hardware.exclusion import Exclusivo
from interfaz.duplicados import cargar_referencias, crear_modelos, medir_muestra
from interfaz.registro import leer_base

CANTIDAD = 30


def escribir_json(ruta, contenido):
    with Path(ruta).open('x', encoding='utf-8') as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2, allow_nan=False)


def referencias(db_path=DB_PATH, fotos_dir=FOTOS_DIR):
    return cargar_referencias(leer_base(db_path), fotos_dir)


def destino_aislado(salida, db_path=DB_PATH, fotos_dir=FOTOS_DIR):
    salida, db, fotos = Path(salida).resolve(), Path(db_path).resolve(), Path(fotos_dir).resolve()
    if (salida == db or salida in db.parents or salida == fotos
            or fotos in salida.parents or salida in fotos.parents):
        raise ValueError('Elige una carpeta de estudio separada de la base activa y de sus fotos.')
    return salida


def preparar(salida, db_path=DB_PATH, fotos_dir=FOTOS_DIR, recuperar=False):
    salida = destino_aislado(salida, db_path, fotos_dir)
    with Exclusivo('escritorio'):
        # Comprobar TODAS las referencias antes de crear o recuperar el estudio.
        grupos, huella = referencias(db_path, fotos_dir)
        if not grupos:
            raise ValueError('Se necesitan personas ya registradas para estudiar duplicados.')
        salida.mkdir(parents=True, exist_ok=True)
        manifiesto = salida/'estudio.json'
        anterior = None
        if manifiesto.exists():
            try:
                anterior = json.loads(manifiesto.read_text(encoding='utf-8'))
            except (ValueError, UnicodeError):
                pass
            valido = (isinstance(anterior, dict) and anterior.get('version') == 1
                      and isinstance(anterior.get('base_sha256'), str)
                      and isinstance(anterior.get('personas_registradas'), list)
                      and anterior.get('capturas_por_sesion') == CANTIDAD)
            if valido:
                if anterior['base_sha256'] != huella or anterior['personas_registradas'] != sorted(grupos):
                    raise ValueError('El estudio existente corresponde a otra base. Se conserva; usa otra --salida.')
                return salida  # Repetir no reescribe manifiesto, sesiones ni informes.
            if not recuperar:
                raise ValueError('Hay un manifiesto incompleto. Usa --recuperar o una nueva --salida; no se sobrescribio.')
        otros = [p for p in salida.iterdir() if p != manifiesto]
        if otros:
            raise ValueError('El estudio parcial contiene otros datos. Se conservan; prepara otra --salida.')
        contenido = {
            'version': 1, 'base_sha256': huella, 'personas_registradas': sorted(grupos),
            'capturas_por_sesion': CANTIDAD,
            'creado_utc': datetime.now(timezone.utc).isoformat(),
            'nota': 'Datos de evaluacion independientes. No son registros ni una calibracion aprobada.'}
        # Publicar un manifiesto completo; un fallo nunca deja JSON truncado.
        with tempfile.TemporaryDirectory(prefix='.preparar_', dir=salida) as temporal:
            nuevo = Path(temporal)/'estudio.json'
            escribir_json(nuevo, contenido)
            if json.loads(nuevo.read_text(encoding='utf-8')) != contenido:
                raise ValueError('No se pudo verificar el manifiesto nuevo.')
            if manifiesto.exists():
                copia = salida/('estudio.json.incompleto_'+uuid.uuid4().hex)
                previo = manifiesto.read_bytes()
                with copia.open('xb') as archivo:
                    archivo.write(previo)
                if copia.read_bytes() != previo:
                    raise ValueError('No se pudo verificar el respaldo del manifiesto incompleto.')
                # Recuperacion explicita, solo de un manifiesto invalido sin sesiones.
                manifiesto.unlink()
            nuevo.rename(manifiesto)
    return salida


def abrir_estudio(salida, db_path=DB_PATH, fotos_dir=FOTOS_DIR):
    salida = destino_aislado(salida, db_path, fotos_dir)
    estudio = json.loads((salida/'estudio.json').read_text(encoding='utf-8'))
    grupos, huella = referencias(db_path, fotos_dir)
    if (estudio.get('version') != 1 or estudio.get('base_sha256') != huella
            or estudio.get('personas_registradas') != sorted(grupos)):
        raise ValueError('La base o sus fotos cambiaron. Prepara un nuevo estudio; no mezcles los datos.')
    return salida, estudio, grupos


def sesiones(salida):
    return [(p.parent, json.loads(p.read_text(encoding='utf-8')))
            for p in sorted(Path(salida).glob('sesion_*/sesion.json'))]


def capturar(salida, grupo, persona, sesion, conjunto, condiciones, cam=0,
             db_path=DB_PATH, fotos_dir=FOTOS_DIR, pedir=input, mostrar=print):
    from datos.personas import detectar_rostros, extraer_rostro
    from hardware.camara import abrir_camara
    if grupo not in ('registrada', 'diferente') or conjunto not in ('calibracion', 'prueba'):
        raise ValueError('Grupo o conjunto no valido.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', sesion):
        raise ValueError('La sesion debe tener de 1 a 60 letras ASCII, numeros, guion o guion bajo.')
    if not persona.strip() or not condiciones.strip():
        raise ValueError('Indica persona y condiciones de la sesion.')
    # El mismo bloqueo de actividad impide capturas simultaneas con web/terminal.
    with Exclusivo('escritorio'):
        salida, estudio, _ = abrir_estudio(salida, db_path, fotos_dir)
        if (persona in estudio['personas_registradas']) != (grupo == 'registrada'):
            raise ValueError('Una registrada debe usar su nombre exacto; una diferente debe usar un codigo externo.')
        anteriores = sesiones(salida)
        if any(s['persona'] == persona and s['grupo'] != grupo for _, s in anteriores):
            raise ValueError('La persona ya tiene otro grupo en este estudio.')
        if grupo == 'diferente' and any(s['persona'] == persona and s['conjunto'] != conjunto for _, s in anteriores):
            raise ValueError('Reserva personas externas distintas para calibracion y prueba.')
        destino = salida/('sesion_'+sesion)
        if destino.exists():
            raise ValueError('Esta sesion ya existe; elige otro identificador, sin reemplazarla.')
        hashes = {f['sha256'] for _, s in anteriores for f in s['capturas']}
        mostrar('No se guardara ningun registro. Se necesitan 30 capturas validas. Q cancela toda esta sesion.')
        cap = abrir_camara(cam)
        try:
            if not cap.isOpened():
                raise RuntimeError('No se pudo abrir la camara.')
            with tempfile.TemporaryDirectory(prefix='.captura_', dir=salida) as temporal:
                etapa = Path(temporal)/'sesion'
                etapa.mkdir()
                capturas = []
                while len(capturas) < CANTIDAD:
                    try:
                        respuesta = pedir(f'{len(capturas)}/30. ENTER para capturar; Q para cancelar: ')
                    except (EOFError, KeyboardInterrupt):
                        return False
                    if respuesta.strip().casefold() == 'q':
                        return False
                    # Leer tras la orden; no guardar el fotograma previo al prompt.
                    for _ in range(3):
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError('La camara dejo de entregar imagenes.')
                    rostros = detectar_rostros(frame)
                    if len(rostros) != 1:
                        mostrar(f'Captura descartada: hay {len(rostros)} rostros; debe haber exactamente uno.')
                        continue
                    imagen = extraer_rostro(frame, rostros[0])
                    if imagen is None or imagen.shape != (200, 200):
                        mostrar('Recorte no valido. Repite la captura.')
                        continue
                    ok, jpeg = cv2.imencode('.jpg', imagen)
                    if not ok:
                        raise RuntimeError('No se pudo codificar la captura.')
                    contenido = jpeg.tobytes()
                    huella = hashlib.sha256(contenido).hexdigest()
                    if huella in hashes:
                        mostrar('Imagen exactamente repetida. Varia ligeramente la postura y repite.')
                        continue
                    hashes.add(huella)
                    archivo = f'foto_{len(capturas)+1:03d}.jpg'
                    (etapa/archivo).write_bytes(contenido)
                    capturas.append({'archivo': archivo, 'sha256': huella})
                abrir_estudio(salida, db_path, fotos_dir)  # Detectar cambios durante la recogida.
                escribir_json(etapa/'sesion.json', {
                    'grupo': grupo, 'persona': persona, 'sesion': sesion, 'conjunto': conjunto,
                    'condiciones': condiciones, 'base_sha256': estudio['base_sha256'], 'capturas': capturas})
                etapa.rename(destino)
                mostrar(f'Sesion completa en {destino}. Base activa sin modificar.')
                return True
        finally:
            cap.release()


def medir(salida, conjunto, informe, db_path=DB_PATH, fotos_dir=FOTOS_DIR):
    salida, estudio, grupos = abrir_estudio(salida, db_path, fotos_dir)
    informe = Path(informe).resolve()
    # Los informes tambien quedan dentro del estudio, lejos de los datos activos.
    if informe.parent != salida:
        raise ValueError('Guarda el informe directamente dentro de la carpeta del estudio.')
    modelos = crear_modelos(grupos)
    resultados = []
    for carpeta, sesion in sesiones(salida):
        if sesion['conjunto'] != conjunto:
            continue
        if sesion['base_sha256'] != estudio['base_sha256'] or len(sesion['capturas']) != CANTIDAD:
            raise ValueError('Sesion incompleta o perteneciente a otra base.')
        lecturas = []
        for captura in sesion['capturas']:
            nombre = captura['archivo']
            if not re.fullmatch(r'foto_[0-9]{3}\.jpg', nombre):
                raise ValueError('Ruta de captura no valida.')
            contenido = (carpeta/nombre).read_bytes()
            if hashlib.sha256(contenido).hexdigest() != captura['sha256']:
                raise ValueError('Una captura cambio; no se emitira el informe.')
            lecturas.append(medir_muestra(contenido, modelos))
        resultados.append({**sesion, 'distancias': lecturas})
    if not resultados:
        raise ValueError('No hay sesiones completas para ese conjunto.')
    abrir_estudio(salida, db_path, fotos_dir)
    escribir_json(informe, {'base_sha256': estudio['base_sha256'], 'conjunto': conjunto,
                           'sesiones': resultados,
                           'nota': 'Distancias LBPH sin limites ni probabilidades. No habilita altas.'})
    return informe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='accion', required=True)
    p = subs.add_parser('preparar', help='Crear estudio aislado sin abrir camara')
    p.add_argument('--salida', required=True)
    p.add_argument('--recuperar', action='store_true', help='Respaldar y recuperar un manifiesto incompleto sin sesiones')
    p = subs.add_parser('capturar', help='Recoger una sesion sin crear altas; abre camara')
    p.add_argument('--salida', required=True)
    p.add_argument('--grupo', choices=['registrada', 'diferente'], required=True)
    p.add_argument('--persona', required=True)
    p.add_argument('--sesion', required=True)
    p.add_argument('--conjunto', choices=['calibracion', 'prueba'], required=True)
    p.add_argument('--condiciones', required=True)
    p.add_argument('--cam', type=int, default=0)
    p = subs.add_parser('medir', help='Medir sesiones offline sin abrir camara ni elegir limites')
    p.add_argument('--salida', required=True)
    p.add_argument('--conjunto', choices=['calibracion', 'prueba'], required=True)
    p.add_argument('--informe', required=True)
    args = vars(parser.parse_args(argv))
    accion = args.pop('accion')
    try:
        resultado = {'preparar': preparar, 'capturar': capturar, 'medir': medir}[accion](**args)
        print(resultado if accion != 'capturar' else ('Sesion guardada.' if resultado else 'Sesion cancelada: sin capturas parciales.'))
        return 0 if resultado is not False else 1
    except (OSError, ValueError, RuntimeError, AttributeError, cv2.error) as error:
        print(f'No se pudo completar: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
