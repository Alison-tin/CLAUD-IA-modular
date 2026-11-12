"""Auditoria offline; no abre hardware, no guarda fotos ni modifica politicas."""
import argparse
import csv
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import statistics
import cv2
import numpy as np
from datos.personas import cargar_db, entrenar_reconocedor, FOTOS_DIR, DB_PATH
from interfaz.duplicados import (resolver_referencia, cargar_referencias, crear_modelos,
                                medir_muestra, preparacion_registro)
from vision.reconocimiento import distancias_por_identidad


def auditar():
    base = cargar_db()
    if len(base['nombres']) != len(base['imagenes']):
        raise ValueError('Longitudes de nombres e imagenes distintas')
    rutas, hashes, protegidos = defaultdict(set), defaultdict(set), {}
    protegidos[Path(DB_PATH)] = Path(DB_PATH).read_bytes()
    for nombre, referencia in zip(base['nombres'], base['imagenes']):
        ruta = resolver_referencia(referencia, FOTOS_DIR)
        contenido = ruta.read_bytes()
        imagen = cv2.imdecode(np.frombuffer(contenido, np.uint8), 0)
        if imagen is None or imagen.shape != (200, 200):
            raise ValueError(f'Muestra invalida: {nombre}: {ruta}')
        protegidos[ruta] = contenido
        rutas[str(ruta)].add(nombre)
        hashes[hashlib.sha256(contenido).hexdigest()].add(nombre)
    if any(len(n) > 1 for n in (*rutas.values(), *hashes.values())):
        raise ValueError('Una misma foto esta asociada a varias identidades')
    entradas = []
    reconocedor, nombres = entrenar_reconocedor(db=base, entradas_debug=entradas)
    if len(entradas) != len(base['imagenes']):
        raise ValueError('No se entrenaron todas las entradas')
    grupos, huella = cargar_referencias(base, FOTOS_DIR)
    modelos_dup = crear_modelos(grupos)
    grupos_main = defaultdict(list)
    for e in entradas:
        grupos_main[e['nombre']].append(e['imagen'])
    modelos_main = crear_modelos(grupos_main)
    filas = []
    for indice, e in enumerate(entradas, 1):
        im = e['imagen']
        etiqueta, distancia = reconocedor.predict(im)
        d_main = {n: float(m.predict(im)[1]) for n, m in modelos_main.items()}
        d_colector = distancias_por_identidad(reconocedor, im, nombres)
        assert all(abs(d_main[n]-d_colector[n]) < 1e-6 for n in nombres.values())
        d_dup = medir_muestra(protegidos[Path(e['ruta'])], modelos_dup)
        # Misma sesion, retirando solo esta muestra: NO validacion independiente.
        otros = [o for j, o in enumerate(entradas, 1) if j != indice]
        loo = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        loo.train([o['imagen'] for o in otros], np.array([o['etiqueta'] for o in otros], np.int32))
        et_loo, d_loo = loo.predict(im)
        eq = cv2.equalizeHist(im)
        fila = dict(indice=indice, identidad=e['nombre'], archivo=e['ruta'],
                    etiqueta_esperada=e['etiqueta'], etiqueta_devuelta=int(etiqueta),
                    candidato=nombres[int(etiqueta)], distancia=float(distancia),
                    coherente=int(etiqueta)==e['etiqueta'],
                    loo_candidato=nombres[int(et_loo)], loo_distancia=float(d_loo),
                    gris_media=float(im.mean()), gris_desviacion=float(im.std()),
                    laplaciano=float(cv2.Laplacian(im, cv2.CV_64F).var()),
                    segunda_eq_mae=float(np.abs(im.astype(float)-eq).mean()))
        for n in grupos:
            fila['principal_'+n] = d_main[n]
            fila['duplicados_'+n] = d_dup[n]
        assert abs(min(d_main.values())-distancia) < 1e-6
        filas.append(fila)
    assert all(p.read_bytes() == b for p, b in protegidos.items())
    lineas = ['# Auditoria offline LBPH', '',
              f'OpenCV {cv2.__version__}; etiquetas: {nombres}.',
              f'Entradas: {len(entradas)}; sin rutas ni hashes cruzados.',
              'Predicciones sobre entrenamiento: solo coherencia interna, NO generalizacion.', '',
              '| Identidad | n | Etiquetas devueltas | Distancia min/max | Inconsistencias | LOO correctas | LOO min/mediana/max |',
              '|---|---:|---|---|---:|---:|---|']
    for n in grupos:
        fs = [f for f in filas if f['identidad']==n]
        ds = [f['distancia'] for f in fs]; ls = [f['loo_distancia'] for f in fs]
        lineas.append(f"| {n} | {len(fs)} | {dict(Counter(f['etiqueta_devuelta'] for f in fs))} | {min(ds):.6f}/{max(ds):.6f} | {sum(not f['coherente'] for f in fs)} | {sum(f['loo_candidato']==n for f in fs)}/{len(fs)} | {min(ls):.2f}/{statistics.median(ls):.2f}/{max(ls):.2f} |")
    diferencias = [abs(f['principal_'+n]-f['duplicados_'+n]) for f in filas for n in grupos]
    lineas += ['', 'LOO excluye una foto pero conserva fotos de la misma sesion: no mide generalizacion a una nueva sesion ni falsos positivos en desconocidos.',
               f'Diferencia absoluta principal/duplicados sobre las 270 distancias por identidad: media={statistics.mean(diferencias):.4f}; maxima={max(diferencias):.4f}.',
               f'Huella del flujo de duplicados (sin cambios): {huella}.',
               f"Preparacion de altas permitida: {preparacion_registro(DB_PATH, FOTOS_DIR)['permitido']}.",
               'Base y fotos verificadas sin cambios byte a byte. No se abrio hardware.']
    return filas, '\n'.join(lineas)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, help='Carpeta explicita para CSV y resumen Markdown; sin imagenes')
    args = parser.parse_args()
    filas, resumen = auditar()
    print(resumen)
    if args.salida:
        args.salida.mkdir(parents=True, exist_ok=True)
        with (args.salida/'entradas_entrenamiento.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
        (args.salida/'resumen.md').write_text(resumen, encoding='utf-8')
    else:
        for fila in filas:
            print(f"{fila['indice']:03d} {fila['identidad']} etiqueta={fila['etiqueta_devuelta']} distancia={fila['distancia']:.6f} coherente={fila['coherente']}")


if __name__ == '__main__':
    main()
