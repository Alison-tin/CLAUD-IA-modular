"""Auditoria offline del ensayo genuino declarado por Alison. Fuentes de solo lectura."""
from collections import Counter, defaultdict
from pathlib import Path
import ast
import hashlib
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
from evaluar_sface import ROOT as EVAL, SFace, csv_leer, csv_guardar
from configuracion.config import DB_PATH, FOTOS_DIR
from interfaz.registro import leer_base
from interfaz.duplicados import resolver_referencia, preparacion_registro
from interfaz.evaluacion import POSTURAS
from vision.galeria_sface import cargar, inventario, GALERIAS

SESION = ROOT/'evaluacion_duplicados_sface/recorrido_3d774410b99a4bed8b701f7608d214c3'
SALIDA = ROOT/'pruebas/auditoria_alison_alias_3d774410'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pixeles(data):
    im = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    assert im is not None
    return sha(str(im.shape).encode()+im.tobytes())


def auditar():
    protegidos = set(ROOT.rglob('*.json')) | set(ROOT.parent.glob('*.json'))
    for nombre in ('configuracion','respaldos','respaldo_configuracion_20261026_190346',
                   'hardware','movimientos','control','secuencias','datos/galerias_sface',
                   'evaluacion_sface','evaluacion_duplicados_sface','aplicacion','web_local'):
        protegidos.update(p for p in (ROOT/nombre).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protegidos.update((ROOT/'main.py',ROOT/'evaluar_sface.py'))
    antes = {p:sha(p.read_bytes()) for p in protegidos if p.is_file()}
    originales = sorted(SESION.glob('*.png'))
    assert {p.stem for p in originales} == set(POSTURAS)
    filas = csv_leer(SESION/'puntuaciones.csv')
    assert len(filas)==15
    grupos, base_sha, version = cargar()
    referencias, _, version_actual = inventario()
    assert version==version_actual
    assert csv_leer(SESION/'referencias_galeria.csv')==referencias
    resumen=(SESION/'resumen.md').read_text(encoding='utf-8')
    assert 'Galeria: '+version in resumen
    linea=next(l for l in resumen.splitlines() if l.startswith('Participante declarado: '))
    participante=ast.literal_eval(linea.split(': ',1)[1])
    assert participante['identificador']=='Alison' and participante['nombre_propuesto']=='Alison_Prueba'
    assert participante['tipo']=='registrada' and participante['acepta_conservar'] is True
    base=leer_base(DB_PATH)
    assert Counter(base['nombres'])=={'Alexa':30,'Alison':30,'daniela':30}
    assert 'Alison_Prueba' not in base['nombres']
    # Contrastar bytes y pixeles, para detectar tambien una simple recodificacion.
    fuentes = [resolver_referencia(r,FOTOS_DIR) for r in base['imagenes']]
    fuentes += [EVAL/f['archivo'] for f in referencias]
    bytes_fuente={sha(p.read_bytes()) for p in fuentes}
    pixeles_fuente={pixeles(p.read_bytes()) for p in fuentes}
    modelo=SFace()
    por_postura=defaultdict(list)
    for f in filas:
        assert f['postura'] in POSTURAS and f['persona'] in grupos and not f['error']
        por_postura[f['postura']].append(f)
    diferencias=[]; tabla=[]; hashes=[]
    for postura in POSTURAS:
        p=SESION/(postura+'.png'); contenido=p.read_bytes(); h=sha(contenido)
        im=cv2.imdecode(np.frombuffer(contenido,np.uint8),cv2.IMREAD_UNCHANGED)
        assert im.dtype==np.uint8 and im.shape==(720,1280,3)
        assert not (np.array_equal(im[:,:,0],im[:,:,1]) and np.array_equal(im[:,:,1],im[:,:,2]))
        assert h not in bytes_fuente and pixeles(contenido) not in pixeles_fuente
        hashes.append(h)
        fs=por_postura[postura]
        assert len(fs)==3 and {f['persona'] for f in fs}==set(grupos)
        assert {f['sha256'] for f in fs}=={h}
        scores={f['persona']:float(f['coseno']) for f in fs}
        orden=sorted(scores,key=scores.get,reverse=True)
        gap=scores[orden[0]]-scores[orden[1]]
        assert orden[0]=='Alison'
        assert all(f['primera']==orden[0] and f['segunda']==orden[1] and abs(float(f['separacion'])-gap)<1e-12 for f in fs)
        assert f'| {postura} | Alison | {orden[1]} | {gap} |  |' in resumen
        feature,error=modelo.extraer(im)
        assert feature is not None and not error
        for n,refs in grupos.items():
            recalculado=max(float(modelo.reconocedor.match(feature.copy(),ref.copy(),cv2.FaceRecognizerSF_FR_COSINE)) for ref in refs)
            diferencias.append(abs(recalculado-scores[n]))
            assert abs(recalculado-scores[n])<1e-6
        tabla.append(dict(postura=postura,Alison=scores['Alison'],segunda=orden[1],
                          coseno_segunda=scores[orden[1]],separacion=gap))
    assert len(set(hashes))==5
    assert not preparacion_registro(DB_PATH,FOTOS_DIR)['permitido']
    assert all(sha(p.read_bytes())==h for p,h in antes.items())
    SALIDA.mkdir(parents=True,exist_ok=True)
    # Solo documentacion derivada fuera de la carpeta original. No se escriben fotos.
    for nombre, datos in (('fuentes_sha256.csv',[dict(archivo=str(p.relative_to(ROOT)),sha256=antes[p]) for p in sorted(SESION.iterdir())]),
                          ('resultados_verificados.csv',tabla)):
        destino=SALIDA/nombre
        if destino.exists():
            # Auditoria repetible: se conserva el archivo si el contenido ya coincide.
            import io,csv
            s=io.StringIO(newline=''); w=csv.DictWriter(s,fieldnames=list(datos[0]));w.writeheader();w.writerows(datos)
            assert destino.read_text(encoding='utf-8-sig').replace('\r\n','\n')==s.getvalue().replace('\r\n','\n')
        else:
            csv_guardar(destino,datos)
    print('Cinco originales BGR 1280x720, hashes distintos y correctos; sin coincidencias de bytes/pixeles con 90 fotos activas o 25 referencias.')
    print('15 puntuaciones, primeras/segundas y separaciones coherentes con resumen y recalculo. Error maximo:',max(diferencias))
    print('Alison media:',statistics.mean(f['Alison'] for f in tabla),'rango:',min(f['Alison'] for f in tabla),max(f['Alison'] for f in tabla))
    print('Galeria:',version,'; referencias:',{n:len(fs) for n,fs in grupos.items()})
    print('Base activa 90 muestras, sin Alison_Prueba. Altas bloqueadas. Fuentes y archivos protegidos sin cambios.')
    print('Unidad experimental: UN recorrido genuino declarado de Alison con alias; cinco posturas correlacionadas, no cinco ensayos independientes.')
    print('Salida de auditoria:',SALIDA)


if __name__=='__main__':
    auditar()
