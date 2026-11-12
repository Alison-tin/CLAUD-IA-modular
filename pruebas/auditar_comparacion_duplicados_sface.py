"""Recalculo local y comparacion de dos recorridos existentes, sin dispositivos."""
from pathlib import Path
from collections import Counter
import ast
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
from auditar_alison_alias import sha, pixeles
from evaluar_sface import ROOT as EVAL, SFace, csv_leer, csv_guardar
from configuracion.config import DB_PATH,FOTOS_DIR
from interfaz.evaluacion import POSTURAS
from interfaz.registro import leer_base
from interfaz.duplicados import resolver_referencia,preparacion_registro
from vision.galeria_sface import cargar,inventario

SESIONES = {
    'Alison_Prueba':'recorrido_3d774410b99a4bed8b701f7608d214c3',
    'EXTERNA_001':'recorrido_648976afa0744b2da9ef5f9a8bb7f962',
}
SALIDA = ROOT/'pruebas/comparacion_duplicados_sface_Alison_EXTERNA_001'


def auditar():
    protegidos=set(ROOT.rglob('*.json')) | set(ROOT.parent.glob('*.json'))
    for carpeta in ('configuracion','respaldos','respaldo_configuracion_20261026_190346',
                    'hardware','control','movimientos','secuencias','datos/galerias_sface',
                    'evaluacion_sface','evaluacion_duplicados_sface','interfaz','vision','aplicacion','web_local'):
        protegidos.update(p for p in (ROOT/carpeta).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protegidos.update((ROOT/'main.py',ROOT/'evaluar_sface.py'))
    antes={p:sha(p.read_bytes()) for p in protegidos if p.is_file()}
    grupos,base_sha,version=cargar()
    referencias,_,actual=inventario()
    assert version==actual
    base=leer_base(DB_PATH)
    assert Counter(base['nombres'])=={'Alexa':30,'Alison':30,'daniela':30}
    assert not {'EXTERNA_001','Alison_Prueba'} & set(base['nombres'])
    fuentes=[resolver_referencia(p,FOTOS_DIR) for p in base['imagenes']]
    fuentes += [EVAL/f['archivo'] for f in referencias]
    fuente_sha={sha(p.read_bytes()) for p in fuentes}
    fuente_pix={pixeles(p.read_bytes()) for p in fuentes}
    modelo=SFace()
    resultados={};inventario_fuentes=[];dif=[];imagenes_sha=set();imagenes_pix=set()
    for etiqueta,identificador in SESIONES.items():
        carpeta=ROOT/'evaluacion_duplicados_sface'/identificador
        assert {p.stem for p in carpeta.glob('*.png')}==set(POSTURAS)
        resumen=(carpeta/'resumen.md').read_text(encoding='utf-8')
        participante=ast.literal_eval(next(l for l in resumen.splitlines() if l.startswith('Participante declarado: ')).split(': ',1)[1])
        assert participante['acepta_conservar'] is True and participante['motor']=='sface'
        if etiqueta=='EXTERNA_001':
            assert participante['identificador']==etiqueta and participante['tipo']=='no_registrada'
            assert participante['nombre_propuesto']==''
        else:
            assert participante['identificador']=='Alison' and participante['tipo']=='registrada'
            assert participante['nombre_propuesto']=='Alison_Prueba'
        assert 'Galeria: '+version in resumen
        assert csv_leer(carpeta/'referencias_galeria.csv')==referencias
        filas=csv_leer(carpeta/'puntuaciones.csv')
        assert len(filas)==15
        resultados[etiqueta]=[]
        for postura in POSTURAS:
            contenido=(carpeta/(postura+'.png')).read_bytes()
            h=sha(contenido);hp=pixeles(contenido)
            assert h not in fuente_sha and hp not in fuente_pix
            assert h not in imagenes_sha and hp not in imagenes_pix
            imagenes_sha.add(h);imagenes_pix.add(hp)
            im=cv2.imdecode(np.frombuffer(contenido,np.uint8),cv2.IMREAD_UNCHANGED)
            assert im.dtype==np.uint8 and im.shape==(720,1280,3)
            assert not (np.array_equal(im[:,:,0],im[:,:,1]) and np.array_equal(im[:,:,1],im[:,:,2]))
            fs=[f for f in filas if f['postura']==postura]
            assert len(fs)==3 and {f['persona'] for f in fs}==set(grupos)
            assert all(f['sha256']==h and not f['error'] for f in fs)
            scores={f['persona']:float(f['coseno']) for f in fs}
            assert all(np.isfinite(v) and -1.00001<=v<=1.00001 for v in scores.values())
            orden=sorted(scores,key=scores.get,reverse=True)
            gap=scores[orden[0]]-scores[orden[1]]
            assert all(f['primera']==orden[0] and f['segunda']==orden[1] and abs(float(f['separacion'])-gap)<1e-12 for f in fs)
            assert f'| {postura} | {orden[0]} | {orden[1]} | {gap} |  |' in resumen
            feature,error=modelo.extraer(im)
            assert feature is not None and not error
            for nombre,templates in grupos.items():
                nuevo=max(float(modelo.reconocedor.match(feature.copy(),ref.copy(),cv2.FaceRecognizerSF_FR_COSINE)) for ref in templates)
                dif.append(abs(nuevo-scores[nombre]));assert dif[-1]<1e-6
            resultados[etiqueta].append(dict(postura=postura,primera=orden[0],segunda=orden[1],separacion=gap,**scores))
        for p in sorted(carpeta.iterdir()):
            if p.is_file():inventario_fuentes.append(dict(archivo=str(p.relative_to(ROOT)),sha256=antes[p]))
    assert not preparacion_registro(DB_PATH,FOTOS_DIR)['permitido']
    assert all(sha(p.read_bytes())==h for p,h in antes.items())
    comparacion=[]
    for genuino,externo in zip(resultados['Alison_Prueba'],resultados['EXTERNA_001']):
        comparacion.append(dict(postura=genuino['postura'],coseno_Alison_genuino=genuino['Alison'],
            coseno_Alison_externo=externo['Alison'],diferencia_coseno=genuino['Alison']-externo['Alison'],
            separacion_genuino=genuino['separacion'],separacion_externo=externo['separacion'],
            diferencia_separacion=genuino['separacion']-externo['separacion'],
            primera_genuino=genuino['primera'],primera_externo=externo['primera'],
            segunda_genuino=genuino['segunda'],segunda_externo=externo['segunda']))
    SALIDA.mkdir(parents=True,exist_ok=True)
    for nombre,filas in (('fuentes_sha256.csv',inventario_fuentes),('comparacion_por_postura.csv',comparacion)):
        p=SALIDA/nombre
        if p.exists():
            assert csv_leer(p)==[{k:str(v) for k,v in f.items()} for f in filas]
        else:csv_guardar(p,filas)
    lineas=['AUDITORIA OFFLINE: dos recorridos, no diez ensayos independientes.',
            f'Galeria {version}; base {base_sha}; referencias '+str({n:len(fs) for n,fs in grupos.items()}),
            'Diez originales BGR 1280x720, hashes y pixeles distintos entre si y separados de 90 fotos activas/25 referencias.',
            f'30 cosenos recalculados; diferencia absoluta maxima CSV/modelo: {max(dif)}. Resumen y ranking coherentes.']
    for etiqueta,filas in resultados.items():
        lineas.append(etiqueta+' primeras='+str(dict(Counter(f['primera'] for f in filas))))
        for nombre in grupos:
            vs=[f[nombre] for f in filas]
            lineas.append(f'  {nombre}: min={min(vs):.8f}, media={statistics.mean(vs):.8f}, max={max(vs):.8f}')
        gs=[f['separacion'] for f in filas]
        lineas.append(f'  separacion: min={min(gs):.8f}, media={statistics.mean(gs):.8f}, max={max(gs):.8f}')
    lineas += ['Unanimidad de primera candidata no distingue estos dos recorridos. No se define corte ni decision automatica.',
               'Todos los archivos protegidos conservan sus hashes. Altas bloqueadas. Sin camara ni Arduino.']
    texto='\n'.join(lineas)+'\n'
    (SALIDA/'VERIFICACION.txt').write_text(texto,encoding='utf-8')
    print(texto)
    print('Documentos derivados:',SALIDA)


if __name__=='__main__':
    auditar()
