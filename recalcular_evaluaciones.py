"""Recalculo offline de Alison y Vanessa; conserva todas las fuentes."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import uuid
from configuracion.config import DB_PATH,FOTOS_DIR,RAIZ_PROYECTO
from interfaz.duplicados import cargar_referencias,crear_modelos,medir_muestra,preparacion_registro
from interfaz.evaluacion import POSTURAS,validar_participante
from interfaz.recorrido import PROTOCOLO
from interfaz.registro import leer_base


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    raiz=Path(RAIZ_PROYECTO)/'evaluacion_duplicados'
    fuentes={}
    for p in raiz.glob('*/resumen.json'):
        d=json.loads(p.read_text(encoding='utf-8-sig')); n=d['participante_declarado']['identificador']
        if n in ('Alison','Vanessa'):
            if n in fuentes: raise ValueError('Varios recorridos para una persona; se debe elegir la fuente.')
            fuentes[n]=(p,d)
    if set(fuentes)!={'Alison','Vanessa'}: raise ValueError('Faltan los recorridos originales.')
    base=leer_base(DB_PATH);grupos,huella=cargar_referencias(base,FOTOS_DIR)
    if set(grupos)!={'Alison','Alexa','daniela'}: raise ValueError('Base activa inesperada.')
    if preparacion_registro(DB_PATH,FOTOS_DIR)['permitido']: raise ValueError('Las altas deben seguir bloqueadas.')
    protegidos=[Path(DB_PATH),*Path(FOTOS_DIR).rglob('*'),*raiz.rglob('*')]
    protegidos={p.resolve():sha(p) for p in protegidos if p.is_file()}
    modelos=crear_modelos(grupos);informes={}
    for nombre,(p,anterior) in fuentes.items():
        if anterior['protocolo']!=PROTOCOLO: raise ValueError('Protocolo incompatible.')
        validar_participante(anterior['participante_declarado'],base)
        fotos=[p.parent/f'{pose}.jpg' for pose in POSTURAS]
        lecturas=[medir_muestra(f.read_bytes(),modelos) for f in fotos]
        personas=[]
        for identidad in sorted(modelos):
            valores=[x[identidad] for x in lecturas]
            personas.append(dict(nombre=identidad,posturas=dict(zip(POSTURAS,valores)),minima=min(valores),media=statistics.mean(valores),maxima=max(valores)))
        previos={x['nombre']:x for x in anterior['personas']}
        iguales=all(math.isclose(x['posturas'][pose],previos[x['nombre']]['posturas'][pose],rel_tol=0,abs_tol=1e-9) for x in personas for pose in POSTURAS)
        informes[nombre]={'version':1,'tipo_informe':'recalculo_sin_captura','fecha_recalculo_utc':datetime.now(timezone.utc).isoformat(),
            'protocolo':PROTOCOLO,'base_sha256':huella,'participante_declarado':anterior['participante_declarado'],
            'origen':{'resumen':str(p.resolve()),'resumen_sha256':sha(p),'base_sha256_anterior':anterior['base_sha256'],
                'fecha_captura_utc':anterior['fecha_utc'],'muestras':[{'postura':pose,'archivo':str(f.resolve()),'sha256':sha(f)} for pose,f in zip(POSTURAS,fotos)]},
            'metrica':anterior['metrica'],'decision':'Sin decision de identidad. Sin umbrales de alta.',
            'distancias_coinciden_con_columnas_conservadas':iguales,'personas':personas}
    _,actual=cargar_referencias(leer_base(DB_PATH),FOTOS_DIR)
    if actual!=huella or any(sha(p)!=h for p,h in protegidos.items()): raise ValueError('Las fuentes cambiaron; se cancela.')
    destino=raiz/'recalculos'/('recalculo_'+uuid.uuid4().hex)
    destino.mkdir(parents=True,exist_ok=False)
    for nombre,informe in informes.items():
        carpeta=destino/nombre;carpeta.mkdir()
        with (carpeta/'resumen.json').open('x',encoding='utf-8') as f: json.dump(informe,f,ensure_ascii=False,indent=2,allow_nan=False)
        with (carpeta/'distancias.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['persona',*POSTURAS,'minima','media','maxima'])
            for x in informe['personas']: w.writerow([x['nombre'],*x['posturas'].values(),x['minima'],x['media'],x['maxima']])
    assert all(sha(p)==h for p,h in protegidos.items())
    evidencia={'base_sha256':huella,'identidades_activas':sorted(grupos),'altas_bloqueadas':True,
        'archivos_fuente_verificados_sin_cambios':len(protegidos),'fuentes_sha256':{str(p):h for p,h in protegidos.items()}}
    with (destino/'verificacion.json').open('x',encoding='utf-8') as f: json.dump(evidencia,f,ensure_ascii=False,indent=2)
    print('RESULTADOS',destino);print('HUELLA',huella)
    for n,d in informes.items(): print(n,[(x['nombre'],round(x['media'],2)) for x in d['personas']],'coinciden',d['distancias_coinciden_con_columnas_conservadas'])


if __name__=='__main__': main()
