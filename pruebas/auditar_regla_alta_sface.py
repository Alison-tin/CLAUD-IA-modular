"""Recalcula diez recorridos conservados; no registra personas ni abre hardware."""
import ast
import configparser
import csv
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def galeria_historica(version):
    """Solo auditoría: usar la galería original, nunca activarla en el sistema."""
    import numpy as np
    from evaluar_sface import csv_leer
    from vision.galeria_sface import digest
    if len(version)!=64 or any(c not in '0123456789abcdef' for c in version):
        raise ValueError('Version historica invalida')
    carpeta=ROOT/'datos/galerias_sface'/('v1_'+version)
    meta=configparser.ConfigParser();meta.read(carpeta/'metadatos.ini',encoding='utf-8')
    m=meta['galeria'];assert m['version']==version
    for nombre,campo in [('embeddings.npz','embeddings_sha256'),('referencias.csv','referencias_sha256')]:
        assert digest((carpeta/nombre).read_bytes())==m[campo]
    filas=csv_leer(carpeta/'referencias.csv')
    for fila in filas:
        p=(ROOT/'evaluacion_sface'/fila['archivo']).resolve()
        assert p.is_relative_to((ROOT/'evaluacion_sface').resolve())
        assert digest(p.read_bytes())==fila['sha256']
    with np.load(carpeta/'embeddings.npz',allow_pickle=False) as datos:
        features=datos['features'].copy();nombres=datos['nombres'].tolist()
    assert nombres==[f['persona'] for f in filas]
    assert features.shape==(len(filas),128) and np.isfinite(features).all()
    assert np.all(np.linalg.norm(features,axis=1)>1e-8)
    grupos={}
    for nombre,feature in zip(nombres,features):grupos.setdefault(nombre,[]).append(feature.reshape(1,128))
    return grupos


def main():
    import cv2
    import numpy as np
    from interfaz.alta_sface import decidir
    from evaluar_sface import SFace,csv_leer,csv_guardar
    from vision.galeria_sface import cargar,digest
    grupos,_,version=cargar(); galerias={version:grupos};modelo=SFace();resultados=[]
    with patch('cv2.VideoCapture',side_effect=AssertionError('Sin camara')),patch('serial.Serial',side_effect=AssertionError('Sin Arduino')):
        for p in sorted((ROOT/'evaluacion_duplicados_sface').glob('*/puntuaciones.csv')):
            texto=(p.parent/'resumen.md').read_text(encoding='utf-8')
            datos=ast.literal_eval(next(l.split(': ',1)[1] for l in texto.splitlines() if l.startswith('Participante declarado: ')))
            original=next(l.split(': ',1)[1] for l in texto.splitlines() if l.startswith('Galeria: '))
            if original not in galerias:galerias[original]=galeria_historica(original)
            grupos=galerias[original]
            filas=csv_leer(p); lecturas=[];diferencia=0.
            for postura in ('frente','derecha','izquierda','arriba','abajo'):
                rows=[r for r in filas if r['postura']==postura]
                assert {r['persona'] for r in rows}==set(grupos)
                imagen=(p.parent/(postura+'.png')).read_bytes()
                assert len(rows)==len(grupos) and all(r['sha256']==digest(imagen) for r in rows)
                frame=cv2.imdecode(np.frombuffer(imagen,np.uint8),cv2.IMREAD_COLOR)
                f,e=modelo.extraer(frame);assert f is not None,e
                scores={n:max(float(modelo.reconocedor.match(f.copy(),r.copy(),cv2.FaceRecognizerSF_FR_COSINE)) for r in refs) for n,refs in grupos.items()}
                for row in rows:
                    delta=abs(scores[row['persona']]-float(row['coseno']));diferencia=max(diferencia,delta)
                    assert delta<1e-5
                ranking=sorted(scores,key=scores.get,reverse=True)
                for row in rows:
                    assert row['primera']==ranking[0] and row['segunda']==ranking[1]
                    assert math.isclose(float(row['separacion']),scores[ranking[0]]-scores[ranking[1]],abs_tol=1e-5)
                lecturas.append(scores)
            estado,persona=decidir(lecturas,grupos)
            esperado='duplicado' if datos['tipo']=='registrada' else 'nueva'
            assert estado==esperado,(p.parent.name,estado)
            if estado=='duplicado': assert persona==datos['identificador']
            resultados.append(dict(recorrido=p.parent.name,declaracion=datos['identificador'],tipo=datos['tipo'],
                resultado=estado,persona=persona or '',maximo=max(max(r.values()) for r in lecturas),
                minima_ganadora=min(max(r.values()) for r in lecturas),diferencia_recalculo=diferencia))
    salida=ROOT/'documentacion/AUDITORIA_REGLA_ALTA_SFACE.csv'
    if salida.exists():
        assert csv_leer(salida)==[{k:str(v) for k,v in r.items()} for r in resultados]
    else:
        csv_guardar(salida,resultados)
    print(f'{len(resultados)} recorridos verificados con sus galerias originales; sin altas ni cambios en las fuentes.')
    print('Regla local provisional: retrospectiva sobre el mismo conjunto, NO validacion independiente.')


if __name__=='__main__':main()
