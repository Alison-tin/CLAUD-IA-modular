"""Recalcula diez recorridos conservados; no registra personas ni abre hardware."""
import ast
import csv
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import cv2
    import numpy as np
    from interfaz.alta_sface import decidir
    from evaluar_sface import SFace,csv_leer,csv_guardar
    from vision.galeria_sface import cargar,digest
    grupos,_,version=cargar(); modelo=SFace();resultados=[]
    with patch('cv2.VideoCapture',side_effect=AssertionError('Sin camara')),patch('serial.Serial',side_effect=AssertionError('Sin Arduino')):
        for p in sorted((ROOT/'evaluacion_duplicados_sface').glob('*/puntuaciones.csv')):
            texto=(p.parent/'resumen.md').read_text(encoding='utf-8')
            datos=ast.literal_eval(next(l.split(': ',1)[1] for l in texto.splitlines() if l.startswith('Participante declarado: ')))
            assert 'Galeria: '+version in texto
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
    print(f'{len(resultados)} recorridos verificados; 6 duplicados declarados y 4 externas nuevas, sin altas ni cambios en las fuentes.')
    print('Regla local provisional: retrospectiva sobre el mismo conjunto, NO validacion independiente.')


if __name__=='__main__':main()
