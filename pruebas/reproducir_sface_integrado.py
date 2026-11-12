"""Reproduccion offline de sesiones existentes. No es otra validacion independiente."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
from evaluar_sface import ROOT as EVAL, csv_leer, csv_guardar, RecorteLBPH, categoria, sha
from vision.fotogramas import imagen_mediapipe
from vision.motor_sface import MotorSFace


def reproducir():
    motor = MotorSFace()
    detector = RecorteLBPH()
    filas = []
    try:
        for manifest in sorted((EVAL/'sesiones').glob('*/capturas.csv')):
            resultados = csv_leer(manifest.parent/'resultados.csv')
            if set(resultados[0]['galeria_personas'].split('|')) != {'Alexa','Alison','daniela'}:
                continue  # La sesion parcial inicial no es la comparacion completa.
            for f in csv_leer(manifest):
                data = (manifest.parent/f['archivo']).read_bytes()
                assert sha(data)==f['sha256']
                frame = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                ds = detector.detector.detect(imagen_mediapipe(frame)).detections
                cajas = [(d.bounding_box.origin_x,d.bounding_box.origin_y,d.bounding_box.width,d.bounding_box.height) for d in ds]
                t = time.perf_counter()
                r = motor.reconocer(frame,cajas)
                filas.append(dict(sesion=manifest.parent.name,persona=f['persona'],tipo=f['tipo'],
                                  archivo=f['archivo'],decision=r['nombre'] or 'Sin lectura',
                                  categoria=categoria(f['persona'],f['tipo'],r['nombre'] or 'Sin lectura'),
                                  estado=r['estado_reconocimiento'],ms=(time.perf_counter()-t)*1000))
    finally:
        detector.close()
    if filas:
        import uuid
        p = ROOT/'pruebas'/('REPRODUCCION_SFACE_'+uuid.uuid4().hex[:8]+'.csv')
        csv_guardar(p,filas)
        from collections import Counter
        print('Reproduccion de datos ya evaluados, no nuevas capturas ni validacion independiente.')
        for tipo in ('registrada','no_registrada'):
            print(tipo,Counter(f['categoria'] for f in filas if f['tipo']==tipo))
        print('Galeria',motor.version,'CSV',p)


if __name__=='__main__':
    reproducir()
