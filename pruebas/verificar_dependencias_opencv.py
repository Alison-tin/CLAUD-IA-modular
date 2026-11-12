"""Sondas sintéticas; nunca abre cámara/serie ni usa fotografías de personas."""
import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import distributions
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emociones', action='store_true')
    parser.add_argument('--ventana', action='store_true', help='Ventana sintética breve, sin cámara')
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    import cv2
    import numpy as np
    variantes = sorted(d.metadata['Name'] for d in distributions()
                       if d.metadata['Name'].lower().startswith('opencv-'))
    print('Python:', sys.version.split()[0], 'OpenCV:', cv2.__version__)
    print('Variantes:', variantes)
    assert len(variantes) == 1, 'Instalacion superpuesta: no es una combinacion limpia'
    print('GUI:', next(s.strip() for s in cv2.getBuildInformation().splitlines() if 'GUI:' in s))
    imagen = np.random.default_rng(2026).integers(0, 256, (320, 320, 3), np.uint8)
    with ExitStack() as stack:
        stack.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camara prohibida')))
        stack.enter_context(patch('socket.socket.connect', side_effect=AssertionError('Red prohibida en procesamiento')))
        try:
            import serial
        except ImportError:
            pass
        else:
            stack.enter_context(patch.object(serial, 'Serial', side_effect=AssertionError('Serie prohibida')))
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        modelo = cv2.face.LBPHFaceRecognizer_create()
        modelo.train([gris], np.array([7], np.int32))
        etiqueta, distancia = modelo.predict(gris)
        assert etiqueta == 7 and abs(distancia) < 1e-6
        print('LBPH: entrenamiento/prediccion sinteticos OK; no mide exactitud facial')
        from evaluar_sface import SFace
        sf = SFace()
        caras = sf.detectar(np.zeros_like(imagen))
        assert len(caras) == 0
        # Landmarks inventados: prueba únicamente alineación y ejecución del grafo.
        fila = np.array([65, 65, 160, 190, 110, 120, 180, 120, 145, 160,
                         115, 200, 175, 200, .99], np.float32)
        feature, error = sf.extraer_cara(imagen, fila)
        assert not error and feature.shape == (1, 128)
        coseno = sf.reconocedor.match(feature, feature.copy(), cv2.FaceRecognizerSF_FR_COSINE)
        assert np.isfinite(coseno) and abs(coseno-1) < 1e-4
        print('YuNet cero rostros y SFace alineacion/embedding/coseno sinteticos OK')
        if args.ventana:
            try:
                cv2.namedWindow('Prueba sintetica OpenCV - sin camara', cv2.WINDOW_NORMAL)
                cv2.imshow('Prueba sintetica OpenCV - sin camara', imagen)
                cv2.waitKey(100)
            finally:
                cv2.destroyAllWindows()
            print('HighGUI: ventana sintetica creada, presentada y cerrada')
        if args.emociones:
            try:
                cv2.namedWindow('Headless no admite ventanas')
            except cv2.error as error:
                assert 'not implemented' in str(error).lower()
                print('HighGUI headless: rechazo esperado; no sirve para ventanas de comparacion')
            else:
                cv2.destroyAllWindows()
                raise AssertionError('La variante headless inesperadamente admite ventanas')
            with tempfile.TemporaryDirectory(prefix='claud_emociones_') as temporal:
                destino = Path(temporal)/'.deepface'/'weights'
                destino.mkdir(parents=True)
                origen = Path.home()/'.deepface'/'weights'
                for nombre in ('facial_expression_model_weights.onnx', 'haarcascade_eye.xml',
                               'haarcascade_frontalface_default.xml'):
                    fuente = origen/nombre
                    if not fuente.is_file():
                        raise FileNotFoundError(f'Modelo local ausente: {fuente}; no se descarga en esta prueba')
                    shutil.copyfile(fuente, destino/nombre)
                    assert hashlib.sha256(fuente.read_bytes()).digest() == hashlib.sha256((destino/nombre).read_bytes()).digest()
                with patch.dict(os.environ, {'DEEPFACE_HOME': temporal, 'DEEPFACE_BACKEND_ENGINE': 'onnx'}):
                    from deepface import DeepFace
                    resultado = DeepFace.analyze(np.full((100, 100, 3), 128, np.uint8),
                        actions=['emotion'], enforce_detection=False, silent=True)
                    if isinstance(resultado, list):
                        resultado = resultado[0]
                    assert len(resultado['emotion']) == 7
                    assert all(np.isfinite(v) for v in resultado['emotion'].values())
                    print('DeepFace ONNX: analyze sintetico y siete salidas finitas OK; sin validez emocional')
        else:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from configuracion.config import MODELO_CARA_PATH, MODELO_MANO_PATH
            from vision.fotogramas import imagen_mediapipe
            with vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                    base_options=python.BaseOptions(model_asset_path=MODELO_CARA_PATH))) as detector:
                assert not detector.detect(imagen_mediapipe(np.zeros_like(imagen))).detections
            with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=MODELO_MANO_PATH))) as detector:
                assert not detector.detect(imagen_mediapipe(np.zeros_like(imagen))).hand_landmarks
            with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(ROOT/'face_landmarker.task')))) as detector:
                assert not detector.detect(imagen_mediapipe(np.zeros_like(imagen))).face_landmarks
            print('MediaPipe: cara/manos/landmarks ejecutados en imagen sintetica, sin detecciones')
    if args.suite:
        from verificar_integracion_sface import ejecutar
        return ejecutar(ROOT/'documentacion'/'PRUEBA_OPENCV_SUITE.log')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
