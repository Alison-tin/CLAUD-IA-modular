"""Experimento local YuNet/SFace frente a LBPH. No integra ni habilita altas.

Solo capturar y probar abren camara, cuando el operador los ejecuta.
No importa Arduino, movimientos, web ni servicios externos.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import platform
import re
import statistics
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent / 'evaluacion_sface'
MODELOS = ROOT / 'modelos'
REVISION = '48b10fa505aeb51fbd127fc60a5c660040e823b7'
MODELOS_SHA = {
    'face_detection_yunet_2023mar.onnx': '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    'face_recognition_sface_2021dec.onnx': '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
}
PERSONAS = ('Alexa', 'Alison', 'daniela')
COSENO_PROVISIONAL = 0.363  # Ejemplo OpenCV/LFW, NO calibracion local/open-set.
LBPH_UMBRAL = 58.0


def sha(contenido):
    return hashlib.sha256(contenido).hexdigest()


def huella_activa(base):
    from datos.personas import FOTOS_DIR
    from interfaz.duplicados import resolver_referencia
    huella = hashlib.sha256()
    for nombre, ruta in zip(base['nombres'], base['imagenes']):
        contenido = resolver_referencia(ruta, FOTOS_DIR).read_bytes()
        huella.update(repr((nombre, sha(contenido))).encode('utf-8'))
    return huella.hexdigest()


def csv_leer(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_guardar(path, filas):
    if not filas:
        return
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def verificar_modelos():
    for nombre, esperado in MODELOS_SHA.items():
        p = MODELOS / nombre
        if not p.is_file() or sha(p.read_bytes()) != esperado:
            raise ValueError(f'Modelo ausente o hash incorrecto: {p}. Consulta ORIGEN.md.')
    if not all(hasattr(cv2, api) for api in ('FaceDetectorYN_create', 'FaceRecognizerSF_create')):
        raise RuntimeError('OpenCV no tiene FaceDetectorYN/FaceRecognizerSF; conservar entorno activo.')


def entorno():
    return (f'Python={platform.python_version()}\nOpenCV={cv2.__version__}\nNumPy={np.__version__}\n'
            f'Plataforma={platform.platform()}\nOpenCV_Zoo={REVISION}\n'
            f'Modelos_SHA256={MODELOS_SHA}\n'
            'YuNet: letterbox 320x320, score=0.9, NMS=0.3, top_k=5000\n'
            'SFace: alignCrop con cinco landmarks, feature 128, maximo coseno por identidad\n'
            'DNN backend solicitado=OpenCV, target solicitado=CPU; ver advertencias de la compilacion\n'
            'LBPH: r=1,n=8,grid=8x8; margen=15%,gris,200x200,EQ unica\n'
            f'LBPH acepta d<{LBPH_UMBRAL}; SFace acepta cos>={COSENO_PROVISIONAL} PROVISIONAL\n'
            'Ninguna puntuacion es una probabilidad. Sin decision de alta.\n')


def validar_imagen(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
        raise ValueError('Se requiere una imagen BGR uint8 valida')


def letterbox(frame):
    validar_imagen(frame)
    h, w = frame.shape[:2]
    escala = min(320/w, 320/h)
    nw, nh = max(1, round(w*escala)), max(1, round(h*escala))
    salida = np.zeros((320, 320, 3), np.uint8)
    salida[:nh, :nw] = cv2.resize(frame, (nw, nh))
    return salida, (nw/w, nh/h)


def desescalar_cara(cara, escalas):
    cara = np.array(cara, np.float32, copy=True)
    cara[[0, 2, 4, 6, 8, 10, 12]] /= escalas[0]
    cara[[1, 3, 5, 7, 9, 11, 13]] /= escalas[1]
    return cara


class SFace:
    def __init__(self):
        verificar_modelos()
        self.detector = cv2.FaceDetectorYN_create(
            str(MODELOS/'face_detection_yunet_2023mar.onnx'), '', (320, 320), .9, .3, 5000,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        self.reconocedor = cv2.FaceRecognizerSF_create(
            str(MODELOS/'face_recognition_sface_2021dec.onnx'), '',
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)

    def detectar(self, frame):
        entrada, escalas = letterbox(frame)
        _, caras = self.detector.detect(entrada)
        return [] if caras is None else [desescalar_cara(cara, escalas) for cara in caras]

    def extraer_cara(self, frame, cara):
        if not np.isfinite(cara).all() or min(cara[2:4]) <= 0:
            return None, 'YuNet: caja/landmarks invalidos'
        alineada = self.reconocedor.alignCrop(frame, cara)
        feature = self.reconocedor.feature(alineada).copy()
        if feature.shape != (1, 128) or not np.isfinite(feature).all() or np.linalg.norm(feature) < 1e-8:
            return None, 'SFace: embedding invalido'
        return feature, ''

    def extraer(self, frame):
        caras = self.detectar(frame)
        if len(caras) != 1:
            return None, f'YuNet: {len(caras)} rostros; se exige uno'
        return self.extraer_cara(frame, caras[0])

    def comparar(self, feature, galeria, diagnostico=None):
        puntuaciones = {n: max(float(self.reconocedor.match(feature.copy(), f.copy(), cv2.FaceRecognizerSF_FR_COSINE))
                               for f in fs) for n, fs in galeria.items()}
        candidato = max(puntuaciones, key=puntuaciones.get)
        if diagnostico is not None:
            diagnostico.update(puntuaciones=puntuaciones.copy())
        score = puntuaciones[candidato]
        if not math.isfinite(score):
            raise ValueError('Coseno no finito')
        return candidato, score, candidato if score >= COSENO_PROVISIONAL else 'Desconocido'


class RecorteLBPH:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from configuracion.config import MODELO_CARA_PATH
        self.detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=MODELO_CARA_PATH),
            running_mode=vision.RunningMode.IMAGE, min_detection_confidence=.55))

    def extraer(self, frame):
        from vision.fotogramas import imagen_mediapipe
        from datos.personas import extraer_rostro
        caras = self.detector.detect(imagen_mediapipe(frame)).detections
        if len(caras) != 1:
            return None, f'MediaPipe: {len(caras)} rostros; se exige uno'
        b = caras[0].bounding_box
        im = extraer_rostro(frame, (b.origin_x, b.origin_y, b.width, b.height))
        return im, '' if im is not None else 'LBPH: recorte vacio'

    def close(self):
        self.detector.close()


def procesar(frame, sf, lb):
    salidas = {}
    for nombre, extractor in (('SFace', sf), ('LBPH', lb)):
        t = time.perf_counter()
        try:
            dato, error = extractor.extraer(frame)
        except (cv2.error, ValueError) as e:
            dato, error = None, str(e)
        salidas[nombre] = (dato, error, (time.perf_counter()-t)*1000)
    return salidas


def cargar_galeria(sf, lb, parcial=False):
    features, fotos = defaultdict(list), defaultdict(list)
    hashes, manifiestos = set(), []
    for manifest in sorted((ROOT/'referencias').glob('*/capturas.csv')):
        manifiestos.append(manifest.read_bytes())
        for fila in csv_leer(manifest):
            if fila['uso'] != 'referencias' or fila['persona'] not in PERSONAS:
                raise ValueError('Manifiesto de referencias invalido')
            archivo = archivo_local(manifest.parent, fila['archivo'])
            contenido = archivo.read_bytes()
            h = sha(contenido)
            if h != fila['sha256'] or h in hashes:
                raise ValueError('Referencia modificada o repetida')
            hashes.add(h)
            frame = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_COLOR)
            salidas = procesar(frame, sf, lb)
            if any(x[0] is None for x in salidas.values()):
                raise ValueError(f'Referencia no utilizable por ambos metodos: {archivo}')
            features[fila['persona']].append(salidas['SFace'][0])
            fotos[fila['persona']].append(salidas['LBPH'][0])
    if not features or (not parcial and set(features) != set(PERSONAS)):
        raise ValueError('Captura referencias de Alexa, Alison y daniela. Solo para iniciar con Alexa usa --galeria-parcial.')
    nombres = dict(enumerate(sorted(fotos)))
    modelo = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    modelo.train([im for n in nombres.values() for im in fotos[n]],
                 np.array([i for i, n in nombres.items() for _ in fotos[n]], np.int32))
    return features, modelo, nombres, hashes, sha(b''.join(manifiestos))


def archivo_local(carpeta, nombre):
    p = (carpeta/nombre).resolve()
    if p.parent != carpeta.resolve() or p.suffix != '.png':
        raise ValueError('Archivo fuera de la sesion o formato invalido')
    return p


def categoria(persona, tipo, decision):
    if decision == 'Sin lectura':
        return 'sin_lectura'
    if tipo == 'no_registrada':
        return 'rechazo_correcto' if decision == 'Desconocido' else 'aceptacion_desconocido'
    if decision == 'Desconocido':
        return 'rechazo'
    return 'acierto' if decision == persona else 'confusion'


def evaluar_salidas(salidas, sf, galeria, modelos, persona, tipo):
    from vision.reconocimiento import decidir_reconocimiento
    filas = []
    for metodo in ('SFace', *modelos):
        dato, error, pre_ms = salidas['SFace' if metodo == 'SFace' else 'LBPH']
        candidato, score, decision = '', '', 'Sin lectura'
        t = time.perf_counter()
        if dato is not None:
            try:
                if metodo == 'SFace':
                    candidato, score, decision = sf.comparar(dato, galeria)
                else:
                    modelo, nombres = modelos[metodo]
                    etiqueta, score = modelo.predict(dato)
                    candidato = nombres.get(int(etiqueta), '')
                    decision = decidir_reconocimiento(etiqueta, score, nombres, LBPH_UMBRAL)['nombre']
            except (cv2.error, ValueError) as e:
                candidato, score, decision, error = '', '', 'Sin lectura', str(e)
        match_ms = (time.perf_counter()-t)*1000
        filas.append(dict(metodo=metodo, candidato=candidato, puntuacion=score,
                          decision=decision, categoria=categoria(persona, tipo, decision),
                          preprocesamiento_ms=pre_ms, comparacion_ms=match_ms,
                          total_ms=pre_ms+match_ms, error=error,
                          umbral=COSENO_PROVISIONAL if metodo == 'SFace' else LBPH_UMBRAL))
    return filas


def resumen(filas):
    texto = ['# Evaluacion experimental: sin conclusion automatica de mejora', '',
             'SFace: coseno >=0.363 PROVISIONAL; LBPH: distancia <58. Ninguno es probabilidad.',
             'SFace y LBPH_nuevo usan las mismas referencias nuevas. LBPH_activo usa las 90 originales.',
             'Se exige un solo rostro por consulta en ambos metodos; no es la seleccion del mayor de la web.',
             'Sin lectura se informa aparte; no se cuenta como rechazo correcto de desconocidos.', '',
             '| Persona/tipo | Metodo | n | Aciertos | Rechazos registrados | Confusiones | Aceptaciones externos | Rechazos externos | Sin lectura | ms mediana/p95 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    galerias = {f.get('galeria_personas', '') for f in filas}
    aviso = 'Galerias nuevas presentes: '+', '.join(sorted(g for g in galerias if g))
    if any(set(g.split('|')) != set(PERSONAS) for g in galerias):
        aviso += '. GALERIA PARCIAL: no evalua confusiones entre las tres registradas.'
    texto.insert(2, aviso)
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f['persona'], f['tipo'], f['metodo'])].append(f)
    for (persona, tipo, metodo), fs in sorted(grupos.items()):
        c = Counter(f['categoria'] for f in fs)
        ts = [float(f['total_ms']) for f in fs]
        texto.append(f"| {persona}/{tipo} | {metodo} | {len(fs)} | {c['acierto']} | {c['rechazo']} | {c['confusion']} | {c['aceptacion_desconocido']} | {c['rechazo_correcto']} | {c['sin_lectura']} | {statistics.median(ts):.2f}/{np.percentile(ts,95):.2f} |")
    texto += ['', 'Denominador n incluye todas las capturas, tambien errores de deteccion. En registrados,',
              'sin_lectura tambien es un fallo operativo, adicional al rechazo por umbral.',
              'Los tiempos incluyen deteccion/preprocesamiento/alineacion y comparacion; excluyen camara,',
              'entrenamiento, disco y visualizacion. El preprocesamiento LBPH se comparte, se atribuye',
              'a cada variante para comparar pipelines; los tiempos de variantes no deben sumarse.',
              'Los fotogramas de una sesion estan correlacionados. Comparar por persona y sesion,',
              'con mas de un externo y sesiones posteriores. No ajustar y validar con las mismas sesiones.']
    return '\n'.join(texto)+'\n'


def guardar_imagen(destino, frame):
    ok, contenido = cv2.imencode('.png', frame)
    if not ok:
        raise ValueError('No se pudo codificar PNG')
    contenido = contenido.tobytes()
    with destino.open('xb') as f:
        f.write(contenido)
    return sha(contenido)


def nueva_sesion(uso, persona):
    identificador = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:10]
    p = ROOT/uso/(persona+'_'+identificador)
    p.mkdir(parents=True, exist_ok=False)
    (p/'entorno.txt').write_text(entorno(), encoding='utf-8')
    return p


def capturar(args):
    if args.persona not in PERSONAS and args.tipo != 'no_registrada':
        raise ValueError('Persona registrada desconocida')
    if args.tipo == 'no_registrada' and args.persona in PERSONAS:
        raise ValueError('Un externo necesita un identificador distinto de los nombres activos')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', args.persona):
        raise ValueError('Identificador de 1 a 60 letras ASCII, numeros, guion o guion bajo')
    referencias = args.accion == 'capturar'
    if referencias and args.tipo != 'registrada':
        raise ValueError('Las personas externas no se anaden a la galeria')
    sf, lb = SFace(), RecorteLBPH()
    cap = None
    filas, capturas = [], []
    destino = None
    try:
        hashes = set()
        if not referencias:
            galeria, nuevo, nombres, hashes, huella = cargar_galeria(sf, lb, args.galeria_parcial)
            if args.tipo == 'registrada' and args.persona not in galeria:
                raise ValueError('La persona evaluada no tiene referencias separadas')
            from datos.personas import entrenar_reconocedor, cargar_db
            base_activa = cargar_db()
            huella_lbph_activo = huella_activa(base_activa)
            activo, nombres_activos = entrenar_reconocedor(db=base_activa)
            if activo is None:
                raise ValueError('LBPH activo no esta disponible')
            modelos = {'LBPH_nuevo': (nuevo, nombres), 'LBPH_activo': (activo, nombres_activos)}
            print('Galeria:', {n: len(fs) for n, fs in galeria.items()}, 'parcial=', args.galeria_parcial)
        # Esta es la unica apertura de camara; solo en comandos explicitos del operador.
        from hardware.camara import abrir_camara
        cap = abrir_camara(args.cam)
        if not cap.isOpened():
            raise RuntimeError('Camara no disponible')
        print('ESPACIO inicia captura temporizada; Q termina. Se guardan PNG originales SOLO en evaluacion_sface.')
        print('Una persona, cara completa. Mover suavemente la cabeza; no cambiar de participante durante la sesion.')
        iniciado, siguiente, numero = False, 0., 0
        cantidad = args.cantidad or (5 if referencias else 15)
        while numero < cantidad:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('La camara dejo de entregar imagenes')
            validar_imagen(frame)
            vista = frame.copy()
            cv2.putText(vista, f'{args.persona} {numero}/{cantidad} | SPACE=iniciar Q=salir',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            cv2.imshow('Evaluacion local SFace / LBPH', vista)
            tecla = cv2.waitKey(1) & 0xff
            if tecla in (ord('q'), ord('Q')):
                break
            if tecla == 32 and not iniciado:
                iniciado, siguiente = True, time.monotonic()+2
            if not iniciado or time.monotonic() < siguiente:
                continue
            siguiente = time.monotonic()+args.intervalo
            salidas = procesar(frame, sf, lb)
            if referencias:
                if np.array_equal(frame[:,:,0], frame[:,:,1]) and np.array_equal(frame[:,:,1], frame[:,:,2]):
                    print('Referencia omitida: la camara entrega gris; no se simula recuperar color.')
                    continue
                if any(s[0] is None for s in salidas.values()):
                    print('Referencia omitida:', [s[1] for s in salidas.values() if s[0] is None])
                    continue
            if destino is None:
                destino = nueva_sesion('referencias' if referencias else 'sesiones', args.persona)
            # Comparar antes de superponer textos; ambos reciben exactamente el mismo frame.
            actuales = [] if referencias else evaluar_salidas(salidas, sf, galeria, modelos, args.persona, args.tipo)
            okpng, png = cv2.imencode('.png', frame)
            if not okpng:
                raise ValueError('Error de codificacion PNG')
            h = sha(png.tobytes())
            if h in hashes:
                raise ValueError('Imagen identica a una referencia o captura previa: no se permite reutilizarla')
            numero += 1
            nombre = f'captura_{numero:03d}.png'
            guardar_imagen(destino/nombre, frame)
            hashes.add(h)
            captura = dict(archivo=nombre, sha256=h, persona=args.persona, tipo=args.tipo,
                           uso='referencias' if referencias else 'evaluacion',
                           sesion=destino.name, utc=datetime.now(timezone.utc).isoformat(),
                           camara=args.cam, ancho=frame.shape[1], alto=frame.shape[0])
            capturas.append(captura)
            for i, fila in enumerate(actuales):
                fila.update(captura)
                fila['galeria_sha256'] = huella
                fila['galeria_personas'] = '|'.join(sorted(galeria))
                fila['lbph_activo_sha256'] = huella_lbph_activo
                filas.append(fila)
                print(f"{numero:02d} {fila['metodo']} candidato={fila['candidato']} puntuacion={fila['puntuacion']} decision={fila['decision']} ms={fila['total_ms']:.2f}")
                cv2.putText(vista, f"{fila['metodo']}: {fila['candidato']} {fila['puntuacion']} -> {fila['decision']}",
                            (12, 60+25*i), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,0), 1)
            if actuales:
                cv2.putText(vista, f'CAPTURA GUARDADA {numero} (imagen fija)', (12, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
                cv2.imshow('Ultima captura evaluada (no video en vivo)', vista)
                if cv2.waitKey(150) & 0xff in (ord('q'), ord('Q')):
                    break
            print('Guardada', numero, '/', cantidad)
    finally:
        if cap is not None:
            cap.release()
        lb.close()
        cv2.destroyAllWindows()
        if destino is not None:
            # Solo referencias completas entran en galeria. Las parciales quedan documentadas.
            completo = len(capturas) == (args.cantidad or (5 if referencias else 15))
            manifest = 'capturas.csv' if not referencias or completo else 'capturas_incompletas.csv'
            csv_guardar(destino/manifest, capturas)
            if filas:
                csv_guardar(destino/'resultados.csv', filas)
                (destino/'resumen.md').write_text(resumen(filas), encoding='utf-8')
            print('Sesion conservada:', destino, 'completa=', completo)


def auditar_antiguas():
    from datos.personas import cargar_db, FOTOS_DIR
    from interfaz.duplicados import resolver_referencia
    base = cargar_db()
    sf = SFace()
    filas = []
    for n, referencia in zip(base['nombres'], base['imagenes']):
        p = resolver_referencia(referencia, FOTOS_DIR)
        contenido = p.read_bytes()
        original = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_UNCHANGED)
        gris = original.ndim == 2 or (np.array_equal(original[:,:,0], original[:,:,1]) and np.array_equal(original[:,:,1], original[:,:,2]))
        # Replica de canales SOLO para satisfacer la API. No recupera informacion.
        entrada = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if original.ndim == 2 else original
        t = time.perf_counter()
        f, error = sf.extraer(entrada)
        filas.append(dict(persona=n, archivo=str(p), sha256=sha(contenido), gris=gris,
                          ancho=entrada.shape[1], alto=entrada.shape[0], embedding_valido=f is not None,
                          error=error, ms=(time.perf_counter()-t)*1000))
    destino = ROOT/('auditoria_antiguas_'+uuid.uuid4().hex[:10])
    destino.mkdir(parents=True, exist_ok=False)
    csv_guardar(destino/'auditoria.csv', filas)
    lineas = ['# Compatibilidad tecnica de fotos antiguas', '', entorno(),
              'No es una evaluacion de identidad ni de precision. Replicar gris en tres canales no restaura color.',
              'Una deteccion/embedding valido no demuestra referencias adecuadas para generalizar.', '',
              '| Persona | n | Grises | Embeddings validos |', '|---|---:|---:|---:|']
    for n in sorted(set(base['nombres'])):
        fs = [f for f in filas if f['persona']==n]
        lineas.append(f"| {n} | {len(fs)} | {sum(f['gris'] for f in fs)} | {sum(f['embedding_valido'] for f in fs)} |")
    lineas += ['', 'Se prefieren referencias nuevas BGR originales y sesiones independientes.',
               'No se modificaron ni convirtieron en disco las fotos activas.']
    (destino/'resumen.md').write_text('\n'.join(lineas), encoding='utf-8')
    print('\n'.join(lineas)); print(destino)


def informar():
    por_galeria = defaultdict(list)
    for p in sorted((ROOT/'sesiones').glob('*/resultados.csv')):
        for f in csv_leer(p):
            clave = (f['galeria_sha256'], f['lbph_activo_sha256'])
            por_galeria[clave].append(f)
    if not por_galeria:
        print('Sin sesiones reales de evaluacion. No hay evidencia de mejora.')
    for (huella, huella_lbph), filas in por_galeria.items():
        # Nunca mezclar galerias diferentes como si fueran el mismo experimento.
        salida = ROOT/f'informe_{huella[:12]}_{uuid.uuid4().hex[:8]}.md'
        texto = f'Galeria SHA256: {huella}\nLBPH activo SHA256: {huella_lbph}\n\n'+resumen(filas)
        # Conservar tambien desgloses por sesion para no ocultar variaciones.
        for sesion in sorted({f['sesion'] for f in filas}):
            texto += '\n## Sesion '+sesion+'\n\n'+resumen([f for f in filas if f['sesion']==sesion])
        salida.write_text(texto, encoding='utf-8')
        print(texto); print(salida)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='accion', required=True)
    subs.add_parser('comprobar', help='Versiones e inferencia sintetica, sin dispositivos')
    subs.add_parser('auditar-antiguas', help='Solo lectura de fotos activas; informe separado')
    subs.add_parser('informar', help='Resume sesiones reales existentes sin abrir dispositivos')
    for nombre in ('capturar', 'probar'):
        p = subs.add_parser(nombre)
        p.add_argument('--persona', required=True)
        p.add_argument('--tipo', choices=('registrada','no_registrada'), default='registrada')
        p.add_argument('--cam', type=int, default=0)
        p.add_argument('--cantidad', type=int)
        p.add_argument('--intervalo', type=float, default=1.0)
        p.add_argument('--galeria-parcial', action='store_true')
    args = parser.parse_args()
    if args.accion in ('capturar','probar'):
        if (args.cantidad is not None and not 1 <= args.cantidad <= 1000) or not .2 <= args.intervalo <= 60:
            parser.error('Cantidad 1..1000; intervalo .2..60 segundos')
        if not 0 <= args.cam <= 9:
            parser.error('Camara 0..9')
        capturar(args)
    elif args.accion == 'comprobar':
        print(entorno()); sf = SFace()
        assert sf.extraer(np.zeros((720,1280,3),np.uint8))[0] is None
        f = sf.reconocedor.feature(np.zeros((112,112,3),np.uint8))
        assert f.shape == (1,128) and np.isfinite(f).all()
        print('Inferencia YuNet/SFace OK. Prueba sintetica, ninguna afirmacion de precision.')
    elif args.accion == 'auditar-antiguas':
        auditar_antiguas()
    else:
        informar()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, cv2.error) as e:
        raise SystemExit(str(e))
