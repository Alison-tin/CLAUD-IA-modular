"""
inmoov_vision.py  Sistema de vision

Uso:
  python inmoov_vision.py --cam 1 --port COM8
  python inmoov_vision.py --registrar "Nombre" --cam 1
  python inmoov_vision.py --listar
  python inmoov_vision.py --borrar "Nombre"
  python inmoov_vision.py --demo

Teclas:
  S  seguimiento ON/OFF
  R  reconocimiento ON/OFF
  G  gestos ON/OFF
  E  emocin ON/OFF
  Q  salir

"""

import os
# Suprimir logs verbosos de TensorFlow y oneDNN antes de cualquier import
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["GLOG_minloglevel"]       = "3"

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


import numpy as np
import pickle
import time
import argparse
import random
import math
import json
import threading
import urllib.request

from configuracion.config import *
from hardware.arduino import Arduino
from datos.personas import (
    cargar_db,
    guardar_db,
    listar_personas,
    registrar_persona,
    detectar_rostros,
    extraer_rostro,
    entrenar_reconocedor
)
from movimientos.expresiones import (
    mover_boca,
    hablar,
    expresion_saludo,
    expresion_neutro,
    expresion_reconocido
)
from movimientos.gestos import (
    _ojos_h,
    _ojos_v,
    _ojos_centro,
    _cabeza_centro,
    gesto_bostezo,
    gesto_pensar,
    gesto_mirar_alrededor,
    gesto_curiosear,
    gesto_estirar_cuello,
    gesto_doble_parpadeo,
    gesto_suspiro,
    GESTOS_VIDA
)
from secuencias.reproductor import (
    resolver_json,
    Reproductor
)
from vision.manos import (
    HAND_CONNECTIONS,
    dibujar_mano,
    dedos_arriba,
    dedo_ext,
    dist_lm,
    clasificar_gesto,
    GESTO_ACCION,
    GESTOS_HABLA,
    COLORES_G
)
from vision.emociones import (
    _cargar_deepface,
    analizar_emocion,
    deepface_estado
)
from control.pid import PID
# 
# CONFIGURACIN
# 
PCA  = 2
BAUD = 115200
SECUENCIAS_DIR = os.path.dirname(os.path.abspath(__file__))

SERVO = {
    # PCA 2 — cabeza facial
    0:(340,590,465), 1:(418,510,464), 2:(440,700,570), 3:(200,450,325),
    4:(410,640,525), 5:(435,665,550), 6:(470,565,517), 7:(525,625,575),
    8:(370,620,495), 9:(450,660,555),10:(570,700,635),11:(570,700,635),
   12:(420,600,510),13:(520,655,587),14:(435,580,507),15:(480,620,550),
}

# PCA 0 — cuello y maxilar
PCA0 = 0
SERVO_CUELLO = {
    # canal: (min, max, neutro, vel, nombre,        dir_min,  dir_max)
    15: (300, 800, 530, 1, "rotacion_cabeza",    "IZQ",    "DER"),
    14: (250, 750, 550, 4, "inclinacion_cabeza", "DER",    "IZQ"),
    12: (300, 650, 300, 5, "maxilar_inf_pca0",   "sube",   "baja"),
}

# Maxilar ahora en PCA0 canal 12
MAXILAR_PCA  = 0
MAXILAR_CANAL = 12
MAXILAR_CIERRA = 300   # sube (boca cerrada)
MAXILAR_ABRE   = 650   # baja (boca abierta)
PARPADOS_ABIERTOS = {12:600,13:520,14:580,15:480}
PARPADOS_CERRADOS = {12:420,13:655,14:435,15:620}

Kp,Ki,Kd       = 0.20, 0.002, 0.08   # ganancia reducida para movimiento suave
ZONA_MUERTA_X  = 30
ZONA_MUERTA_Y  = 25
VEL_SEG        = 4
VEL_EXP        = 3
TIMEOUT_ROSTRO = 2.0
PARPADEO_INT   = (3.0, 6.0)
TOLERANCIA_REC = 0.5
FOTOS_POR_PER  = 30
RECONOCER_CADA = 8
COOLDOWN_GESTO = 2.5
FRAMES_CONF    = 8

MODELO_CARA_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODELO_CARA_PATH = "blaze_face_short_range.tflite"
MODELO_MANO_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODELO_MANO_PATH = "hand_landmarker.task"
DB_PATH   = os.path.join(SECUENCIAS_DIR, "rostros_conocidos.pkl")
FOTOS_DIR = os.path.join(SECUENCIAS_DIR, "rostros_fotos")

# Emociones  espaol
EMOCIONES_ES = {
    "happy":"Feliz","sad":"Triste","angry":"Enojado",
    "surprise":"Sorprendido","fear":"Asustado",
    "disgust":"Disgustado","neutral":"Neutro",
}
# Color por emocin (BGR)
COLOR_EMOCION = {
    "Feliz":(0,220,0),"Triste":(200,100,0),"Enojado":(0,0,220),
    "Sorprendido":(0,200,255),"Asustado":(180,0,255),
    "Disgustado":(0,150,150),"Neutro":(180,180,180),
}

# 
# DEEPFACE  carga diferida para no ralentizar el arranque
# 
_df_state = {"ok": False, "intentado": False}  # dict mutable = compartido entre hilos



# 
# MODELOS MEDIAPIPE
# 
def descargar_modelos():
    for url, path, desc in [
        (MODELO_CARA_URL, MODELO_CARA_PATH, "cara (~1 MB)"),
        (MODELO_MANO_URL, MODELO_MANO_PATH, "mano (~24 MB)"),
    ]:
        if not os.path.exists(path):
            print(f" Descargando modelo {desc}...")
            urllib.request.urlretrieve(url, path)
            print(f" {path}")
        else:
            print(f" Modelo: {path}")




# Lista de gestos que se reproducen al azar cuando no hay nadie
GESTOS_VIDA = [
    gesto_bostezo, gesto_pensar, gesto_mirar_alrededor, gesto_curiosear,
    gesto_estirar_cuello, gesto_doble_parpadeo, gesto_suspiro,
]


# 
# HUD HELPERS
# 
def put(img,txt,x,y,escala=0.55,color=(255,255,255),grosor=1):
    cv2.putText(img,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX,escala,color,grosor,cv2.LINE_AA)

def panel(img,x,y,w,h,color=(30,30,30),alpha=0.55):
    sub=img[y:y+h,x:x+w]
    rect=np.full_like(sub,color)
    cv2.addWeighted(rect,alpha,sub,1-alpha,0,sub)
    img[y:y+h,x:x+w]=sub

def barra_conf(img,x,y,w,val,color,h=8):
    cv2.rectangle(img,(x,y),(x+w,y+h),(60,60,60),-1)
    cv2.rectangle(img,(x,y),(x+int(w*val),y+h),color,-1)

# 
# LOOP PRINCIPAL
# 
def ejecutar(cam_idx=0, port=None, demo=False):
    descargar_modelos()
    threading.Thread(target=_cargar_deepface, daemon=True).start()
    df_estado = deepface_estado()
    db = cargar_db()

    hay_personas = len(db["imagenes"]) > 0

    if hay_personas:
        listar_personas(db)
        reconocedor, nombres_por_id = entrenar_reconocedor()

        if reconocedor is not None:
            print(" Reconocimiento facial listo")
        else:
            print(" No se pudo crear el reconocedor")
    else:
        reconocedor = None
        nombres_por_id = {}
        print("  Sin personas registradas  solo deteccion")

    arduino=Arduino(port=port,demo=demo)
    rep=Reproductor(arduino)
    time.sleep(0.5); arduino.ir_a_centro(); arduino.parpado(abrir=True)
    # -- TEST MAXILAR: abre/cierra para verificar PCA0 canal 12 ----------
    arduino.mover_maxilar(abrir=True, vel=5);  time.sleep(0.5)
    arduino.mover_maxilar(abrir=False, vel=5)
    print(" Test maxilar enviado (PCA0 canal 12: abre->cierra)")
    print(" Posicin inicial enviada")

    # Detectores
    det_cara=mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELO_CARA_PATH),
            min_detection_confidence=0.55))

    det_mano=mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELO_MANO_PATH),
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5))

    cap=cv2.VideoCapture(cam_idx)

    # -- IMX179 AF 8MP  configuracin ptima ------------------------
    # 1280x720 @ 30fps: mejor tasa de refresco manteniendo buena calidad
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,   720)
    cap.set(cv2.CAP_PROP_FPS,            30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)     # frame ms reciente siempre
    cap.set(cv2.CAP_PROP_AUTOFOCUS,       1)     # autofoco continuo ON
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,   1)   # manual
    cap.set(cv2.CAP_PROP_BRIGHTNESS,    160)
    cap.set(cv2.CAP_PROP_CONTRAST,      128)
    cap.set(cv2.CAP_PROP_SATURATION,    128)
    cap.set(cv2.CAP_PROP_SHARPNESS,     150)
    cap.set(cv2.CAP_PROP_GAIN,           50)
    cap.set(cv2.CAP_PROP_EXPOSURE,       -3)   # mas luz posible

    # Verificar resolucin real obtenida
    rw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_real=cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolucion real: {rw}x{rh} @ {fps_real:.0f}fps")
    if not cap.isOpened(): print(f"Camara {cam_idx} no disponible"); return

    # Pantalla completa
    cv2.namedWindow("InMoov Vision", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("InMoov Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" Frame HUD: {W}x{H}")
    cx,cy=W//2,H//2
    print(f"Camara {cam_idx}: {W}x{H}")
    print("Activo | S=Seguim  I=Evadir  R=Reconoc  G=Gestos  E=Emocion  Q=salir\n")

    # Flags
    f_seg=True; f_rec=hay_personas; f_ges=True; f_emo=True; f_ign=False

    # Estado seguimiento
    pid_x=PID(Kp,Ki,Kd,ZONA_MUERTA_X); pid_y=PID(Kp,Ki,Kd,ZONA_MUERTA_Y)
    # PID separado para rotacion de cabeza (movimientos grandes)
    pid_cab=PID(0.15, 0.001, 0.05, ZONA_MUERTA_X*2)  # mas lento y suave que ojos
    pos_hor=SERVO[1][2]; pos_ver=SERVO[0][2]
    pos_rot=SERVO_CUELLO[15][2]   # posicion actual rotacion cabeza (neutro)
    rostro_presente=False; t_ult_rostro=0.0

    # Estado reconocimiento
    nombre_cache={}     # bbox_key  (nombre, conf, timestamp)
    nombre_primario=None
    frame_count=0

    # Estado emocin (anlisis en hilo)
    emo_cache={}        # bbox_key  (emo_es, conf, timestamp)
    emo_lock=threading.Lock()
    analizando=set()

    # Parpadeo
    t_ult_parp=time.time(); int_parp=random.uniform(*PARPADEO_INT)

    # Gestos de vida (idle) - se disparan cuando no hay nadie al frente
    t_ult_idle=time.time(); int_idle=random.uniform(7.0,14.0)
    idle_activo={"v":False}   # evita solapar gestos de vida

    # Gestos
    g_cand=None; g_frames=0; ult_g_t={}; ult_g_act=None

    def _parpado():
        arduino.parpado(abrir=False,vel=2); time.sleep(0.10)
        arduino.parpado(abrir=True,vel=2)

    def _analizar_emo(key, roi_rgb, x1,y1,x2,y2):
        emo,conf=analizar_emocion(roi_rgb,x1,y1,x2,y2)
        with emo_lock:
            if emo: emo_cache[key]=(emo,conf,time.time())
            analizando.discard(key)

    while True:
        df_estado = deepface_estado()
        ret,frame=cap.read()
        if not ret: break

        ahora=time.time(); frame_count+=1
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

        # -- Parpadeo automatico ----------------------------------------
        if ahora-t_ult_parp>int_parp and not idle_activo["v"]:
            threading.Thread(target=_parpado,daemon=True).start()
            t_ult_parp=ahora; int_parp=random.uniform(*PARPADEO_INT)

        # --------------------------------------------------------------
        # DETECCIN DE ROSTROS (todos)
        # --------------------------------------------------------------
        res_cara=det_cara.detect(mp_img)
        detecciones=res_cara.detections if res_cara.detections else []

        # Ordenar por rea (ms grande = ms cercano) descendente
        def area(d): b=d.bounding_box; return b.width*b.height
        detecciones=sorted(detecciones,key=area,reverse=True)

        n_personas=len(detecciones)

        # Primario = el ms grande
        primario=detecciones[0] if detecciones else None

        # -- Seguimiento (suave) o Evasion (brusca) ----------------------
        if primario and f_seg:
            bb=primario.bounding_box
            rx=int(bb.origin_x+bb.width/2); ry=int(bb.origin_y+bb.height/2)
            ex=rx-cx; ey=ry-cy

            # ── ROTACION DE CABEZA (movimientos grandes) ─────────────
            delta_cab = max(-5, min(5, pid_cab.update(ex) * 0.05))   # mas lento
            pos_rot = max(SERVO_CUELLO[15][0], min(SERVO_CUELLO[15][1], pos_rot + delta_cab))
            if abs(delta_cab) > 1.5:
                arduino.enviar(PCA0, 15, pos_rot, SERVO_CUELLO[15][3])

            # ── OJOS HORIZONTAL (ajuste fino) ────────────────────────
            err_x_fino = ex * 0.4
            dx = pid_x.update(err_x_fino)
            delta_h = max(-6, min(6, -dx * 0.05))
            pos_hor = max(SERVO[1][0], min(SERVO[1][1], pos_hor + delta_h))
            rh  = (pos_hor - SERVO[1][0]) / (SERVO[1][1] - SERVO[1][0])
            ph6 = SERVO[6][0] + rh * (SERVO[6][1] - SERVO[6][0])
            if abs(delta_h) > 0.5:
                arduino.enviar(PCA, 1, pos_hor, VEL_SEG)
                arduino.enviar(PCA, 6, ph6,     VEL_SEG)

            # ── OJOS VERTICAL ────────────────────────────────────────
            dy = pid_y.update(ey)
            delta_v = max(-5, min(5, dy * 0.04))
            pv0 = max(SERVO[0][0], min(SERVO[0][1], pos_ver + delta_v))
            rv  = (pv0 - SERVO[0][0]) / (SERVO[0][1] - SERVO[0][0])
            pv2 = SERVO[2][1] - rv * (SERVO[2][1] - SERVO[2][0])
            pos_ver = pv0
            if abs(delta_v) > 0.5:
                arduino.enviar(PCA, 0, pv0, VEL_SEG)
                arduino.enviar(PCA, 2, pv2, VEL_SEG)

        elif primario and f_ign:
            # ── MODO EVADIR: se aleja del rostro con velocidad intermedia ──
            bb=primario.bounding_box
            rx=int(bb.origin_x+bb.width/2)
            ex=rx-cx   # + = rostro a la derecha de la imagen

            ROT_MIN, ROT_MAX, ROT_NEU = SERVO_CUELLO[15][0], SERVO_CUELLO[15][1], SERVO_CUELLO[15][2]
            INC_MIN, INC_MAX, INC_NEU = SERVO_CUELLO[14][0], SERVO_CUELLO[14][1], SERVO_CUELLO[14][2]
            VEL_EVADIR = 7   # intermedio entre suave(4) y brusco(10)

            if abs(ex) > 50:
                # Rostro a la DERECHA -> huye a IZQ (min);  IZQUIERDA -> huye a DER (max)
                objetivo_rot = ROT_MIN if ex > 0 else ROT_MAX
                # Movimiento gradual hacia el objetivo (no salto directo)
                delta = (objetivo_rot - pos_rot) * 0.35   # 35% del camino por frame
                if abs(delta) > 1:
                    pos_rot = int(pos_rot + delta)
                    pos_rot = max(ROT_MIN, min(ROT_MAX, pos_rot))
                    arduino.enviar(PCA0, 15, pos_rot, 2)   # velocidad reducida a 2 (cuello) en evasion

                    # Ojos acompanan gradualmente al lado opuesto
                    ojo_obj  = SERVO[1][0] if ex > 0 else SERVO[1][1]
                    pos_hor  = int(pos_hor + (ojo_obj - pos_hor) * 0.35)
                    pos_hor  = max(SERVO[1][0], min(SERVO[1][1], pos_hor))
                    rh  = (pos_hor - SERVO[1][0]) / (SERVO[1][1] - SERVO[1][0])
                    ph6 = SERVO[6][0] + rh * (SERVO[6][1] - SERVO[6][0])
                    arduino.enviar(PCA, 1, pos_hor, VEL_EVADIR)
                    arduino.enviar(PCA, 6, ph6,     VEL_EVADIR)

                # Inclinacion: levanta un poco la cabeza al evadir (gesto de rechazo)
                # INC_MIN=DER  INC_MAX=IZQ  -- usamos posicion media-alta para "levantar"
                inc_objetivo = int(INC_NEU - (INC_NEU - INC_MIN) * 0.4)  # inclina hacia arriba/atras
                if not hasattr(ejecutar, "_inc_evadir") or abs(inc_objetivo - getattr(ejecutar,"_inc_actual",INC_NEU)) > 5:
                    arduino.enviar(PCA0, 14, inc_objetivo, 5)
                    ejecutar._inc_actual = inc_objetivo
            else:
                # Persona centrada o fuera -> inclinacion vuelve a neutro
                if getattr(ejecutar, "_inc_actual", INC_NEU) != INC_NEU:
                    arduino.enviar(PCA0, 14, INC_NEU, 4)
                    ejecutar._inc_actual = INC_NEU
        elif not primario or (not f_seg and not f_ign):
            pid_x.reset(); pid_y.reset(); pid_cab.reset()

        if primario and not rostro_presente:
            threading.Thread(target=expresion_saludo,args=(arduino,),daemon=True).start()
            rostro_presente=True
        if primario: t_ult_rostro=ahora

        if rostro_presente and not primario and ahora-t_ult_rostro>TIMEOUT_ROSTRO:
            threading.Thread(target=expresion_neutro,args=(arduino,),daemon=True).start()
            arduino.ir_a_centro()
            arduino.enviar(PCA0, 15, SERVO_CUELLO[15][2], SERVO_CUELLO[15][3])  # cabeza neutro
            pos_hor=SERVO[1][2]; pos_ver=SERVO[0][2]; pos_rot=SERVO_CUELLO[15][2]
            rostro_presente=False; nombre_primario=None

        # --------------------------------------------------------------
        # GESTOS DE VIDA  (idle): se disparan al azar cuando no hay nadie
        # --------------------------------------------------------------
        if primario:
            t_ult_idle=ahora                      # mientras hay gente, no hay idle
        elif not idle_activo["v"] and ahora-t_ult_idle>int_idle:
            g=random.choice(GESTOS_VIDA)
            idle_activo["v"]=True
            def _run_idle(g=g):
                try: g(arduino)
                except Exception as e: print(f" idle: {e}")
                finally: idle_activo["v"]=False
            threading.Thread(target=_run_idle,daemon=True).start()
            t_ult_idle=ahora; int_idle=random.uniform(7.0,14.0)

        # -- Dibujar todos los rostros ----------------------------------
        for i,det in enumerate(detecciones):
            bb=det.bounding_box
            x1,y1=bb.origin_x,bb.origin_y
            x2,y2=x1+bb.width,y1+bb.height
            es_primario=(i==0)
            key=f"{int(x1//40)}_{int(y1//40)}"  # clave estable por posicin

            # Color del marco
            col_box=(0,255,0) if es_primario else (180,180,180)
            grosor=2 if es_primario else 1
            cv2.rectangle(frame,(x1,y1),(x2,y2),col_box,grosor)

            # --------------------------------------------------
            # RECONOCIMIENTO FACIAL CON MEDIAPIPE + LBPH
            # --------------------------------------------------

            nombre_disp = "Desconocido"
            conf_rec = 0.0
            col_nom = (0, 0, 200)

            if (
                f_rec
                and hay_personas
                and reconocedor is not None
                and frame_count % RECONOCER_CADA == 0
                and es_primario
            ):

                try:

                    # Extraer el rostro detectado por MediaPipe
                    rostro_img = extraer_rostro(
                        frame,
                        (x1, y1, x2 - x1, y2 - y1)
                    )

                    if rostro_img is not None:

                        # Reconocer con LBPH
                        etiqueta, distancia = reconocedor.predict(
                            rostro_img
                        )

                        nn = nombres_por_id.get(
                            int(etiqueta),
                            "Desconocido"
                        )

                        # Menor distancia = mejor coincidencia
                        if distancia < TOLERANCIA_REC:

                            # Convertimos la distancia en una confianza
                            # aproximada para conservar el funcionamiento
                            # del resto de la interfaz.
                            cf = max(
                                0.0,
                                min(
                                    1.0,
                                    1.0 - distancia / 100.0
                                )
                            )

                            nombre_cache[key] = (
                                nn,
                                cf,
                                ahora
                            )

                            if nn != nombre_primario:

                                nombre_primario = nn

                                threading.Thread(
                                    target=expresion_reconocido,
                                    args=(arduino,),
                                    daemon=True
                                ).start()

                                print(
                                    f" Reconocido: {nn} "
                                    f"(distancia {distancia:.1f})"
                                )

                        else:

                            nombre_cache.pop(
                                key,
                                None
                            )

                            if es_primario:
                                nombre_primario = None

                except Exception as e:

                    print(
                        f" Rec: {e}"
                    )

            # Usar cache de reconocimiento
            if key in nombre_cache:
                nn,cf,ts=nombre_cache[key]
                if ahora-ts<5.0:
                    nombre_disp=f"{nn} ({cf:.0%})"; conf_rec=cf; col_nom=(0,200,0)

            #  Emocion 
            emo_disp=None; col_emo=(180,180,180)
            if f_emo and df_estado["ok"]:
                with emo_lock:
                    if key in emo_cache:
                        ee,ec,et=emo_cache[key]
                        if ahora-et<3.0:
                            emo_disp=f"{ee} {ec:.0%}"; col_emo=COLOR_EMOCION.get(ee,(180,180,180))
                # Lanzar anlisis cada 2s por persona
                if key not in analizando:
                    with emo_lock: ts_e=emo_cache[key][2] if key in emo_cache else 0
                    if ahora-ts_e>2.0:
                        with emo_lock: analizando.add(key)
                        threading.Thread(target=_analizar_emo,
                                         args=(key,rgb,x1,y1,x2,y2),daemon=True).start()

            #  Etiqueta sobre el cuadro 
            lh=22  # altura lnea
            etiquetas=[(nombre_disp,col_nom)]
            if emo_disp: etiquetas.append((emo_disp,col_emo))

            ly=y1-len(etiquetas)*lh-6
            if ly<0: ly=y2+6
            for txt,col in etiquetas:
                tw=len(txt)*10+8
                cv2.rectangle(frame,(x1,ly),(x1+tw,ly+lh),col,-1)
                put(frame,txt,x1+4,ly+15,0.5,(255,255,255),1)
                ly+=lh

            # Indicador primario
            if es_primario:
                cv2.circle(frame,(x1+12,y1+12),8,(0,255,0),-1)
                put(frame,"1",(x1+7),(y1+17),0.45,(0,0,0),1)

        # 
        # GESTOS  mano ms cercana al centro de imagen
        # 
        gesto_frame=None
        if f_ges:
            res_mano=det_mano.detect(mp_img)
            if res_mano.hand_landmarks:
                # Elegir la mano cuyo centroide est ms cerca del centro de imagen
                mejor_mano=None; mejor_dist=float("inf")
                for lms in res_mano.hand_landmarks:
                    mx_=np.mean([l.x for l in lms])*W
                    my_=np.mean([l.y for l in lms])*H
                    d=math.sqrt((mx_-cx)**2+(my_-cy)**2)
                    if d<mejor_dist: mejor_dist=d; mejor_mano=lms
                # Dibujar TODAS las manos detectadas con esqueleto completo
                for idx_m, lms in enumerate(res_mano.hand_landmarks):
                    # Mano primaria (mas cercana) en verde, otras en gris
                    es_primaria = (lms is mejor_mano)
                    col_l = (0,255,128) if es_primaria else (180,180,180)
                    col_p = (0,0,255)   if es_primaria else (120,120,120)
                    bbox_mano = dibujar_mano(frame, lms, W, H, col_l, col_p)

                    # Etiqueta Left/Right + dedos levantados
                    handedness = ""
                    if res_mano.handedness and idx_m < len(res_mano.handedness):
                        handedness = res_mano.handedness[idx_m][0].category_name
                    dedos = dedos_arriba(lms)
                    n_dedos = sum(dedos)
                    etiqueta = f"{handedness} [{n_dedos}]"
                    put(frame, etiqueta, bbox_mano[0], bbox_mano[1]-8, 0.6,
                        (255,0,255), 2)

                if mejor_mano:
                    gesto_frame=clasificar_gesto(mejor_mano)

        if gesto_frame and gesto_frame==g_cand: g_frames+=1
        else: g_cand=gesto_frame; g_frames=1

        if g_frames>=FRAMES_CONF and g_cand:
            t_ult=ult_g_t.get(g_cand,0)
            if ahora-t_ult>=COOLDOWN_GESTO and g_cand in GESTO_ACCION:
                jf,et=GESTO_ACCION[g_cand]
                print(f" {g_cand}  {et}")
                rep.reproducir(jf)
                if g_cand in GESTOS_HABLA:
                    hablar(arduino, ciclos=5)   # mueve la boca en gestos vocales
                ult_g_t[g_cand]=ahora; ult_g_act=(g_cand,et,ahora); g_frames=0

        # --------------------------------------------------------------
        # HUD  panel superior
        # --------------------------------------------------------------
        ph=52  # altura panel
        panel(frame,0,0,W,ph)

        # Personas
        col_n=(0,220,0) if n_personas>0 else (120,120,120)
        put(frame,f"Personas: {n_personas}",12,20,0.65,col_n,2)
        # Indicador de posicion de cabeza
        rot_pct = int((pos_rot - SERVO_CUELLO[15][0]) / (SERVO_CUELLO[15][1] - SERVO_CUELLO[15][0]) * 100)
        col_rot = (0,200,255) if abs(rot_pct-56)>10 else (180,180,180)  # 56% = neutro aprox
        put(frame,f"Cabeza: {rot_pct}% ({'IZQ' if rot_pct<50 else 'DER'})",12,44,0.45,col_rot,1)

        # Nombre primario
        if nombre_primario:
            put(frame,f"Siguiendo: {nombre_primario}",12,42,0.55,(0,200,255),1)

        # Estados de mdulos centrados
        estados=[
            (f"[S] Seguim {'ON' if f_seg else 'OFF'}",  f_seg),
            (f"[I] Evadir {'ON' if f_ign else 'OFF'}",  f_ign),
            (f"[R] Reconoc {'ON' if f_rec else 'OFF'}", f_rec),
            (f"[G] Gestos {'ON' if f_ges else 'OFF'}",  f_ges),
            (f"[E] Emocion {'ON' if f_emo else 'OFF'}", f_emo),
        ]
        sx=W//2-280
        for txt,act in estados:
            col=(0,220,0) if act else (100,100,100)
            put(frame,txt,sx,32,0.5,col,1); sx+=130

        # DeepFace status
        emo_status = (
            "DeepFace: OK"
            if df_estado["ok"]
            else (
            "DeepFace: cargando..."
            if not df_estado["intentado"]
            else "DeepFace: error"
            )
        )

        col_df = (
            (0, 200, 100)
            if df_estado["ok"]
            else (
                (0, 200, 255)
                if not df_estado["intentado"]
                else (0, 80, 220)
            )
        )
        put(frame,emo_status,W-200,20,0.45,col_df,1)
        put(frame,"Q=salir",W-100,42,0.45,(150,150,150),1)

        #  Panel inferior gestos 
        pb=30
        panel(frame,0,H-pb,W,pb)

        # Barra de progreso del gesto actual
        if g_cand:
            col_g=COLORES_G.get(g_cand,(200,200,200))
            barra_conf(frame,12,H-pb+11,140,min(g_frames/FRAMES_CONF,1),col_g,h=10)
            put(frame,g_cand.replace("_"," "),160,H-pb+20,0.5,col_g,1)

        # ltima accin
        if ult_g_act:
            gn,et,ta=ult_g_act
            el=ahora-ta
            if el<4.0:
                a=max(0,1-el/4.0)
                col_a=tuple(int(c*a) for c in COLORES_G.get(gn,(200,200,200)))
                put(frame,f">> {et}",W//2-60,H-pb+20,0.65,col_a,2)

        # Gua gestos (derecha panel inferior)
        guia=[("Abierta:Alegria","mano_abierta"),("Puno:Enojo","puno"),
              ("PulgarUp:Sorpresa","pulgar_arriba"),("PulgarDn:Tristeza","pulgar_abajo"),
              ("Indice:Habla","indice"),("V:Guino","dos_dedos"),
              ("3Ded:Parpadeo","tres_dedos"),("4Ded:Barrido","cuatro_dedos"),
              ("OK:Risa","ok")]
        gx=W-520
        for label,gnom in guia:
            col_g=COLORES_G.get(gnom,(160,160,160))
            put(frame,label,gx,H-pb+20,0.38,col_g,1); gx+=58

        # Crosshair
        cv2.line(frame,(cx-20,cy),(cx+20,cy),(255,255,255),1)
        cv2.line(frame,(cx,cy-20),(cx,cy+20),(255,255,255),1)

        cv2.imshow("InMoov Vision",frame)
        key=cv2.waitKey(1)&0xFF
        if   key==ord('q'): break
        elif key==ord('s'):
            f_seg=not f_seg
            if f_seg: f_ign=False   # evasion y seguimiento son excluyentes
            print(f"Seguim: {'ON' if f_seg else 'OFF'}  Evadir: {'ON' if f_ign else 'OFF'}")
        elif key==ord('i'):
            f_ign=not f_ign
            if f_ign: f_seg=False   # excluyente con seguimiento
            print(f"Evadir: {'ON' if f_ign else 'OFF'}  Seguim: {'ON' if f_seg else 'OFF'}")
        elif key==ord('r'):
            if hay_personas: f_rec=not f_rec; print(f"Reconoc: {'ON' if f_rec else 'OFF'}")
            else: print("  Sin personas registradas")
        elif key==ord('g'): f_ges=not f_ges;  print(f"Gestos: {'ON' if f_ges else 'OFF'}")
        elif key==ord('e'): f_emo=not f_emo;  print(f"Emocion: {'ON' if f_emo else 'OFF'}")

    print("\n Cerrando")
    arduino.ir_a_centro(); cap.release()
    cv2.destroyAllWindows(); det_cara.close(); det_mano.close()

# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser(description="InMoov Vision v2")
    ap.add_argument("--registrar",type=str,default=None,metavar="NOMBRE")
    ap.add_argument("--listar",   action="store_true")
    ap.add_argument("--borrar",   type=str,default=None,metavar="NOMBRE")
    ap.add_argument("--cam",      type=int,default=0)
    ap.add_argument("--port",     type=str,default=None)
    ap.add_argument("--demo",     action="store_true")
    args=ap.parse_args()

    if args.listar:
        listar_personas(cargar_db())
    elif args.borrar:
        db=cargar_db()
        idx=[i for i,n in enumerate(db["nombres"]) if n==args.borrar]
        if not idx: print(f"  '{args.borrar}' no encontrado.")
        else:
            keep=set(range(len(db["nombres"])))-set(idx)
            db["nombres"]  =[db["nombres"][i]   for i in sorted(keep)]
            db["encodings"]=[db["encodings"][i] for i in sorted(keep)]
            guardar_db(db); print(f"  '{args.borrar}' eliminado ({len(idx)} muestras).")
        listar_personas(db)
    elif args.registrar:
        registrar_persona(args.registrar,cam_idx=args.cam)
    else:
        try:
            ejecutar(cam_idx=args.cam,port=args.port,demo=args.demo)
        except Exception as e:
            import traceback; traceback.print_exc()
            input("\n Error  ENTER para cerrar...")
