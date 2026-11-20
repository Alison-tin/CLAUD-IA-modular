import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import random
import math
import threading
from configuracion.config import *
from hardware.arduino import Arduino
from hardware.camara import abrir_camara
from datos.personas import cargar_db, listar_personas, extraer_rostro, entrenar_reconocedor
from movimientos.expresiones import hablar, expresion_saludo, expresion_neutro, expresion_reconocido
from movimientos.gestos import GESTOS_VIDA
from secuencias.reproductor import Reproductor, resolver_json, EmocionesMovimiento
from vision.manos import dibujar_mano, dedos_arriba, clasificar_gesto, GESTO_ACCION, COLORES_G
from vision.emociones import _cargar_deepface, analizar_emocion, deepface_estado
from vision.modelos import descargar_modelos
from vision.fotogramas import imagen_mediapipe, validar_bgr
from vision.reconocimiento import decidir_reconocimiento, sin_lectura, distancias_por_identidad
from vision.motor_sface import MotorSFace, sin_resultado
from vision.galeria_sface import GaleriaInvalida
from control.pid import PID
from interfaz.pantalla import put, panel, barra_conf

def ejecutar(cam_idx=0, port=None, demo=False, comparar_rostro=False, interfaz=None, persona_comparacion="Alexa", motor="lbph", emociones_externas=None):
    recursos = {}
    try:
        return _ejecutar(cam_idx, port, demo, comparar_rostro, interfaz, recursos, persona_comparacion, motor, emociones_externas)
    finally:
        if 'arduino' in recursos:
            recursos['arduino'].cancelado = lambda: True
        if "reproductor" in recursos:
            recursos["reproductor"].detener()
        for nombre, metodo in (("cap", "release"), ("cara", "close"),
                               ("mano", "close"), ("serial", "close")):
            recurso = recursos.get(nombre)
            if recurso is not None:
                try:
                    getattr(recurso, metodo)()
                except Exception:
                    pass
        if interfaz is None:
            cv2.destroyAllWindows()
        if emociones_externas is not None:
            emociones_externas.cerrar()


def _ejecutar(cam_idx, port, demo, comparar_rostro, interfaz, recursos, persona_comparacion="Alexa", motor="lbph", emociones_externas=None):
    if motor not in ("lbph", "sface"):
        raise ValueError("Motor de reconocimiento invalido")
    if comparar_rostro and motor != "lbph":
        raise ValueError("--comparar-rostro es diagnostico exclusivo de LBPH")
    if comparar_rostro and not demo:
        raise ValueError("--comparar-rostro requiere --demo")
    descargar_modelos()
    if emociones_externas is None:
        threading.Thread(target=_cargar_deepface, daemon=True).start()
    df_estado = deepface_estado() if emociones_externas is None else emociones_externas.estado()
    db = cargar_db()

    hay_personas = len(db["imagenes"]) > 0
    muestra_debug = {} if comparar_rostro else None

    sf_motor = None
    if motor == "sface":
        sf_motor = MotorSFace()  # Valida antes de abrir camara o Arduino; sin fallback.
        reconocedor, nombres_por_id = sf_motor, {}
        print(f" SFace listo; galeria {sf_motor.version}")
    elif hay_personas:
        listar_personas(db)
        if comparar_rostro:
            reconocedor, nombres_por_id = entrenar_reconocedor(
                muestra_debug, persona_comparacion=persona_comparacion)
        else:
            reconocedor, nombres_por_id = entrenar_reconocedor(muestra_debug)

        if reconocedor is not None:
            print(" Reconocimiento facial listo")
        else:
            print(" No se pudo crear el reconocedor")
    else:
        reconocedor = None
        nombres_por_id = {}
        print("  Sin personas registradas  solo deteccion")

    if interfaz is not None:
        interfaz.emitir("motor_reconocimiento", {"motor": motor,
                         "version": sf_motor.version if sf_motor else None})

    if comparar_rostro:
        if "imagen" in muestra_debug:
            print(f" Comparacion LBPH: muestra real de entrenamiento: {muestra_debug['ruta']}")
            print(f" Comparacion LBPH: izquierda={muestra_debug['nombre']} entrenada; derecha=entrada en vivo. Q=salir.")
        else:
            raise ValueError(f"No se encontro una muestra valida de {persona_comparacion} para comparar")

    if interfaz is not None and interfaz.detener():
        return
    if not demo and interfaz is None:
        print("Cabeza fisica: se centraran servos, se abriran parpados y se probara el maxilar.")
        print("Los gestos activaran secuencias propuestas pendientes de validacion fisica.")
        if input("Escribe INICIAR para continuar: ").strip() != "INICIAR":
            return
    arduino=Arduino(port=port,demo=demo)
    recursos['arduino'] = arduino
    if interfaz is not None:
        arduino.cancelado = interfaz.detener
    recursos["serial"] = arduino.ser
    if interfaz is not None and hasattr(interfaz, "preparar_arduino"):
        interfaz.preparar_arduino(arduino, demo)
        if interfaz.detener():
            return
    if interfaz is not None:
        interfaz.emitir("conexion", "Demo: sin envio de ordenes" if arduino.demo
                        else f"Cabeza fisica: puerto {arduino.ser.port}")
    rep=Reproductor(arduino, lambda mensaje: interfaz.emitir("aviso_secuencia", mensaje) if interfaz else None)
    recursos["reproductor"] = rep
    movimientos_emocion = EmocionesMovimiento(rep)
    time.sleep(0.5); arduino.ir_a_centro(); arduino.parpado(abrir=True)
    # Arranque fisico: abre y cierra el maxilar (PCA0, canal 12).
    arduino.mover_maxilar(abrir=True, vel=5);  time.sleep(0.5)
    arduino.mover_maxilar(abrir=False, vel=5)
    print(" Test maxilar enviado (PCA0 canal 12: abre->cierra)")
    print(" Posicin inicial enviada")

    
    det_cara=mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELO_CARA_PATH),
            min_detection_confidence=0.55))

    recursos["cara"] = det_cara
    det_mano=mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELO_MANO_PATH),
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5))

    recursos["mano"] = det_mano
    cap = abrir_camara(cam_idx)
    recursos["cap"] = cap

    
    rw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_real=cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolucion real: {rw}x{rh} @ {fps_real:.0f}fps")
    if not cap.isOpened():
        raise RuntimeError(f"Camara {cam_idx} no disponible")
    if interfaz is not None:
        interfaz.emitir("camara", f"Activa: {rw} x {rh}")

    
    if interfaz is None:
        cv2.namedWindow("InMoov Vision", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("InMoov Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" Frame HUD: {W}x{H}")
    cx,cy=W//2,H//2
    print(f"Camara {cam_idx}: {W}x{H}")
    print("Activo | S=Seguim  I=Evadir  R=Reconoc  G=Gestos  E=Emocion  Q=salir\n")

    
    f_seg=True; f_rec=hay_personas; f_ges=True; f_emo=True; f_ign=False

    # Estado seguimiento
    pid_x=PID(Kp,Ki,Kd,ZONA_MUERTA_X); pid_y=PID(Kp,Ki,Kd,ZONA_MUERTA_Y)
    # PID separado para rotacion de cabeza (movimientos grandes)
    pid_cab=PID(0.15, 0.001, 0.05, ZONA_MUERTA_X*2)  # mas lento y suave que ojos
    pos_hor=SERVO[1][2]; pos_ver=SERVO[0][2]
    pos_rot=SERVO_CUELLO[15][2]   # posicion actual rotacion cabeza (neutro)
    rostro_presente=False; t_ult_rostro=0.0

    # Estado reconocimiento
    nombre_primario=None
    frame_count=0
    resolucion_informada = False

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

    def _parpado(a):
        a.parpado(abrir=False,vel=2); a.esperar(0.10)
        a.parpado(abrir=True,vel=2)

    def _analizar_emo(key, roi_rgb, x1,y1,x2,y2):
        emo,conf=analizar_emocion(roi_rgb,x1,y1,x2,y2)
        with emo_lock:
            if emo: emo_cache[key]=(emo,conf,time.time())
            analizando.discard(key)
        if interfaz is not None and emo:
            interfaz.emitir("expresion", {"nombre": emo, "confianza": float(conf)})

    while True:
        if interfaz is not None:
            if hasattr(interfaz, "comprobar_arduino"):
                interfaz.comprobar_arduino()
            if interfaz.detener():
                break
            opciones = interfaz.opciones()
            f_rec = opciones.get("reconocimiento", f_rec) and hay_personas and reconocedor is not None
            f_ges = opciones.get("gestos", f_ges)
            f_emo = opciones.get("emociones", f_emo)
            if not f_rec:
                nombre_primario = None
        if not f_emo:movimientos_emocion.actualizar(None, activo=False)
        respuesta_emocion = None
        lectura_actual = (sin_resultado("sin_rostro" if f_rec else "desactivado")
                          if sf_motor else sin_lectura())
        panel_debug = None
        df_estado = deepface_estado() if emociones_externas is None else emociones_externas.estado()
        ret,frame=cap.read()
        if not ret:
            if interfaz is not None:
                interfaz.emitir("error", "La camara dejo de entregar imagenes.")
            break
        try:
            validar_bgr(frame)
        except ValueError as error:
            movimientos_emocion.actualizar(None)
            print(f" Fotograma descartado: {error}")
            if interfaz is not None:
                interfaz.emitir("reconocimiento", lectura_actual)
            if emociones_externas is not None:
                emociones_externas.actualizar(None, activo=f_emo)
                if interfaz is not None:
                    interfaz.emitir("expresion", {"nombre": None, "confianza": 0.})
            continue
        H, W = frame.shape[:2]
        cx, cy = W//2, H//2
        if not resolucion_informada:
            print(f" Fotograma de reconocimiento: {frame.shape[1]}x{frame.shape[0]}")
            resolucion_informada = True
        frame_limpio = frame.copy()

        ahora=time.time(); frame_count+=1
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if comparar_rostro and frame_count == 1:
            print(f" MediaPipe BGR shape={frame.shape} dtype={frame.dtype} strides={frame.strides} C={frame.flags.c_contiguous}; RGB strides={rgb.strides}")

        # Parpadeo automatico
        if ahora-t_ult_parp>int_parp and not idle_activo["v"]:
            rep.accion('parpadeo', _parpado)
            t_ult_parp=ahora; int_parp=random.uniform(*PARPADEO_INT)

        
        # DETECCIN DE ROSTROS (todos)
        
        res_cara=det_cara.detect(imagen_mediapipe(frame_limpio))
        detecciones=res_cara.detections if res_cara.detections else []

        # Ordenar por rea (ms grande = ms cercano) descendente
        def area(d): b=d.bounding_box; return b.width*b.height
        detecciones=sorted(detecciones,key=area,reverse=True)

        n_personas=len(detecciones)

        # Primario = el ms grande
        primario=detecciones[0] if detecciones else None

        # Seguimiento (suave) o Evasion (brusca)
        if primario and f_seg:
            bb=primario.bounding_box
            rx=int(bb.origin_x+bb.width/2); ry=int(bb.origin_y+bb.height/2)
            ex=rx-cx; ey=ry-cy

            # ROTACION DE CABEZA (movimientos grandes)
            delta_cab = max(-5, min(5, pid_cab.update(ex) * 0.05))   # mas lento
            pos_rot = max(SERVO_CUELLO[15][0], min(SERVO_CUELLO[15][1], pos_rot + delta_cab))
            if abs(delta_cab) > 1.5:
                arduino.enviar(PCA0, 15, pos_rot, SERVO_CUELLO[15][3])

            # OJOS HORIZONTAL (ajuste fino)
            err_x_fino = ex * 0.4
            dx = pid_x.update(err_x_fino)
            delta_h = max(-6, min(6, -dx * 0.05))
            pos_hor = max(SERVO[1][0], min(SERVO[1][1], pos_hor + delta_h))
            rh  = (pos_hor - SERVO[1][0]) / (SERVO[1][1] - SERVO[1][0])
            ph6 = SERVO[6][0] + rh * (SERVO[6][1] - SERVO[6][0])
            if abs(delta_h) > 0.5:
                arduino.enviar(PCA, 1, pos_hor, VEL_SEG)
                arduino.enviar(PCA, 6, ph6,     VEL_SEG)

            # OJOS VERTICAL
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
            # MODO EVADIR: se aleja del rostro con velocidad intermedia
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
            rep.accion('saludo', expresion_saludo)
            rostro_presente=True
        if primario: t_ult_rostro=ahora

        if rostro_presente and not primario and ahora-t_ult_rostro>TIMEOUT_ROSTRO:
            rep.accion('neutro', expresion_neutro)
            arduino.ir_a_centro()
            arduino.enviar(PCA0, 15, SERVO_CUELLO[15][2], SERVO_CUELLO[15][3])  # cabeza neutro
            pos_hor=SERVO[1][2]; pos_ver=SERVO[0][2]; pos_rot=SERVO_CUELLO[15][2]
            rostro_presente=False; nombre_primario=None

        
        # GESTOS DE VIDA  (idle): se disparan al azar cuando no hay nadie
        
        if primario:
            t_ult_idle=ahora                      # mientras hay gente, no hay idle
        elif not idle_activo["v"] and ahora-t_ult_idle>int_idle:
            g=random.choice(GESTOS_VIDA)
            idle_activo["v"]=True
            def _run_idle(a, g=g):
                try: g(a)
                except Exception as e: print(f" idle: {e}")
                finally: idle_activo["v"]=False
            if not rep.accion('vida', _run_idle):idle_activo['v']=False
            t_ult_idle=ahora; int_idle=random.uniform(7.0,14.0)

        # Dibujar todos los rostros
        for i,det in enumerate(detecciones):
            bb=det.bounding_box
            x1,y1=bb.origin_x,bb.origin_y
            x2,y2=x1+bb.width,y1+bb.height
            es_primario=(i==0)
            key=f"{int(x1//40)}_{int(y1//40)}"  # clave estable por posicin

            
            col_box=(0,255,0) if es_primario else (180,180,180)
            grosor=2 if es_primario else 1
            cv2.rectangle(frame,(x1,y1),(x2,y2),col_box,grosor)

            
            # RECONOCIMIENTO FACIAL CON MEDIAPIPE + LBPH
            

            nombre_disp = "Sin lectura"
            conf_rec = 0.0
            col_nom = (0, 0, 200)

            if (
                f_rec
                and motor == "lbph"
                and hay_personas
                and reconocedor is not None
                and es_primario
            ):

                try:

                    
                    caja_rostro = (x1, y1, x2 - x1, y2 - y1)
                    if comparar_rostro:
                        diagnostico_recorte = {}
                        rostro_img = extraer_rostro(
                            frame_limpio, caja_rostro, diagnostico=diagnostico_recorte)
                    else:
                        rostro_img = extraer_rostro(frame_limpio, caja_rostro)

                    if rostro_img is not None:

                        
                        etiqueta, distancia = reconocedor.predict(
                            rostro_img
                        )

                        if comparar_rostro:
                            distancias_debug = distancias_por_identidad(
                                reconocedor, rostro_img, nombres_por_id)
                            print(f" [DIAG LBPH] frame={frame_count} caja={caja_rostro} "
                                  f"{diagnostico_recorte} distancias={distancias_debug}")

                        if comparar_rostro and "imagen" in muestra_debug:
                            # Solo el panel de visualizacion recibe textos.
                            panel_debug = np.zeros((270, 440), dtype=np.uint8)
                            panel_debug[35:235, 10:210] = muestra_debug["imagen"]
                            panel_debug[35:235, 230:430] = rostro_img
                            for texto, posicion in (
                                (f"{muestra_debug['nombre']}: entrenamiento", (10, 22)),
                                ("En vivo: entrada LBPH", (230, 22)),
                                (f"{nombres_por_id.get(int(etiqueta))}: {distancia:.2f} / {TOLERANCIA_REC:.0f}", (10, 258)),
                            ):
                                cv2.putText(panel_debug, texto, posicion,
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)

                        nn = nombres_por_id.get(
                            int(etiqueta),
                            "Desconocido"
                        )
                        lectura_actual = decidir_reconocimiento(
                            etiqueta, distancia, nombres_por_id, TOLERANCIA_REC)
                        
                        print(
                            f" [LBPH] candidato={nn} | "
                            f"distancia={distancia:.2f} | "
                            f"umbral={TOLERANCIA_REC:.2f} | "
                            f"resultado={lectura_actual['nombre']}"
                        )

                        # Menor distancia = mejor coincidencia
                        nombre_disp = lectura_actual["nombre"]
                        if lectura_actual["similitud_estimada"] is not None:

                            # Confianza aproximada derivada de la distancia LBPH.
                            
                            
                            cf = lectura_actual["similitud_estimada"]

                            nombre_disp = f"{nn} ({cf:.0%})"
                            conf_rec = cf
                            col_nom = (0, 200, 0)

                            if nn != nombre_primario:

                                nombre_primario = nn

                                rep.accion('reconocido', expresion_reconocido)

                                print(
                                    f" Reconocido: {nn} "
                                    f"(distancia {distancia:.1f})"
                                )

                        else:

                            if es_primario:
                                nombre_primario = None

                except Exception as e:

                    print(
                        f" Rec: {e}"
                    )

            
            if sf_motor is not None and f_rec and es_primario:
                try:
                    cajas = [(d.bounding_box.origin_x, d.bounding_box.origin_y,
                              d.bounding_box.width, d.bounding_box.height) for d in detecciones]
                    if os.environ.get('CLAUD_IA_DIAGNOSTICO_SFACE') == '1':
                        lectura_actual = sf_motor.reconocer(frame_limpio, cajas, fotograma_id=frame_count)
                    else:
                        lectura_actual = sf_motor.reconocer(frame_limpio, cajas)
                    nombre_disp = lectura_actual["nombre"] or "Sin lectura"
                    if lectura_actual["similitud_coseno"] is not None:
                        nn = lectura_actual["nombre"]
                        nombre_disp = f"{nn} - Similitud {lectura_actual['similitud_coseno']*100:.1f} %".replace(".", ",")
                        col_nom = (0, 200, 0)
                        if nn != nombre_primario:
                            nombre_primario = nn
                            rep.accion('reconocido', expresion_reconocido)
                    else:
                        nombre_primario = None
                    if os.environ.get('CLAUD_IA_DIAGNOSTICO_SFACE') == '1':
                        print(f" [SFace] resultado={lectura_actual['nombre']} "
                              f"coseno={lectura_actual['similitud_coseno']} estado={lectura_actual['estado_reconocimiento']}")
                except GaleriaInvalida:
                    raise
                except Exception as error:
                    lectura_actual = sin_resultado("error", str(error))
                    nombre_disp = "Sin lectura"
                    print(f" SFace: {error}")

            #  Emocion 
            emo_disp=None; col_emo=(180,180,180)
            if emociones_externas is not None and es_primario and len(detecciones) == 1:
                # El proceso externo recibe únicamente una copia BGR sin anotaciones.
                roi = frame_limpio[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                respuesta = emociones_externas.actualizar(roi if roi.size else None,
                    activo=f_emo, objetivo=lectura_actual.get("nombre"))
                if interfaz is not None:
                    interfaz.emitir("expresion", {"nombre": respuesta["nombre"] if respuesta else None,
                        "confianza": respuesta["confianza"] if respuesta else 0.})
                respuesta_emocion = respuesta
                if respuesta:
                    emo_disp = respuesta['nombre']
            elif emociones_externas is None and f_emo and df_estado["ok"]:
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
                                         args=(key,rgb.copy(),x1,y1,x2,y2),daemon=True).start()

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

        
        if emociones_externas is not None:
            if len(detecciones) != 1:
                emociones_externas.actualizar(None, activo=f_emo)
                if interfaz is not None:
                    interfaz.emitir("expresion", {"nombre": None, "confianza": 0.})
            df_estado = emociones_externas.estado()

        if sf_motor is not None and (not f_rec or not detecciones):
            sf_motor.registrar_sin_comparacion(frame_count, 'desactivado' if not f_rec else 'sin_rostro')

        if comparar_rostro:
            if panel_debug is None:
                panel_debug = np.zeros((270, 440), dtype=np.uint8)
                panel_debug[35:235, 10:210] = muestra_debug["imagen"]
                cv2.putText(panel_debug, "Sin entrada LBPH en este fotograma", (10, 258),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
            cv2.imshow("Comparacion temporal LBPH", panel_debug)

        # GESTOS  mano ms cercana al centro de imagen
        
        gesto_frame=None
        if f_ges:
            res_mano=det_mano.detect(imagen_mediapipe(frame_limpio))
            if res_mano.hand_landmarks:
                # Elegir la mano cuyo centroide est ms cerca del centro de imagen
                mejor_mano=None; mejor_dist=float("inf")
                for lms in res_mano.hand_landmarks:
                    mx_=np.mean([l.x for l in lms])*W
                    my_=np.mean([l.y for l in lms])*H
                    d=math.sqrt((mx_-cx)**2+(my_-cy)**2)
                    if d<mejor_dist: mejor_dist=d; mejor_mano=lms
                
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

        if interfaz is not None:
            interfaz.emitir("reconocimiento", lectura_actual)
            interfaz.emitir("deteccion", {"rostros": n_personas, "gesto": gesto_frame,
                                         "deepface": dict(df_estado)})

        if gesto_frame and gesto_frame==g_cand: g_frames+=1
        else: g_cand=gesto_frame; g_frames=1

        if g_frames>=FRAMES_CONF and g_cand:
            t_ult=ult_g_t.get(g_cand,0)
            if ahora-t_ult>=COOLDOWN_GESTO and g_cand in GESTO_ACCION:
                jf,et=GESTO_ACCION[g_cand]
                iniciada = rep.reproducir(jf)
                if interfaz is not None:
                    interfaz.emitir("secuencia", {"gesto": g_cand, "archivo": jf,
                                                 "disponible": resolver_json(jf) is not None,
                                                 "iniciada": iniciada, "propuesta": True,
                                                 "error": rep.error})
                
                if iniciada:
                    print(f" {et} iniciada")
                # La boca pertenece al JSON; no ejecutar hablar() adicional.
                ult_g_t[g_cand]=ahora; ult_g_act=(g_cand,et,ahora); g_frames=0

        
        movimientos_emocion.actualizar(respuesta_emocion, activo=f_emo and n_personas == 1)

        # HUD  panel superior
        
        ph=52  # altura panel
        panel(frame,0,0,W,ph)

        # Personas
        col_n=(0,220,0) if n_personas>0 else (120,120,120)
        put(frame,f"Personas: {n_personas}",12,20,0.65,col_n,2)
        # Indicador de posicion de cabeza
        rot_pct = int((pos_rot - SERVO_CUELLO[15][0]) / (SERVO_CUELLO[15][1] - SERVO_CUELLO[15][0]) * 100)
        col_rot = (0,200,255) if abs(rot_pct-56)>10 else (180,180,180)  # 56% = neutro aprox
        put(frame,f"Cabeza: {rot_pct}% ({'IZQ' if rot_pct<50 else 'DER'})",12,44,0.45,col_rot,1)

        # Mostrar solo identidad aceptada en ESTE fotograma.
        if (lectura_actual["similitud_estimada"] is not None
                or lectura_actual.get("similitud_coseno") is not None):
            put(frame,f"Siguiendo: {lectura_actual['nombre']}",12,42,0.55,(0,200,255),1)

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

        
        cv2.line(frame,(cx-20,cy),(cx+20,cy),(255,255,255),1)
        cv2.line(frame,(cx,cy-20),(cx,cy+20),(255,255,255),1)

        if interfaz is None:
            cv2.imshow("InMoov Vision",frame)
            key=cv2.waitKey(1)&0xFF
        else:
            interfaz.fotograma(frame_limpio if getattr(interfaz, "video_limpio", False) else frame)
            key = -1
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
    rep.detener()
    arduino.ir_a_centro()

