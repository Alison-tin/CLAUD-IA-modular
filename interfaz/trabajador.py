"""Trabajador compartido por web y consola, sin interfaz de escritorio."""
from queue import Empty, Full
import time


class Puente:
    def __init__(self, eventos, cuadros, comandos, detener):
        self.eventos, self.cuadros = eventos, cuadros
        self.comandos, self.parada = comandos, detener
        self.ajustes = {}
        self.ultimo_cuadro = 0
        self.ultimos = {}

    def detener(self):
        return self.parada.is_set()

    def opciones(self):
        while True:
            try:
                tipo, valor = self.comandos.get_nowait()
            except Empty:
                break
            if tipo == "opciones":
                self.ajustes.update(valor)
        return self.ajustes

    def preparar_arduino(self, arduino, demo):
        if getattr(self, "video_limpio", False):
            from interfaz.conexion import preparar_conexion
            preparar_conexion(self, arduino, demo)

    def comprobar_arduino(self):
        serial = getattr(self, "serial_observado", None)
        if serial is not None:
            return serial.is_open

    def emitir(self, tipo, valor):
        if tipo != 'resultado_registro' and self.ultimos.get(tipo) == valor:
            return True
        self.ultimos[tipo] = valor
        try:
            self.eventos.put_nowait((tipo, valor))
        except Full:
            self.ultimos.pop(tipo, None)
            if tipo == 'resultado_registro':
                try:
                    self.eventos.put((tipo,valor),timeout=2)
                    self.ultimos[tipo]=valor
                    return True
                except Full:
                    print('Error: no se pudo entregar el resultado del registro a la web')
            return False
        return True

    def fotograma(self, frame):
        if time.monotonic() - self.ultimo_cuadro < 0.065:
            return
        import cv2
        alto, ancho = frame.shape[:2]
        escala = min(960 / ancho, 640 / alto, 1)
        vista = cv2.resize(frame, (int(ancho*escala), int(alto*escala)))
        ok, jpeg = cv2.imencode(".jpg", vista)
        if ok:
            try:
                self.cuadros.put_nowait(jpeg.tobytes())
            except Full:
                pass
        self.ultimo_cuadro = time.monotonic()


def registrar(puente, nombre, cam_idx, prueba=False, evaluacion=None, alta_sface=False, registro_id=None):
    import cv2
    from configuracion.config import DB_PATH, FOTOS_DIR
    from hardware.camara import abrir_camara
    from interfaz.registro import leer_base, validar_nombre, guardar_persona
    from interfaz.duplicados import (VerificadorDuplicados, ComprobacionFacial,
                                    exigir_alta, FALTA_CONFIGURAR)
    from interfaz.recorrido import Recorrido, PROTOCOLO
    from vision.orientacion import EstimadorOrientacion
    base = leer_base(DB_PATH)
    if prueba and nombre:
        nombre = validar_nombre(nombre, base)
    verificador = None
    evaluador = None
    if evaluacion is not None and evaluacion.get('motor', 'lbph') not in ('lbph', 'sface'):
        raise ValueError('Motor de evaluacion invalido')
    evaluar_sface = evaluacion is not None and evaluacion.get('motor', 'lbph') == 'sface'
    servicio_alta = None
    if alta_sface:
        if prueba or evaluacion is not None:
            raise ValueError('Alta y evaluacion deben estar separadas')
        nombre = validar_nombre(nombre, base)
        from interfaz.alta_sface import AltaSFace
        from configuracion.config import RAIZ_PROYECTO
        from pathlib import Path
        from interfaz.registro_traza import registrar as trazar, error_registro
        import uuid
        registro_id = registro_id or uuid.uuid4().hex
        trazar(DB_PATH,registro_id,'preparacion',evento='inicio',nombre_solicitado=nombre,camara=cam_idx)
        raiz = Path(RAIZ_PROYECTO)
        try:
            servicio_alta = AltaSFace(DB_PATH, FOTOS_DIR, raiz/'evaluacion_sface', raiz/'datos/galerias_sface')
        except Exception as error:
            error_registro(DB_PATH,registro_id,'preparacion',error)
            puente.emitir('resultado_registro',dict(estado='error',motivo='preparacion',
                indicacion='No se pudo preparar el reconocimiento. Intenta de nuevo.',mensaje='Registro no completado'))
            return
        servicio_alta.auditar = lambda etapa,**datos:trazar(DB_PATH,registro_id,etapa,**datos)
    if evaluacion is not None:
        from interfaz.evaluacion import Evaluacion
        from configuracion.config import RAIZ_PROYECTO
        from pathlib import Path
        if evaluar_sface:
            from interfaz.evaluacion_sface import EvaluacionSFace
            evaluador = EvaluacionSFace(evaluacion, DB_PATH, Path(RAIZ_PROYECTO)/'evaluacion_duplicados_sface')
        else:
            evaluador = Evaluacion(evaluacion, DB_PATH, FOTOS_DIR, Path(RAIZ_PROYECTO)/'evaluacion_duplicados')
    if not prueba and evaluador is None and servicio_alta is None:
        nombre = validar_nombre(nombre, base)
        verificador = VerificadorDuplicados(base, DB_PATH, FOTOS_DIR)
        if not verificador.vacia and (verificador.limites is None or verificador.protocolo != PROTOCOLO):
            puente.emitir("instruccion", FALTA_CONFIGURAR)
            puente.emitir("aviso_registro", FALTA_CONFIGURAR)
            return
    cap = estimador = guia = None
    try:
        estimador = EstimadorOrientacion()  # Si falta el modelo, no abrir la camara.
        cap = abrir_camara(cam_idx)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara.")
        guia = (Recorrido(time.monotonic(), conservar_originales=True) if evaluar_sface or alta_sface
                else Recorrido(time.monotonic()))
        puente.emitir("conexion", "Recorrido facial: sin conexion al Arduino")
        puente.emitir("camara", "Activa para recorrido guiado")
        def capturar():
            ultimo_analisis = None
            while not puente.detener() and not guia.completo and not guia.cancelado:
                ahora = time.monotonic()
                while True:
                    try:
                        tipo, valor = puente.comandos.get_nowait()
                    except Empty:
                        break
                    if tipo == "reintentar":
                        guia.reintentar(ahora)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("La camara dejo de entregar imagenes.")
                # Espejo SOLO en la vista previa; landmarks y JPEG usan frame original.
                puente.fotograma(cv2.flip(frame, 1))
                if not guia.pausado and (ultimo_analisis is None or ahora-ultimo_analisis >= guia.INTERVALO):
                    observacion = estimador.analizar(frame)
                    if evaluar_sface or alta_sface:
                        # Recorrido copia el original de CADA candidata y elige el mismo
                        # fotograma que su derivado LBPH, no el ultimo del paso.
                        observacion.original_bgr = frame
                    completado = time.monotonic()
                    observacion.diagnostico['intervalo_entre_analisis_s'] = (
                        round(completado-ultimo_analisis, 3) if ultimo_analisis is not None else None)
                    ultimo_analisis = completado
                else:
                    observacion = None
                    completado = time.monotonic()
                # Contar tambien el tiempo de lectura e inferencia, no solo el del bucle.
                guia.actualizar(observacion, completado)
                puente.emitir("recorrido", guia.estado())
                time.sleep(0.01)
        capturar()
        if puente.detener() and not guia.cancelado:
            guia.cancelar()
        if not guia.completo:
            if alta_sface:
                trazar(DB_PATH,registro_id,'captura',evento='incompleta',mensaje=guia.mensaje,
                       bloqueos=dict(guia.bloqueos),cancelado=guia.cancelado)
                puente.emitir('resultado_registro',dict(estado='cancelado' if guia.cancelado else 'error',
                    motivo='cancelacion' if guia.cancelado else 'capturas_incompletas',
                    indicacion='El recorrido fue cancelado. Puedes volver a intentarlo.' if guia.cancelado else 'Faltan capturas válidas. Inicia los cinco pasos de nuevo.',
                    mensaje='Registro no completado'))
            puente.emitir("recorrido", guia.estado())
            puente.emitir("aviso_registro", guia.mensaje)
            return
        if evaluador is not None:
            try:
                informe = evaluador.completar(guia.originales if evaluar_sface else guia.fotos,
                                              puente.detener)
            except (OSError, ValueError, cv2.error) as error:
                guia.cancelar('No se completó la evaluación. No se conservaron muestras parciales.')
                puente.emitir('recorrido', guia.estado())
                puente.emitir('aviso_registro', f'No se completó la evaluación: {error}')
                return
            if informe is None:
                guia.cancelar()
                puente.emitir('recorrido', guia.estado())
                puente.emitir('aviso_registro', 'Evaluación cancelada. No se conservaron capturas incompletas.')
                return
            mensaje = 'Evaluación completada. Cinco muestras conservadas solo en evaluación. No se creó un registro.'
            estado = guia.estado()
            estado.update(guardadas=0, instruccion=mensaje)
            puente.emitir('recorrido', estado)
            puente.emitir('evaluacion', informe)
            puente.emitir('aviso_registro', mensaje)
            return
        if servicio_alta is not None:
            trazar(DB_PATH,registro_id,'captura',evento='cinco_completadas',cantidad=len(guia.originales),
                   seleccionadas=[dict(paso=d['paso'],nitidez=float(d['nitidez']),angulos=list(map(float,d['angulos']))) for d in guia.detalles])
            puente.emitir('instruccion','Comprobando las cinco capturas...')
            manual = False
            limite = time.monotonic()+180
            while True:
                try:
                    resultado = servicio_alta.completar(nombre,guia.originales,guia.fotos,puente.detener, **({"confirmacion_manual":True} if manual else {}))
                except InterruptedError:
                    resultado = dict(estado='cancelado',motivo='cancelacion',indicacion='El recorrido fue cancelado. Puedes volver a intentarlo.',mensaje='Recorrido cancelado. No se creó un registro.')
                except Exception as error:
                    print(f'Error de alta SFace: {error}')
                    from interfaz.registro_traza import indicacion
                    error_registro(DB_PATH,registro_id,servicio_alta.etapa,error)
                    resultado = dict(estado='error',motivo=servicio_alta.etapa,indicacion=indicacion(servicio_alta.etapa),mensaje='Registro no completado')
                trazar(DB_PATH,registro_id,servicio_alta.etapa,evento='resultado',resultado=resultado)
                entregado = puente.emitir('resultado_registro',resultado)
                trazar(DB_PATH,registro_id,'respuesta',evento='entrega',entregado=entregado is not False)
                if resultado['estado']=='guardado':
                    puente.emitir('guardado',nombre)
                if not (resultado.get('revision_manual') or resultado.get('repetir_postura')):
                    break
                comando = None
                while not puente.detener() and time.monotonic() < limite:
                    try:
                        tipo, valor = puente.comandos.get(timeout=.1)
                    except Empty:
                        continue
                    if tipo == 'confirmar_nueva' and resultado.get('revision_manual') and valor == registro_id:
                        comando = tipo
                        break
                    if tipo == 'repetir_postura' and resultado.get('repetir_postura') and valor == registro_id:
                        comando = tipo
                        break
                if comando is None:
                    puente.emitir('resultado_registro',dict(estado='cancelado',motivo='cancelacion_o_expiracion',
                        indicacion='La revisión fue cancelada o venció. Inicia un nuevo recorrido.',mensaje='Registro no completado'))
                    break
                if comando == 'confirmar_nueva':
                    manual = True
                else:
                    manual = False
                    guia.repetir_postura(resultado['repetir_postura'],time.monotonic())
                    puente.emitir('recorrido',guia.estado())
                    capturar()
                    if not guia.completo:
                        puente.emitir('resultado_registro',dict(estado='cancelado',motivo='captura_incompleta',
                            indicacion='No se completó la postura. Inicia un nuevo recorrido.',mensaje='Registro no completado'))
                        break
            return
        if prueba:
            estado = guia.estado()
            estado.update(guardadas=0, pendiente_aprobacion=True,
                instruccion="Cinco posturas completadas. El alta requiere una aprobación todavía no definida. No se guardaron fotos ni se creó un registro.")
            puente.emitir("recorrido", estado)
            puente.emitir("aviso_registro", estado["instruccion"])
            return
        # Las cinco fotografias se reutilizan: no se solicita otro recorrido.
        try:
            exigir_alta(verificador.comprobar(guia.fotos))
            guardar_persona(nombre, guia.fotos, DB_PATH, FOTOS_DIR, 5, cancelar=puente.detener)
        except ComprobacionFacial as error:
            puente.emitir("instruccion", str(error))
            puente.emitir("aviso_registro", str(error))
            return
        estado = guia.estado()
        estado.update(guardadas=5, instruccion="Registro completo: 5 fotografias guardadas.")
        puente.emitir("recorrido", estado)
        puente.emitir("guardado", nombre)
    except KeyboardInterrupt:
        if guia is not None:
            guia.cancelar()
            puente.emitir("recorrido", guia.estado())
        puente.emitir("aviso_registro", "Recorrido cancelado. No se guardaron fotografias.")
    finally:
        if guia is not None:
            guia.limpiar()
        try:
            if cap is not None:
                cap.release()
        finally:
            if estimador is not None:
                estimador.close()


def ejecutar_trabajador(accion, parametros, eventos, cuadros, comandos, parada, video_limpio=False):
    puente = Puente(eventos, cuadros, comandos, parada)
    puente.video_limpio = video_limpio
    emociones = None
    try:
        # Evita actividades simultaneas del trabajador compartido.
        from hardware.exclusion import Exclusivo
        with Exclusivo("escritorio"):
            if accion == "sistema":
                from experimental.emociones_proceso import EmocionesWeb
                emociones = EmocionesWeb(demo=parametros.get('demo',False))
                from aplicacion.sistema import ejecutar
                ejecutar(interfaz=puente, emociones_externas=emociones, **parametros)
            else:
                registrar(puente, **parametros)
    except Exception as error:
        if parametros.get('alta_sface'):
            from interfaz.registro_traza import error_registro
            from configuracion.config import DB_PATH
            error_registro(DB_PATH,parametros.get('registro_id'),'captura_o_preparacion',error)
            camara = 'camara' in str(error).casefold() or 'cámara' in str(error).casefold()
            puente.emitir('resultado_registro',dict(estado='error',motivo='camara' if camara else 'preparacion',
                indicacion='No se pudo acceder a la cámara. Comprueba que esté disponible.' if camara else 'No se pudo iniciar la verificación. Intenta de nuevo.',
                mensaje='Registro no completado'))
        puente.emitir("error", f"{type(error).__name__}: {error}")
    finally:
        try:
            if emociones is not None:emociones.cerrar()
        except Exception as error:
            print(f'Error al cerrar emociones: {error}')
        finally:
            puente.emitir("finalizado", accion)
