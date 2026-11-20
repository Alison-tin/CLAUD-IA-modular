"""Un solo trabajador y una sola camara compartida por los navegadores."""
from collections import Counter
import copy
import logging
import math
import os
import multiprocessing as mp
from queue import Empty, Full
import threading
import time
import uuid

from configuracion.config import DB_PATH, FOTOS_DIR, FOTOS_POR_PER
from interfaz.bajas import retirar_persona
from interfaz.registro import leer_base, validar_nombre
from interfaz.trabajador import ejecutar_trabajador

LOG = logging.getLogger("claudia.web")
GESTOS = {"mano_abierta": "Mano abierta", "puno": "Puño", "pulgar_arriba": "Pulgar arriba",
          "pulgar_abajo": "Pulgar abajo", "indice": "Índice", "dos_dedos": "Dos dedos",
          "tres_dedos": "Tres dedos", "cuatro_dedos": "Cuatro dedos", "ok": "OK"}


class Control:
    def __init__(self, db_path=DB_PATH, fotos_dir=FOTOS_DIR, trabajador=None):
        self.trabajador = trabajador
        self.diagnostico_desarrollo = os.environ.get("CLAUD_IA_DIAGNOSTICO_REGISTRO") == "1"
        self.db_path = db_path
        self.fotos_dir = fotos_dir
        self.lock = threading.RLock()
        self.ctx = mp.get_context("spawn")
        self.proceso = None
        self.jpeg = None
        self.version = 0
        self.deteniendo = False
        self.captura_pendiente = False
        self.ultimo_frame = 0
        self.ultima_expresion = 0
        self.numero_aviso = 0
        self.estado = self.vacio()
        self.hilo = None

    @staticmethod
    def vacio():
        return {"fase": "detenido", "actividad": None, "camara": False,
                "modo": "demo", "conexion": "Sin conexión con Arduino",
                "arduino": "desconectado", "puerto": None,
                "motor_seleccionado": "sface", "motor_activo": None, "galeria_sface": None,
                "estado_reconocimiento": "sin_lectura", "detalle_reconocimiento": "",
                "similitud_coseno": None, "nombre": None, "similitud_estimada": None, "expresion": None, "gesto": None, "rostros": 0,
                "emociones_estado": "sin iniciar", "detalle_emociones": "", "progreso": 0, "total": 5,
                "prueba_registro": False, "recorrido": {},
                "registro_nombre": "", "instruccion": "", "aviso": "", "aviso_id": 0,
                "opciones": {"reconocimiento": True, "emociones": True, "gestos": True}}

    def personas(self):
        with self.lock:
            return [{"nombre": n, "fotos": c} for n, c in sorted(Counter(leer_base(self.db_path)["nombres"]).items())]

    def foto(self):
        with self.lock:
            return self.jpeg if self.estado["camara"] and time.monotonic()-self.ultimo_frame < 4 else None

    def eliminar(self, nombre):
        with self.lock:
            if self.proceso is not None:
                raise ValueError("Detén el sistema o el registro y espera a que se cierre antes de eliminar.")
            if not isinstance(nombre, str) or not nombre:
                raise ValueError("Selecciona una persona registrada.")
            try:
                respaldo = retirar_persona(nombre, self.db_path, self.fotos_dir,
                    antes_publicar=self._preparar_galeria_retiro)
            except RuntimeError:
                raise ValueError("Cierra la actividad de escritorio o terminal antes de retirar un registro.")
            except OSError:
                LOG.exception('No se pudo publicar la eliminación')
                raise ValueError('No se pudo completar la eliminación. El registro se conserva. Intenta de nuevo.')
            self.estado.update(nombre=None, similitud_estimada=None, similitud_coseno=None, expresion=None, gesto=None)
            self.aviso("Persona eliminada y galería actualizada. Fotos respaldadas. Puedes iniciar el sistema.")
            return respaldo

    def _preparar_galeria_retiro(self, temporal):
        from vision.galeria_sface import preparar, cargar, inventario, GALERIAS
        from interfaz.alta_sface import eliminar_propio
        nueva=None
        propia=False
        try:
            version=inventario(temporal)[2]
            nueva=GALERIAS/('v1_'+version)
            propia=not nueva.exists()
            preparar(temporal)
            cargar(temporal)
        except Exception:
            LOG.exception('Falló la galería antes de confirmar la eliminación')
            if propia and nueva is not None:
                try:
                    eliminar_propio(nueva,GALERIAS)
                except OSError:
                    LOG.exception('No se pudo retirar la galería provisional')
            raise ValueError('No se pudo actualizar la galería. La eliminación no se confirmó y la persona se conserva. Intenta de nuevo.')

    def ver(self):
        with self.lock:
            estado = copy.deepcopy(self.estado)
            if self.ultima_expresion and time.monotonic()-self.ultima_expresion > 4:
                estado["expresion"] = None
            estado["captura_lista"] = False  # El recorrido ya no requiere capturas manuales.
            recorrido = estado.get("recorrido", {})
            if not self.diagnostico_desarrollo:
                estables = recorrido.pop("lecturas_estables", 0)
                necesarias = recorrido.pop("lecturas_necesarias", 6)
                recorrido.pop("diagnostico", None)
                recorrido.pop("analizados", None)
                recorrido["progreso_postura"] = estables / max(1, necesarias)
            return estado

    def aviso(self, texto):
        self.numero_aviso += 1
        self.estado["aviso"] = texto
        self.estado["aviso_id"] = self.numero_aviso

    def puertos(self):
        import serial.tools.list_ports
        return [{"puerto": p.device, "descripcion": p.description or "Puerto serie"}
                for p in serial.tools.list_ports.comports()]

    def preparacion_registro(self):
        with self.lock:
            from vision.galeria_sface import cargar
            try:
                cargar(db_path=self.db_path)
            except (OSError,ValueError):
                return dict(permitido=False,mensaje='SFace no esta disponible. Comprueba modelos y galeria.')
            return dict(permitido=True,mensaje='Puedes iniciar las cinco capturas. Se comprobara SFace antes de guardar.')

    def iniciar(self, actividad, nombre="", cam_idx=0, modo="demo", puerto=None, confirmar_fisico=False, prueba=False, evaluacion=None, alta_sface=False):
        with self.lock:
            if self.proceso is not None:
                raise ValueError("Detén la actividad actual antes de abrir otra.")
            if actividad not in ("sistema", "registro") or type(cam_idx) is not int or not 0 <= cam_idx <= 9:
                raise ValueError("Selecciona una cámara válida.")
            if type(prueba) is not bool or (prueba and actividad != 'registro'):
                raise ValueError('Modo de prueba de captura no valido.')
            if type(alta_sface) is not bool or (alta_sface and (actividad!='registro' or prueba or evaluacion is not None)):
                raise ValueError('Modo de alta SFace no valido')
            if modo not in ("demo", "fisico") or (actividad == "registro" and modo != "demo"):
                raise ValueError("Modo no válido. El registro no conecta Arduino.")
            if modo == "fisico":
                if confirmar_fisico is not True:
                    raise ValueError("Confirma primero el centrado, apertura de párpados y prueba de maxilar.")
                if puerto is not None and (not isinstance(puerto, str) or puerto not in {p["puerto"] for p in self.puertos()}):
                    raise ValueError("El puerto seleccionado ya no está disponible. Actualiza los puertos.")
            if evaluacion is not None:
                from interfaz.evaluacion import validar_participante
                if actividad != 'registro' or prueba or modo != 'demo':
                    raise ValueError('La evaluación usa captura separada sin Arduino.')
                if not isinstance(evaluacion, dict) or evaluacion.get('motor', 'sface') != 'sface':
                    raise ValueError('La evaluacion utiliza SFace')
                evaluacion = dict(evaluacion, motor='sface')
                if evaluacion.get('motor') == 'sface':
                    from interfaz.evaluacion_sface import validar_participante_sface
                    from vision.galeria_sface import cargar
                    evaluacion = validar_participante_sface(evaluacion, leer_base(self.db_path))
                    cargar(db_path=self.db_path)  # Antes de crear proceso o abrir camara.
                else:
                    evaluacion = validar_participante(evaluacion, leer_base(self.db_path))
            if actividad == "registro" and not prueba and evaluacion is None:
                nombre = validar_nombre(nombre, leer_base(self.db_path))
                if not alta_sface:
                    raise ValueError('Usa Comenzar registro para el recorrido y comprobacion SFace.')
                from vision.galeria_sface import asegurar
                asegurar(db_path=self.db_path)
            motor = self.estado["motor_seleccionado"]
            if actividad == "sistema" and motor == "sface":
                from vision.galeria_sface import asegurar
                asegurar(db_path=self.db_path)  # Preflight local sin proceso, camara o Arduino.
            opciones = self.estado["opciones"].copy()
            self.estado = self.vacio()
            self.estado["motor_seleccionado"] = motor
            self.estado.update(fase="iniciando", actividad=actividad, registro_nombre=nombre, opciones=opciones,
                               modo=modo, arduino="conectando" if modo == "fisico" else "demo")
            self.estado['evaluacion_registro'] = evaluacion is not None
            self.estado['evaluacion_motor'] = evaluacion.get('motor', 'lbph') if evaluacion else None
            self.estado['resultado_evaluacion'] = None
            self.estado['resultado_registro'] = None
            self.estado['registro_id'] = uuid.uuid4().hex if actividad == 'registro' else None
            self.estado['alta_sface'] = alta_sface
            self.estado['prueba_registro'] = prueba
            if prueba:
                self.estado['registro_nombre'] = nombre or 'Captura sin alta'
            self.jpeg = None
            self.deteniendo = self.captura_pendiente = False
            self.eventos = self.ctx.Queue(maxsize=100)
            self.cuadros = self.ctx.Queue(maxsize=2)
            self.comandos = self.ctx.Queue(maxsize=30)
            self.parada = self.ctx.Event()
            parametros = {"cam_idx": cam_idx}
            parametros.update({"demo": modo != "fisico", "port": puerto} if actividad == "sistema" else {"nombre": nombre, "prueba": prueba})
            if actividad == "sistema":
                parametros["motor"] = motor
            if alta_sface:
                parametros['alta_sface'] = True
                parametros['registro_id'] = self.estado['registro_id']
            if evaluacion is not None:
                self.estado['registro_nombre'] = 'Evaluación: ' + evaluacion['identificador']
                parametros['evaluacion'] = evaluacion
            self.proceso = self.ctx.Process(target=self.trabajador or ejecutar_trabajador,
                args=(actividad, parametros, self.eventos, self.cuadros, self.comandos, self.parada, True))
            try:
                self.proceso.start()
                self.comandos.put_nowait(("opciones", opciones))
            except Exception:
                self.proceso = None
                self.estado.update(fase="error", actividad=None)
                for cola in (self.eventos, self.cuadros, self.comandos):
                    cola.close()
                raise
            self.hilo = threading.Thread(target=self._leer, daemon=True)
            self.hilo.start()

    def detener(self):
        with self.lock:
            self.estado.update(expresion=None,emociones_estado='detenido',detalle_emociones='')
            if self.proceso is not None:
                self.deteniendo = True
                self.estado["fase"] = "deteniendo"
                self.estado["motor_activo"] = None
                self.estado.update(nombre=None, similitud_estimada=None, similitud_coseno=None)
                if self.estado['actividad'] == 'registro' and not self.estado.get('resultado_evaluacion'):
                    self.estado['recorrido'].update(seleccionadas=0, pausado=False, confirmacion='',
                                                   cancelado=True, completo=False)
                self.parada.set()

    def opciones(self, opciones):
        with self.lock:
            if not isinstance(opciones, dict) or set(opciones)-set(self.estado["opciones"]) or any(type(v) is not bool for v in opciones.values()):
                raise ValueError("Opciones no válidas.")
            if self.proceso is not None and not self.deteniendo:
                try:
                    self.comandos.put_nowait(("opciones", opciones))
                except Full:
                    raise ValueError("Espera un momento y vuelve a intentarlo.")
            self.estado["opciones"].update(opciones)
            for opcion, campo in (("reconocimiento", "nombre"), ("emociones", "expresion"), ("gestos", "gesto")):
                if not self.estado["opciones"][opcion]:
                    self.estado[campo] = None
            if not self.estado['opciones']['emociones']:
                self.estado.update(emociones_estado='desactivado',detalle_emociones='')
            if not self.estado["opciones"]["reconocimiento"]:
                self.estado["similitud_estimada"] = None
                self.estado["similitud_coseno"] = None
                self.estado["estado_reconocimiento"] = "desactivado"

    def seleccionar_motor(self, motor):
        with self.lock:
            if self.proceso is not None or self.estado["fase"] not in ("detenido", "error"):
                raise ValueError("Deten el sistema y espera al cierre antes de cambiar el motor")
            if motor != "sface":
                raise ValueError("El reconocimiento utiliza SFace")
            self.estado.update(motor_seleccionado=motor, motor_activo=None, galeria_sface=None,
                               nombre=None, similitud_estimada=None, similitud_coseno=None,
                               estado_reconocimiento="sin_lectura", detalle_reconocimiento="")

    def capturar(self):
        raise ValueError('La captura ahora es automatica en los cinco pasos guiados.')

    def iniciar_recorrido(self, nombre, cam_idx=0):
        if not isinstance(nombre, str):
            raise ValueError('Escribe el nombre de la persona')
        nombre = validar_nombre(nombre, leer_base(self.db_path))
        self.iniciar('registro', nombre=nombre, cam_idx=cam_idx, alta_sface=True)

    def resolver_registro(self, registro_id, manual=False, confirmacion=False):
        with self.lock:
            s=self.estado
            r=s.get('resultado_registro') or {}
            permitido=r.get('revision_manual') if manual else r.get('repetir_postura')
            if (not registro_id or registro_id != s.get('registro_id') or not permitido
                    or not s.get('alta_sface') or self.deteniendo or self.proceso is None
                    or not self.proceso.is_alive() or (manual and (confirmacion is not True or r.get('posible_duplicado')))):
                raise ValueError('Esta revisión no está disponible. Actualiza el estado del registro.')
            try:
                self.comandos.put_nowait(('confirmar_nueva' if manual else 'repetir_postura',registro_id))
            except Full:
                raise ValueError('Espera a que termine la operación anterior.')
            s['resultado_registro']=None
            s['recorrido']['completo']=False
            s['instruccion']='Verificando el registro...' if manual else 'Repite la postura indicada.'

    def reintentar(self):
        with self.lock:
            if (self.estado['actividad'] != 'registro' or not self.estado['recorrido'].get('pausado')
                    or self.deteniendo or self.proceso is None):
                raise ValueError('Solo puedes reintentar un paso que este pausado.')
            try:
                self.comandos.put_nowait(("reintentar", None))
            except Full:
                raise ValueError("Espera a que termine la captura anterior.")
            self.estado['recorrido']['pausado'] = False

    def evento(self, tipo, valor):
        s = self.estado
        if tipo == 'resultado_registro':
            s['resultado_registro'] = copy.deepcopy(valor)
            s['resultado_registro']['nombre_solicitado'] = s['registro_nombre']
            s['resultado_registro']['registro_id'] = s.get('registro_id')
            s['instruccion'] = valor['mensaje']
            self.aviso(valor['mensaje'])
            return
        if tipo == 'evaluacion':
            s['resultado_evaluacion'] = copy.deepcopy(valor)
            return
        if tipo == 'recorrido':
            if s['actividad'] == 'registro' and not self.deteniendo:
                s['recorrido'] = copy.deepcopy(valor)
                s['progreso'] = valor['completados']
                s['instruccion'] = valor['instruccion']
            return
        if tipo == "camara":
            s["camara"] = True
            if not self.deteniendo and s["fase"] != "error":
                s["fase"] = "activo"
        elif tipo == "conexion":
            if s["modo"] != "fisico":
                s["conexion"] = "Registro sin conexión a Arduino" if s["actividad"] == "registro" else "Demo · sin órdenes al Arduino"
        elif tipo == "arduino":
            s["arduino"], s["puerto"] = valor["estado"], valor["puerto"]
            s["conexion"] = {"demo": "Demo · sin órdenes al Arduino",
                "conectado": f"Puerto abierto: {valor['puerto']} · sin confirmación de movimientos",
                "error": "Error de comunicación con Arduino. Comprueba cable, puerto y otras aplicaciones.",
                "desconectado": "Arduino desconectado. La actividad se está deteniendo."}[valor["estado"]]
            if valor["estado"] in ("error", "desconectado"):
                s["fase"] = "error"
                self.aviso(s["conexion"])
        elif tipo == "motor_reconocimiento" and not self.deteniendo:
            if valor["motor"] != s["motor_seleccionado"]:
                raise ValueError("El trabajador inicio un motor distinto del seleccionado")
            s["motor_activo"] = valor["motor"]
            s["galeria_sface"] = valor.get("version")
        elif tipo == "reconocimiento" and s["opciones"]["reconocimiento"] and not self.deteniendo and s["fase"] != "error":
            if valor.get("motor", "lbph") != s["motor_seleccionado"]:
                s.update(nombre=None, similitud_estimada=None, similitud_coseno=None)
                return
            s["nombre"] = valor["nombre"]
            s["similitud_estimada"] = s["similitud_coseno"] = None
            s["estado_reconocimiento"] = valor.get("estado_reconocimiento", "sin_lectura")
            s["detalle_reconocimiento"] = valor.get("detalle_reconocimiento", "")
            aceptado = s["nombre"] not in (None, "Desconocido")
            if s["motor_seleccionado"] == "sface":
                coseno = valor.get("similitud_coseno")
                if (aceptado and valor.get("motor") == "sface" and type(coseno) in (int, float)
                        and math.isfinite(coseno) and .363 <= coseno <= 1.00001):
                    s["similitud_coseno"] = coseno
                elif aceptado:
                    s["nombre"] = None
            else:
                similitud = valor.get("similitud_estimada")
                s["similitud_estimada"] = (similitud if aceptado
                    and type(similitud) in (int, float) and 0 <= similitud <= 1 else None)
        elif tipo == "expresion" and s["opciones"]["emociones"] and s["rostros"]==1 and not self.deteniendo and s['fase']=='activo':
            s["expresion"] = valor["nombre"]
            self.ultima_expresion = time.monotonic()
        elif tipo == "deteccion":
            s["rostros"] = valor["rostros"]
            s["gesto"] = GESTOS.get(valor["gesto"]) if s["opciones"]["gestos"] else None
            df = valor["deepface"]
            s["emociones_estado"] = "listo" if df["ok"] else "no disponible" if df["intentado"] else "cargando"
            s['detalle_emociones']=df.get('mensaje_usuario','')
            if df.get('externo') and df.get('detalle','').startswith('cargando'):
                s['emociones_estado']='cargando'
            if not df['ok']:s['expresion']=None
            if not s['opciones']['emociones']:s['emociones_estado']='desactivado'
            if self.deteniendo or s['fase']=='detenido':s.update(expresion=None,emociones_estado='detenido',detalle_emociones='')
            if s['fase']=='error':s.update(expresion=None,emociones_estado='no disponible',detalle_emociones='El sistema se detuvo.')
            if not s["rostros"]:
                s["nombre"] = s["expresion"] = s["similitud_estimada"] = s["similitud_coseno"] = None
                s["estado_reconocimiento"] = "sin_rostro"
        elif tipo == "aviso_secuencia":
            self.aviso(str(valor))
        elif tipo == "aviso_registro":
            self.aviso(str(valor))
        elif tipo == "secuencia" and valor.get("iniciada"):
            nombres = {'risa.json':'Risa', 'habla_media.json':'Habla', 'guino_derecho.json':'Guiño derecho',
                'guino_izquierdo.json':'Guiño izquierdo', 'parpadeo_doble.json':'Parpadeo doble',
                'barrido_horizontal.json':'Barrido horizontal', 'mirada_arriba.json':'Mirada arriba',
                'mirada_abajo.json':'Mirada abajo', 'sorpresa.json':'Sorpresa'}
            self.aviso(f"{nombres.get(valor['archivo'], valor['archivo'].removesuffix('.json').replace('_',' ').capitalize())} iniciada")
        elif tipo == "secuencia" and not valor["disponible"]:
            self.aviso("Gesto reconocido. Su movimiento aún no está disponible.")
        elif tipo == "progreso":
            s["progreso"] = valor
            self.captura_pendiente = False
        elif tipo == "instruccion":
            s["instruccion"] = valor
            self.captura_pendiente = False
        elif tipo == "guardado":
            self.aviso("Registro guardado correctamente")
            s["instruccion"] = "Registro guardado correctamente"
        elif tipo == "error":
            LOG.error("Trabajador: %s", valor)
            s["fase"] = "error"
            s.update(expresion=None,emociones_estado='no disponible',detalle_emociones='El sistema se detuvo.')
            s.update(nombre=None, similitud_estimada=None, similitud_coseno=None, motor_activo=None)
            if s["motor_seleccionado"] == "sface":
                self.aviso(f"SFace: {valor}")
                return
            if s["modo"] == "fisico" and s["arduino"] in ("error", "desconectado", "conectando"):
                s["arduino"] = "error"
                s["conexion"] = "No se pudo mantener la conexión con Arduino. Comprueba cable, puerto y otras aplicaciones."
                self.aviso(s["conexion"])
            else:
                self.aviso("No se pudo completar la actividad. Detén el sistema y comprueba la cámara en el computador de CLAUD-IA.")

    def _leer(self):
        try:
            while True:
                with self.lock:
                    for _ in range(100):
                        try:
                            self.evento(*self.eventos.get_nowait())
                        except Empty:
                            break
                    while True:
                        try:
                            self.jpeg = self.cuadros.get_nowait()
                            self.version += 1
                            self.ultimo_frame = time.monotonic()
                        except Empty:
                            break
                    if not self.proceso.is_alive():
                        self.proceso.join()
                        codigo = self.proceso.exitcode
                        # El hijo puede enviar el resultado entre el primer vaciado
                        # de eventos y la consulta is_alive. Recogerlo tras join.
                        while True:
                            try:
                                self.evento(*self.eventos.get_nowait())
                            except Empty:
                                break
                        if self.estado.get('alta_sface') and not self.estado.get('resultado_registro') and not self.deteniendo:
                            from interfaz.registro_traza import registrar
                            registrar(self.db_path,self.estado.get('registro_id'),'respuesta',evento='sin_resultado',codigo_salida=codigo)
                            self.evento('resultado_registro',dict(estado='error',motivo='respuesta_no_confirmada',
                                indicacion='No se recibió la confirmación. Revisa Personas antes de intentarlo de nuevo.',mensaje='Registro no completado'))
                        self.proceso.close()
                        self.proceso = None
                        self.jpeg = None
                        self.estado["motor_activo"] = None
                        self.estado.update(camara=False, actividad=None, nombre=None, similitud_estimada=None, similitud_coseno=None, expresion=None, gesto=None,
                            emociones_estado='detenido',detalle_emociones='')
                        if self.estado["modo"] == "fisico" and self.estado["arduino"] not in ("error", "desconectado"):
                            self.estado.update(arduino="desconectado", conexion="Puerto cerrado. Arduino sin conexión activa.")
                        if self.estado["fase"] != "error":
                            self.estado["fase"] = "detenido" if codigo == 0 else "error"
                        for cola in (self.eventos, self.cuadros, self.comandos):
                            cola.cancel_join_thread()
                            cola.close()
                        return
                time.sleep(0.04)
        except Exception:
            LOG.exception("Error en el puente web")
            self.detener()
            with self.lock:
                self.estado["fase"] = "error"

    def cerrar(self):
        self.detener()
        if self.hilo is not None:
            self.hilo.join()  # No abandona el proceso con camara abierta.
