"""Reconocimiento experimental compartido por web y terminal, sin retencion."""
import math
import os
import json
import uuid
from pathlib import Path

from configuracion.config import DB_PATH
from evaluar_sface import SFace, COSENO_PROVISIONAL
from vision.galeria_sface import cargar, digest, GaleriaInvalida


def sin_resultado(estado='sin_rostro', detalle=''):
    return dict(motor='sface', nombre=None, similitud_estimada=None, similitud_coseno=None,
                estado_reconocimiento=estado, detalle_reconocimiento=detalle)


def decidir_sface(candidato, coseno, activos, puntuaciones=None):
    if candidato not in activos or not math.isfinite(coseno) or not -1.00001 <= coseno <= 1.00001:
        return sin_resultado('error', 'Puntuacion SFace invalida')
    if puntuaciones is not None:
        if (set(puntuaciones)!=set(activos) or any(not math.isfinite(v) or not -1.00001<=v<=1.00001
                for v in puntuaciones.values()) or puntuaciones.get(candidato)!=coseno
                or coseno!=max(puntuaciones.values())):
            return sin_resultado('error','Ranking SFace inconsistente')
        if sum(abs(v-coseno)<=1e-6 for v in puntuaciones.values())>1:
            r=sin_resultado('ambiguo','Primeras candidatas empatadas; sin identidad suficiente')
            r['nombre']='Desconocido'
            return r
    if coseno < COSENO_PROVISIONAL:
        resultado = sin_resultado('rechazado')
        resultado['nombre'] = 'Desconocido'
        return resultado
    return dict(motor='sface', nombre=candidato, similitud_estimada=None,
                similitud_coseno=float(coseno), estado_reconocimiento='aceptado',
                detalle_reconocimiento='Coseno provisional; no es probabilidad')


class ConfirmacionSFace:
    """Tres lecturas consecutivas; jamás retiene un nombre ante rechazo o relevo.

    Medida contra lecturas aisladas, no calibración biométrica ni prueba de identidad.
    """
    REQUERIDAS=3

    def reiniciar(self):
        self.candidato=None;self.cuenta=0;self.caja=None

    def __init__(self):
        self.reiniciar()

    def aplicar(self,lectura,caja):
        if lectura['estado_reconocimiento']!='aceptado' or caja is None:
            self.reiniciar()
            return lectura
        if lectura['nombre']!=self.candidato or self.caja is None or iou(caja,self.caja)<.3:
            self.reiniciar()
        self.candidato=lectura['nombre'];self.caja=tuple(caja)
        self.cuenta=min(self.cuenta+1,self.REQUERIDAS)
        if self.cuenta<self.REQUERIDAS:
            r=sin_resultado('confirmando','Se requieren tres lecturas consecutivas de la misma candidata')
            r['nombre']='Desconocido'
            return r
        return lectura


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    derecha, abajo = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, derecha-x)*max(0, abajo-y)
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.


class MotorSFace:
    def __init__(self, db_path=DB_PATH):
        self.galeria, self.base_sha, self.version = cargar(db_path=db_path)
        self.db_path = Path(db_path)
        self.modelo = SFace()
        self.confirmacion = ConfirmacionSFace()
        self.diagnostico_activo = os.environ.get('CLAUD_IA_DIAGNOSTICO_SFACE') == '1'
        self.sesion_diagnostico = uuid.uuid4().hex

    def reconocer(self, frame, cajas_seguimiento, fotograma_id=None):
        if not hasattr(self,'confirmacion'):
            self.confirmacion=ConfirmacionSFace()
        datos = {}
        try:
            resultado = self._reconocer(frame, cajas_seguimiento, datos)
            resultado = self.confirmacion.aplicar(resultado,cajas_seguimiento[0] if len(cajas_seguimiento)==1 else None)
        except Exception as error:
            self.confirmacion.reiniciar()
            if getattr(self,'diagnostico_activo',False):
                self._registrar(fotograma_id, sin_resultado('error', str(error)), datos)
            raise
        if getattr(self,'diagnostico_activo',False):
            self._registrar(fotograma_id, resultado, datos)
        return resultado

    def _registrar(self, fotograma_id, resultado, datos):
        ranking = sorted(datos.get('puntuaciones', {}).items(), key=lambda x: x[1], reverse=True)
        primera = ranking[0] if ranking else (None, None)
        segunda = ranking[1] if len(ranking) > 1 else (None, None)
        entrada = dict(sesion=self.sesion_diagnostico, fotograma=fotograma_id, galeria=self.version,
            primera=primera[0], coseno_primera=primera[1], segunda=segunda[0], coseno_segunda=segunda[1],
            margen=primera[1]-segunda[1] if segunda[1] is not None else None,
            umbral=COSENO_PROVISIONAL, decision=resultado['estado_reconocimiento'],
            consecutivas=self.confirmacion.cuenta, requeridas=ConfirmacionSFace.REQUERIDAS,
            resultado=resultado['nombre'], motivo=('maximo >= umbral provisional; tres lecturas consecutivas; sin margen minimo calibrado'
                if resultado['estado_reconocimiento']=='aceptado' else resultado['detalle_reconocimiento']) or
                ('maximo por identidad inferior al umbral provisional' if resultado['estado_reconocimiento']=='rechazado'
                 else resultado['estado_reconocimiento']))
        print('[SFace diagnostico] ' + json.dumps(entrada, ensure_ascii=True, allow_nan=False), flush=True)

    def registrar_sin_comparacion(self, fotograma_id, estado):
        self.confirmacion.reiniciar()
        if self.diagnostico_activo:
            self._registrar(fotograma_id, sin_resultado(estado), {})

    def _reconocer(self, frame, cajas_seguimiento, diagnostico=None):
        try:
            base_sha = digest(self.db_path.read_bytes())
        except OSError as error:
            raise GaleriaInvalida('No se puede verificar la base activa; SFace detenido') from error
        if base_sha != self.base_sha:
            raise GaleriaInvalida('La base activa cambio: galeria SFace invalidada. Deten y ejecuta preparar_galeria_sface.py.')
        if not self.galeria:
            return sin_resultado('sin_registros','No hay personas registradas')
        if not cajas_seguimiento:
            return sin_resultado()
        if len(cajas_seguimiento) != 1:
            return sin_resultado('varios_rostros', 'SFace requiere un solo rostro')
        caras = self.modelo.detectar(frame)
        if not caras:
            return sin_resultado('sin_rostro', 'YuNet no detecta un rostro')
        if len(caras) != 1:
            return sin_resultado('varios_rostros', 'YuNet detecta varios rostros')
        cara = caras[0]
        if iou(cara[:4], cajas_seguimiento[0]) < .3:
            return sin_resultado('sin_correspondencia', 'YuNet y seguimiento no corresponden a la misma caja')
        feature, error = self.modelo.extraer_cara(frame, cara)
        if feature is None:
            return sin_resultado('error', error)
        if diagnostico is None:
            candidato, score, _ = self.modelo.comparar(feature, self.galeria)
        else:
            candidato, score, _ = self.modelo.comparar(feature, self.galeria, diagnostico=diagnostico)
        return decidir_sface(candidato, score, self.galeria, diagnostico.get('puntuaciones') if diagnostico is not None else None)
