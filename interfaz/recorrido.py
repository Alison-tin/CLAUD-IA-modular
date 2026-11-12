"""Cinco pasos automaticos; una foto elegida por paso, solo en memoria."""
from dataclasses import dataclass
from collections import Counter
import copy

import cv2
import numpy as np

PROTOCOLO = 'guiado-v1-5'
PASOS = (
    'Mira de frente a la camara.',
    'Gira suavemente hacia TU derecha, sin llegar de perfil.',
    'Gira suavemente hacia TU izquierda, sin llegar de perfil.',
    'Orienta el rostro ligeramente hacia arriba.',
    'Orienta el rostro ligeramente hacia abajo.',
)
RANGOS = ((-8, 8, -8, 8), (10, 25, -8, 8), (-25, -10, -8, 8),
          (-8, 8, 10, 22), (-8, 8, -22, -10))


def indicacion_postura(paso, angulos):
    if angulos is None or not np.isfinite(angulos).all():
        return 'orientacion', 'Mantén el rostro visible para estimar la postura.'
    yaw, pitch, roll = angulos
    if abs(roll) > 10:
        return 'inclinacion_lateral', 'Endereza la cabeza sin inclinarla hacia un hombro.'
    ymin, ymax, pmin, pmax = RANGOS[paso]
    if yaw < ymin:
        return 'giro_horizontal', ('Vuelve ligeramente hacia el frente; giraste demasiado a TU izquierda.' if paso == 2 else
                                   'Gira un poco más hacia TU derecha.' if paso == 1 else
                                   'Vuelve ligeramente hacia TU derecha para centrar el rostro.')
    if yaw > ymax:
        return 'giro_horizontal', ('Vuelve ligeramente hacia el frente; giraste demasiado a TU derecha.' if paso == 1 else
                                   'Gira un poco más hacia TU izquierda.' if paso == 2 else
                                   'Vuelve ligeramente hacia TU izquierda para centrar el rostro.')
    if pitch < pmin:
        return 'giro_vertical', ('Sube ligeramente: bajaste demasiado el rostro.' if paso == 4 else
                                 'Levanta un poco más el rostro.' if paso == 3 else
                                 'Levanta ligeramente el rostro sin cambiar el giro.')
    if pitch > pmax:
        return 'giro_vertical', ('Baja ligeramente: levantaste demasiado el rostro.' if paso == 3 else
                                 'Baja un poco más el rostro.' if paso == 4 else
                                 'Baja ligeramente el rostro sin cambiar el giro.')
    return '', 'Postura alcanzada. Mantén el rostro quieto para capturar.'


def postura_correcta(paso, angulos):
    return indicacion_postura(paso, angulos)[0] == ''


def casi_iguales(a, b):
    a = cv2.resize(a, (32, 32)).astype(float)
    b = cv2.resize(b, (32, 32)).astype(float)
    return float(np.mean(np.abs(a-b))) < 3


@dataclass
class Candidato:
    angulos: tuple
    imagen: object
    nitidez: float
    original_bgr: object = None


class Recorrido:
    LECTURAS = 6
    INTERVALO = 0.12
    TIEMPO_PASO = 20
    TIEMPO_TOTAL = 180
    MAX_REINTENTOS = 2

    def __init__(self, ahora, conservar_originales=False):
        self.inicio = self.inicio_paso = ahora
        self.ultima = None
        self.perdido_desde = None
        self.paso = 0
        self.repeticion_indice = None
        self.analizados = 0
        self.fotos = []
        self.conservar_originales = conservar_originales
        self.originales = []
        self.detalles = []
        self.ventana = []
        self.pausado = False
        self.cancelado = False
        self.reintentos = 0
        self.mensaje = PASOS[0]
        self.confirmacion = ''
        self.postura_ok = self.calidad_ok = False
        self.estables = 0
        self.diagnostico = {}
        self.bloqueos = Counter()
        self.ahora = ahora
        self.ultima_observacion = None

    def informar(self, codigo, mensaje):
        self.diagnostico['bloqueo'] = codigo
        self.mensaje = mensaje
        if codigo:
            self.bloqueos[codigo] += 1

    @property
    def completo(self):
        return self.paso == 5 and not self.cancelado

    def limpiar(self):
        self.repeticion_indice = None
        self.ventana.clear()
        self.fotos.clear()
        self.originales.clear()
        self.detalles.clear()
        self.estables = 0
        self.postura_ok = self.calidad_ok = False

    def cancelar(self, motivo='Recorrido cancelado. No se guardaron fotografias.'):
        self.cancelado = True
        self.pausado = False
        self.informar('cancelado', motivo)
        self.confirmacion = ''
        self.limpiar()

    def reintentar(self, ahora):
        if not self.pausado or self.cancelado:
            return
        if self.reintentos >= self.MAX_REINTENTOS or ahora-self.inicio >= self.TIEMPO_TOTAL:
            self.cancelar('Se alcanzo el limite del recorrido. Puedes iniciar otra prueba.')
            return
        self.reintentos += 1
        self.inicio_paso = ahora
        self.ventana.clear()
        self.ultima = None
        self.pausado = False
        self.mensaje = PASOS[self.paso]
        self.postura_ok = self.calidad_ok = False
        self.estables = 0

    def repetir_postura(self, postura, ahora):
        if not self.completo or type(postura) is not int or not 1 <= postura <= 5:
            raise ValueError('No hay cinco capturas para repetir esta postura')
        self.repeticion_indice = postura-1
        self.paso = postura-1
        self.inicio_paso = ahora
        self.ventana.clear()
        self.ultima = self.perdido_desde = None
        self.pausado = False
        self.estables = 0
        self.postura_ok = self.calidad_ok = False
        self.confirmacion = ''
        self.mensaje = PASOS[self.paso]

    def actualizar(self, observacion, ahora):
        self.ahora = ahora
        if self.cancelado or self.completo:
            return
        if observacion is not None:
            self.analizados += 1
            self.ultima_observacion = ahora
        if ahora-self.inicio >= self.TIEMPO_TOTAL:
            self.cancelar('Tiempo total agotado. No se guardaron fotografias.')
            return
        if self.pausado:
            return
        if ahora-self.inicio_paso >= self.TIEMPO_PASO:
            self.ventana.clear()
            self.estables = 0
            self.postura_ok = self.calidad_ok = False
            self.pausado = True
            self.informar('pausado', 'No se completo el paso. Reintenta o cancela; no fuerces el giro.')
            return
        if observacion is None:
            return
        o = observacion
        self.diagnostico = copy.deepcopy(o.diagnostico)
        self.postura_ok = o.rostros == 1 and postura_correcta(self.paso, o.angulos)
        self.calidad_ok = o.imagen is not None and o.imagen.shape == (200, 200)
        if o.rostros != 1:
            self.ventana.clear()
            self.estables = 0
            if self.perdido_desde is None:
                self.perdido_desde = ahora
            if o.rostros > 1 or ahora-self.perdido_desde >= 2:
                self.limpiar()
                self.paso = 0
                # No reiniciar el reloj: perder el rostro no prolonga el recorrido.
                self.confirmacion = ''
            self.informar('rostros', o.motivo or 'Debe haber una sola persona. Vuelve al frente.')
            return
        self.perdido_desde = None
        if not self.calidad_ok or not self.postura_ok:
            self.ventana.clear()
            self.estables = 0
            codigo, mensaje = indicacion_postura(self.paso, o.angulos)
            self.informar(self.diagnostico.get('bloqueo') or ('calidad' if not self.calidad_ok else codigo),
                          o.motivo or ('No se obtuvo una imagen valida del rostro.' if not self.calidad_ok else mensaje))
            return
        self.informar('estabilidad', 'Postura alcanzada. Mantén el rostro quieto para capturar.')
        if self.ultima is not None and ahora-self.ultima < self.INTERVALO:
            self.diagnostico['bloqueo'] = 'intervalo'
            return
        self.ultima = ahora
        original = getattr(o, 'original_bgr', None) if self.conservar_originales else None
        self.ventana.append(Candidato(o.angulos, o.imagen.copy(), o.nitidez,
                                     original.copy() if original is not None else None))
        if len(self.ventana) > self.LECTURAS:
            self.ventana.pop(0)
        self.estables = 1
        for cantidad in range(2, len(self.ventana)+1):
            if np.ptp(np.array([c.angulos for c in self.ventana[-cantidad:]]), axis=0).max() <= 3:
                self.estables = cantidad
            else:
                break
        dispersion = np.ptp(np.array([c.angulos for c in self.ventana]), axis=0)
        self.diagnostico['dispersion_ventana_grados'] = [round(float(v), 2) for v in dispersion]
        if self.estables < len(self.ventana):
            self.mensaje = 'La postura es correcta, pero aún hay movimiento. Mantén la cabeza quieta.'
        if self.estables < self.LECTURAS:
            return
        imagenes = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
                    for i,b in enumerate(self.fotos) if i != self.repeticion_indice]
        utiles = [c for c in self.ventana if not any(casi_iguales(c.imagen, otra) for otra in imagenes)]
        if not utiles:
            self.ventana.clear()
            self.estables = 0
            self.informar('muestra_parecida', 'Imagen demasiado parecida a otra. Ajusta suavemente la postura indicada.')
            return
        elegido = max(utiles, key=lambda c: c.nitidez)
        if self.conservar_originales and elegido.original_bgr is None:
            self.cancelar('Falta el original BGR de la captura seleccionada. No se completo la evaluacion.')
            return
        ok, jpeg = cv2.imencode('.jpg', elegido.imagen)
        if not ok:
            self.cancelar('No se pudo preparar la fotografia. Vuelve a intentarlo.')
            return
        detalle = {'paso': self.paso+1, 'angulos': elegido.angulos, 'nitidez': elegido.nitidez}
        if self.repeticion_indice is not None:
            i = self.repeticion_indice
            self.fotos[i] = jpeg.tobytes()
            if self.conservar_originales:
                self.originales[i] = elegido.original_bgr
            self.detalles[i] = detalle
            self.paso = 5
            self.repeticion_indice = None
        else:
            self.fotos.append(jpeg.tobytes())
            if self.conservar_originales:
                self.originales.append(elegido.original_bgr)
            self.detalles.append(detalle)
            self.paso += 1
        self.ventana.clear()
        self.estables = 0
        self.postura_ok = self.calidad_ok = False
        self.diagnostico['bloqueo'] = 'capturado'
        self.ultima = None
        self.inicio_paso = ahora
        self.confirmacion = f'Captura del paso {self.paso} confirmada.'
        self.mensaje = PASOS[self.paso] if not self.completo else 'Recorrido completo: 5 fotografias seleccionadas.'

    def estado(self):
        return {'paso': min(self.paso+1, 5), 'completados': 4 if self.repeticion_indice is not None else self.paso,
                'analizados': self.analizados, 'seleccionadas': len(self.fotos),
                'pausado': self.pausado, 'cancelado': self.cancelado,
                'completo': self.completo, 'instruccion': self.mensaje,
                'confirmacion': self.confirmacion, 'postura_ok': self.postura_ok,
                'calidad_ok': self.calidad_ok, 'lecturas_estables': self.estables,
                'lecturas_necesarias': self.LECTURAS,
                'diagnostico': {**copy.deepcopy(self.diagnostico),
                    'rangos_objetivo': dict(zip(('derecha_min', 'derecha_max', 'arriba_min', 'arriba_max'), RANGOS[min(self.paso, 4)])),
                    'lecturas_estables': self.estables, 'lecturas_necesarias': self.LECTURAS,
                    'dispersion_maxima_permitida': 3, 'intervalo_minimo_s': self.INTERVALO,
                    'segundos_restantes_paso': round(max(0, self.TIEMPO_PASO-(self.ahora-self.inicio_paso)), 1),
                    'segundos_desde_ultima_lectura': round(max(0, self.ahora-self.ultima_observacion), 2) if self.ultima_observacion is not None else None,
                    'motivos_acumulados': dict(self.bloqueos)}}
