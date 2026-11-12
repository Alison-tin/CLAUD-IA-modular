"""Cinco exposiciones del recorrido: solo evidencias de duplicados, nunca altas."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import math
import statistics
import unicodedata
import uuid

import cv2
import numpy as np

from evaluar_sface import SFace, csv_guardar, entorno
from interfaz.evaluacion import POSTURAS, validar_participante
from interfaz.recorrido import PROTOCOLO
from interfaz.registro import leer_base
from vision.galeria_sface import cargar, inventario, digest


def validar_participante_sface(datos, base):
    participante = validar_participante(datos, base)
    alias = datos.get('nombre_propuesto', '')
    if not isinstance(alias, str):
        raise ValueError('El nombre propuesto de prueba debe ser texto')
    alias = unicodedata.normalize('NFKC', alias).strip()
    if participante['tipo'] == 'registrada':
        if (not 1 <= len(alias) <= 60 or any(unicodedata.category(c).startswith('C') for c in alias)
                or alias.casefold() in {n.strip().casefold() for n in base['nombres']}):
            raise ValueError('Indica otro nombre de prueba, distinto de las identidades activas; no se registrara')
    participante.update(motor='sface', nombre_propuesto=alias)
    return participante


def analizar_lecturas(lecturas, participante):
    validas = [r for r in lecturas if r['puntuaciones']]
    votos = Counter(r['primera'] for r in validas if not r.get('empate_exacto'))
    candidata = votos.most_common(1)[0][0] if votos else None
    faltantes = [r['postura'] for r in lecturas if not r['puntuaciones']]
    identidad = participante['identificador']
    empatadas = [n for n, cantidad in votos.items() if cantidad == max(votos.values())] if votos else []
    descripcion = f'{len(validas)}/5 posturas con lectura. Primeras candidatas: {dict(votos)}. '
    if len(empatadas) > 1:
        descripcion += f'Empate de votos entre {empatadas}; no hay candidata unica. '
        candidata = None
    elif candidata:
        descripcion += f'Primera candidata repetida: {candidata}, {votos[candidata]} posturas. '
    if any(r.get('empate_exacto') for r in validas):
        descripcion += 'Hay empates exactos de coseno; no se cuentan como voto unico. '
    if participante['tipo'] == 'registrada':
        descripcion += f"Identidad declarada {identidad}: primera en {votos[identidad]}/5 posturas. "
    else:
        descripcion += 'Participante externo: una primera candidata no demuestra coincidencia ni justifica bloquearlo. '
    descripcion += ('Sin criterio de duplicados validado. No se decide duplicado, bloqueo individual ni alta. '
                    'No se aplica el umbral de reconocimiento. Las altas generales siguen bloqueadas.')
    return dict(candidata_repetida=candidata, primeras_por_identidad=dict(votos),
                validas=len(validas), sin_lectura=faltantes, descripcion=descripcion,
                decision_duplicado=None, permitir_alta=False)


class EvaluacionSFace:
    def __init__(self, datos, db_path, destino):
        self.db_path = Path(db_path)
        self.participante = validar_participante_sface(datos, leer_base(db_path))
        self.galeria, self.base_sha, self.version = cargar(db_path=db_path)
        self.filas_referencia, _, version = inventario(db_path=db_path)
        if version != self.version:
            raise ValueError('La galeria cambio al iniciar la evaluacion')
        self.modelo = SFace()
        self.destino = Path(destino).resolve()

    def completar(self, originales, cancelar):
        if len(originales) != 5:
            raise ValueError('Se necesitan los cinco originales del mismo recorrido, sin capturas adicionales')
        lecturas, pngs, hashes = [], [], set()
        referencias = {f['sha256'] for f in self.filas_referencia}
        for postura, frame in zip(POSTURAS, originales):
            if cancelar():
                return None
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                    or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0):
                raise ValueError('Falta un original BGR; no se utilizan recortes grises LBPH para SFace')
            if np.array_equal(frame[:,:,0],frame[:,:,1]) and np.array_equal(frame[:,:,1],frame[:,:,2]):
                raise ValueError('La entrada es gris replicado; no se simula recuperar color para SFace')
            ok, png = cv2.imencode('.png', frame)
            if not ok:
                raise ValueError('No se pudo preparar una de las cinco muestras')
            contenido = png.tobytes(); h = digest(contenido)
            if h in referencias or h in hashes:
                raise ValueError('Se reutilizo una referencia o una captura identica dentro del recorrido')
            hashes.add(h); pngs.append(contenido)
            try:
                feature, error = self.modelo.extraer(frame)
            except (cv2.error, ValueError) as e:
                feature, error = None, str(e)
            puntuaciones = {}
            if feature is not None:
                # Solo scores crudos contra TODAS las identidades. No llamar comparar/decidir_sface.
                for nombre, fs in self.galeria.items():
                    score = max(float(self.modelo.reconocedor.match(feature.copy(), f.copy(), cv2.FaceRecognizerSF_FR_COSINE)) for f in fs)
                    if not math.isfinite(score) or not -1.00001 <= score <= 1.00001:
                        raise ValueError('SFace devolvio coseno invalido')
                    puntuaciones[nombre] = score
            orden = sorted(puntuaciones, key=puntuaciones.get, reverse=True)
            primera = orden[0] if orden else None
            segunda = orden[1] if len(orden)>1 else None
            lecturas.append(dict(postura=postura, puntuaciones=puntuaciones, primera=primera,
                                 segunda=segunda, separacion=(puntuaciones[primera]-puntuaciones[segunda]) if segunda else None,
                                 empate_exacto=bool(segunda and puntuaciones[primera] == puntuaciones[segunda]),
                                 error=error, sha256=h))
        if cancelar():
            return None
        if cargar(db_path=self.db_path)[2] != self.version:
            raise ValueError('La base o la galeria cambio durante el recorrido; no se conserva un informe incompatible')
        personas = []
        for nombre in sorted(self.galeria):
            valores = [r['puntuaciones'][nombre] for r in lecturas if r['puntuaciones']]
            personas.append(dict(nombre=nombre, referencias=len(self.galeria[nombre]),
                                 posturas={r['postura']:r['puntuaciones'].get(nombre) for r in lecturas},
                                 minima=min(valores) if valores else None,
                                 media=statistics.mean(valores) if valores else None,
                                 maxima=max(valores) if valores else None))
        analisis = analizar_lecturas(lecturas,self.participante)
        informe = dict(motor='sface', version=1, protocolo=PROTOCOLO+'-original-bgr-sface-evaluacion-v1',
                       galeria_sha256=self.version, base_sha256=self.base_sha,
                       participante_declarado=self.participante, metrica='similitud coseno; no es probabilidad',
                       lecturas=lecturas, personas=personas, analisis=analisis,
                       decision='Sin decision de alta ni criterio de duplicados validado')
        self.destino.mkdir(parents=True,exist_ok=True)
        final = self.destino/('recorrido_'+uuid.uuid4().hex)
        temporal = self.destino/('.incompleto_'+final.name)
        temporal.mkdir()
        try:
            for postura, contenido in zip(POSTURAS,pngs):
                if cancelar():
                    return None
                with (temporal/(postura+'.png')).open('xb') as f:
                    f.write(contenido)
            filas = [dict(postura=r['postura'], persona=n, coseno=r['puntuaciones'].get(n),
                          primera=r['primera'], segunda=r['segunda'], separacion=r['separacion'],
                          error=r['error'], sha256=r['sha256']) for r in lecturas for n in sorted(self.galeria)]
            csv_guardar(temporal/'puntuaciones.csv',filas)
            csv_guardar(temporal/'referencias_galeria.csv',self.filas_referencia)
            (temporal/'entorno.txt').write_text('Entorno del reconocedor. Sus umbrales NO se usan para decidir duplicados en esta evaluacion.\n'+entorno(),encoding='utf-8')
            texto = '# Evaluacion de duplicados SFace, sin altas\n\n'
            texto += f'Fecha UTC: {datetime.now(timezone.utc).isoformat()}\n\n'
            texto += f'Participante declarado: {self.participante}\n\nGaleria: {self.version}\n\n'
            texto += analisis['descripcion']+'\n\n'
            texto += 'El 0,363 del reconocimiento NO se utiliza para decidir duplicados.\n\n'
            texto += '| Postura | Primera | Segunda | Separacion coseno | Error |\n|---|---|---|---|---|\n'
            texto += ''.join(f"| {r['postura']} | {r['primera']} | {r['segunda']} | {r['separacion']} | {r['error']} |\n" for r in lecturas)
            texto += '\nCosenos por identidad/postura en puntuaciones.csv. No se crearon registros ni politicas.\n'
            (temporal/'resumen.md').write_text(texto,encoding='utf-8')
            if cancelar():
                return None
            temporal.rename(final)
            if cancelar():
                import shutil
                assert final.resolve().parent == self.destino and final.name.startswith('recorrido_')
                shutil.rmtree(final)
                return None
            informe['carpeta']=str(final)
            return informe
        finally:
            if temporal.exists():
                import shutil
                # Solo nuestra carpeta temporal creada arriba, hija del destino de evaluacion.
                assert temporal.resolve().parent == self.destino and temporal.name.startswith('.incompleto_recorrido_')
                shutil.rmtree(temporal)
