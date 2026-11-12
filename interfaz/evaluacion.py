"""Recogida separada de cinco posturas; distancias LBPH sin decisión de identidad."""
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import re
import shutil
import statistics
import uuid

from interfaz.duplicados import cargar_referencias, crear_modelos, medir_muestra
from interfaz.recorrido import PROTOCOLO
from interfaz.registro import leer_base

POSTURAS = ('frente', 'derecha', 'izquierda', 'arriba', 'abajo')
AVISO = ('Se conservarán las cinco fotografías y sus distancias en evaluacion_duplicados, '
         'separadas de la base activa. No se creará un registro ni se decidirá la identidad.')


def validar_participante(datos, base):
    if not isinstance(datos, dict) or datos.get('acepta_conservar') is not True:
        raise ValueError('Confirma que aceptas conservar las cinco muestras de evaluación.')
    tipo, identidad = datos.get('tipo'), datos.get('identificador')
    if tipo == 'registrada':
        if not isinstance(identidad, str) or identidad not in base['nombres']:
            raise ValueError('Selecciona el nombre exacto de una persona activa.')
    elif tipo == 'no_registrada':
        if not isinstance(identidad, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', identidad):
            raise ValueError('Usa un identificador de prueba de 1 a 64 letras, números, guiones o guiones bajos.')
        if identidad in base['nombres']:
            raise ValueError('El identificador de prueba no debe ser un nombre activo.')
    else:
        raise ValueError('Indica si participa una persona registrada o no registrada.')
    return {'tipo': tipo, 'identificador': identidad, 'acepta_conservar': True}


class Evaluacion:
    def __init__(self, datos, db_path, fotos_dir, destino):
        self.db_path, self.fotos_dir = db_path, fotos_dir
        base = leer_base(db_path)
        self.participante = validar_participante(datos, base)
        grupos, self.huella = cargar_referencias(base, fotos_dir)
        if not grupos:
            raise ValueError('Se necesita al menos una identidad activa para evaluar distancias.')
        self.modelos = crear_modelos(grupos)
        self.destino = Path(destino).resolve()

    def completar(self, muestras, cancelar):
        if len(muestras) != 5:
            raise ValueError('La evaluación requiere las cinco posturas completas.')
        lecturas = []
        for muestra in muestras:
            if cancelar():
                return None
            lecturas.append(medir_muestra(muestra, self.modelos))
        _, actual = cargar_referencias(leer_base(self.db_path), self.fotos_dir)
        if actual != self.huella:
            raise ValueError('La base activa cambió durante el recorrido. Repite la evaluación.')
        resumen = []
        for nombre in sorted(self.modelos):
            valores = [fila[nombre] for fila in lecturas]
            resumen.append({'nombre': nombre, 'posturas': dict(zip(POSTURAS, valores)),
                            'minima': min(valores), 'media': statistics.mean(valores),
                            'maxima': max(valores)})
        informe = {'version': 1, 'protocolo': PROTOCOLO,
                   'fecha_utc': datetime.now(timezone.utc).isoformat(),
                   'participante_declarado': self.participante,
                   'base_sha256': self.huella, 'metrica': 'distancia LBPH; menor indica mayor semejanza',
                   'decision': 'Sin decisión de identidad. Sin umbrales de alta.',
                   'personas': resumen}
        if cancelar():
            return None
        self.destino.mkdir(parents=True, exist_ok=True)
        sesion = 'recorrido_' + uuid.uuid4().hex
        temporal = self.destino / ('.incompleto_' + sesion)
        final = self.destino / sesion
        temporal.mkdir()
        try:
            for postura, muestra in zip(POSTURAS, muestras):
                if cancelar():
                    return None
                (temporal / f'{postura}.jpg').write_bytes(muestra)
            (temporal/'resumen.json').write_text(json.dumps(informe, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            salida = io.StringIO(newline='')
            escritor = csv.writer(salida)
            escritor.writerow(['persona', *POSTURAS, 'minima', 'media', 'maxima'])
            for persona in resumen:
                escritor.writerow([persona['nombre'], *persona['posturas'].values(), persona['minima'], persona['media'], persona['maxima']])
            (temporal/'distancias.csv').write_text(salida.getvalue(), encoding='utf-8-sig')
            if cancelar():
                return None
            temporal.rename(final)
            if cancelar():
                if final.parent == self.destino and final.name == sesion:
                    shutil.rmtree(final)
                return None
            informe['carpeta'] = str(final)
            return informe
        finally:
            # Únicamente nuestro directorio temporal, siempre dentro de evaluación.
            if temporal.exists() and temporal.parent == self.destino and temporal.name.startswith('.incompleto_recorrido_'):
                shutil.rmtree(temporal)
