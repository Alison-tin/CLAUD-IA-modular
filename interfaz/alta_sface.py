"""Alta SFace: regla local provisional y publicación de base al final.

Los límites NO proceden del reconocimiento en vivo. Véase REGLA_ALTA_SFACE.md.
"""
import math
import os
from pathlib import Path
import pickle
import shutil
import uuid

import cv2
import numpy as np

from evaluar_sface import SFace, csv_guardar
from hardware.exclusion import Exclusivo
from interfaz.registro import leer_base, validar_nombre
from vision.galeria_sface import cargar, inventario, preparar, digest

VERSION_REGLA = 'alta-sface-local-v1'
MAX_NUEVA = .35
MIN_DUPLICADO = .55
MIN_MARGEN = .20
REPETIR = 'No se pudo confirmar la lectura. Repite las cinco capturas.'


def decidir(lecturas, activos):
    """Cinco entradas completas; dos fronteras con zona de abstención."""
    activos=set(activos)
    if len(lecturas)!=5:
        return 'dudoso', None
    ranking=[]
    for scores in lecturas:
        if set(scores)!=activos or any(type(v) not in (int,float) or not math.isfinite(v)
                or not -1.00001<=v<=1.00001 for v in scores.values()):
            return 'dudoso', None
        ranking.append(sorted(scores.items(),key=lambda p:p[1],reverse=True))
    if not activos:
        return 'nueva', None
    if all(r[0][1]<=MAX_NUEVA for r in ranking):
        return 'nueva', None
    if (len({r[0][0] for r in ranking})==1
            and all(r[0][1]>=MIN_DUPLICADO and (len(r)==1 or r[0][1]-r[1][1]>=MIN_MARGEN) for r in ranking)):
        return 'duplicado',ranking[0][0][0]
    return 'dudoso', None


def eliminar_propio(ruta, padre):
    ruta,padre=Path(ruta).resolve(),Path(padre).resolve()
    if ruta.parent!=padre or not ruta.name.startswith(('alta_','registro_','v1_')):
        raise ValueError('Limpieza fuera de la transaccion')
    if ruta.exists():
        shutil.rmtree(ruta)


class AltaSFace:
    def __init__(self, db_path, fotos_dir, evaluacion, galerias):
        self.db=Path(db_path);self.fotos=Path(fotos_dir)
        self.evaluacion=Path(evaluacion);self.galerias=Path(galerias)
        self.grupos,self.base_sha,self.version=cargar(self.db,self.evaluacion,self.galerias)
        self.modelo=SFace()
        self.etapa='preparacion'
        self.auditar=None

    def _anotar(self,etapa,**datos):
        self.etapa=etapa
        if self.auditar is not None:
            try:
                self.auditar(etapa,**datos)
            except Exception as error:
                print(f'Error de traza de registro: {error}')

    def _dudoso(self,motivo,detalle='',postura=None):
        self._anotar(self.etapa,evento='dudoso',motivo=motivo,detalle=detalle)
        mensajes={
            'comparacion_dudosa':'La comparación no fue concluyente. Repite las cinco capturas.',
            'capturas_incompletas':'Faltan fotografías. Repite los cinco pasos.',
            'original_invalido':'Una fotografía no es válida. Repite las cinco capturas.',
            'recorte_invalido':'No se pudo preparar una captura. Repite los cinco pasos.',
            'codificacion':'No se pudo procesar una fotografía. Repite las capturas.',
            'capturas_repetidas':'Las capturas repiten la misma imagen. Repite los cinco pasos.',
            'extraccion_sin_lectura':'No se confirmó un solo rostro en alguna fotografía. Repite los cinco pasos.',
            'extraccion_error':'No se pudo analizar una fotografía. Repite las capturas.',
            'puntuacion_invalida':'El análisis de una captura no fue válido. Repite los cinco pasos.',
            'galeria_cambiada':'Los registros cambiaron durante el intento. Vuelve a intentarlo.'}
        mensaje=mensajes.get(motivo,'No se pudo verificar la captura. Repite los cinco pasos.')
        resultado=dict(estado='dudoso',motivo=motivo,indicacion=mensaje,mensaje=REPETIR)
        if postura is not None:
            resultado.update(repetir_postura=postura,indicacion=f'No se pudo verificar la postura {postura}. Repite solo esa postura; las otras se conservan.')
        return resultado

    def completar(self, nombre, originales, jpegs, cancelar, confirmacion_manual=False):
        self._anotar('validacion',evento='inicio',regla=VERSION_REGLA,galeria=self.version,
                     limites=dict(nueva_max=MAX_NUEVA,duplicado_min=MIN_DUPLICADO,margen_min=MIN_MARGEN))
        if cancelar(): return dict(estado='cancelado',mensaje='Recorrido cancelado. No se creó un registro.')
        nombre=validar_nombre(nombre,leer_base(self.db))
        if len(originales)!=5 or len(jpegs)!=5:
            return self._dudoso('capturas_incompletas')
        lecturas=[]; pngs=[]; hashes=set()
        for postura,(frame,jpeg) in enumerate(zip(originales,jpegs),1):
            self._anotar('extraccion',postura=len(lecturas)+1)
            if cancelar(): return dict(estado='cancelado',mensaje='Recorrido cancelado. No se creó un registro.')
            if (not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3
                    or frame.shape[2]!=3 or min(frame.shape[:2])<120):
                return self._dudoso('original_invalido',postura=postura)
            try:
                gris=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_GRAYSCALE)
            except (TypeError,ValueError,cv2.error):
                return self._dudoso('recorte_invalido',postura=postura)
            if gris is None or gris.shape!=(200,200):
                return self._dudoso('recorte_invalido',postura=postura)
            ok,png=cv2.imencode('.png',frame)
            if not ok: return self._dudoso('codificacion',postura=postura)
            contenido=png.tobytes();h=digest(contenido)
            if h in hashes: return self._dudoso('capturas_repetidas',postura=postura)
            hashes.add(h);pngs.append(contenido)
            try:
                f,error=self.modelo.extraer(frame)
                if f is None: return self._dudoso('extraccion_sin_lectura',error,postura)
                scores={n:max(float(self.modelo.reconocedor.match(f.copy(),ref.copy(),cv2.FaceRecognizerSF_FR_COSINE))
                              for ref in refs) for n,refs in self.grupos.items()}
            except (ValueError,cv2.error) as error:
                return self._dudoso('extraccion_error',f'{type(error).__name__}: {error}',postura)
            if any(not math.isfinite(v) or not -1.00001<=v<=1.00001 for v in scores.values()):
                return self._dudoso('puntuacion_invalida',repr(scores),postura)
            lecturas.append(scores)
        self._anotar('comparacion',puntuaciones=lecturas)
        estado,persona=decidir(lecturas,self.grupos)
        if cancelar(): return dict(estado='cancelado',mensaje='Recorrido cancelado. No se creó un registro.')
        if digest(self.db.read_bytes())!=self.base_sha or cargar(self.db,self.evaluacion,self.galerias)[2]!=self.version:
            return self._dudoso('galeria_cambiada')
        if estado=='duplicado':
            self._anotar('comparacion',evento='duplicado',persona=persona)
            return dict(estado=estado,persona=persona,
                        mensaje=f'Esta persona ya está registrada como {persona}. No se guardó un nuevo registro')
        if estado!='nueva':
            # Un indicio alto no prueba duplicado, pero tampoco autoriza aprobar
            # una identidad nueva. Reutiliza el límite existente, no el de vivo.
            if any(max(s.values())>=MIN_DUPLICADO for s in lecturas):
                cercana=max(self.grupos,key=lambda n:sum(s[n] for s in lecturas)/5)
                resultado=self._dudoso('posible_duplicado')
                resultado.update(revision_manual=False,posible_duplicado=True,persona_cercana=cercana,
                    puntuaciones=lecturas,indicacion=f'Hay parecido con {cercana}. No se puede confirmar un registro nuevo con estas capturas.')
                return resultado
            if confirmacion_manual is not True:
                resultado=self._dudoso('comparacion_dudosa')
                resultado.update(revision_manual=True,puntuaciones=lecturas,
                    indicacion='Comparación dudosa. Revisa las similitudes antes de confirmar que es una persona nueva.')
                return resultado
            self._anotar('revision_manual',evento='persona_nueva_confirmada',puntuaciones=lecturas)
        self._anotar('comparacion',evento='nueva')
        return self._guardar(nombre,jpegs,pngs,cancelar)

    def _guardar(self,nombre,jpegs,pngs,cancelar):
        self._anotar('preparacion_guardado')
        with Exclusivo('base_personas'):
            antes=self.db.read_bytes()
            if digest(antes)!=self.base_sha or cargar(self.db,self.evaluacion,self.galerias)[2]!=self.version:
                raise ValueError('La base o la galeria cambio. Repite las capturas.')
            base=leer_base(self.db);nombre=validar_nombre(nombre,base)
            identificador=uuid.uuid4().hex
            fotos=self.fotos/('registro_'+identificador)
            referencias=self.evaluacion/'referencias'/('alta_'+identificador)
            temporal=self.db.with_name(self.db.name+'.'+identificador+'.tmp')
            respaldo=self.db.with_name(self.db.name+'.antes_registro_'+identificador)
            galeria_nueva=None;propia=False;publicado=False
            fotos_creadas=refs_creadas=temporal_creado=respaldo_creado=False
            try:
                self._anotar('guardado_fotos')
                fotos.mkdir(parents=True,exist_ok=False)
                fotos_creadas=True
                referencias.mkdir(parents=True,exist_ok=False)
                refs_creadas=True
                filas=[];rutas=[]
                for i,(jpeg,png) in enumerate(zip(jpegs,pngs),1):
                    if cancelar(): raise InterruptedError('Registro cancelado')
                    foto=fotos/f'foto_{i:03d}.jpg'; foto.write_bytes(jpeg);rutas.append(str(foto.resolve()))
                    archivo=f'captura_{i:03d}.png';(referencias/archivo).write_bytes(png)
                    filas.append(dict(persona=nombre,archivo=archivo,tipo='registrada',uso='referencias',
                        sesion=referencias.name,sha256=digest(png),origen='alta-sface-v1'))
                csv_guardar(referencias/'capturas.csv',filas)
                self._anotar('base_provisional')
                nueva=dict(base,nombres=base['nombres']+[nombre]*5,imagenes=base['imagenes']+rutas)
                with temporal.open('xb') as f:
                    temporal_creado=True
                    pickle.dump(nueva,f);f.flush();os.fsync(f.fileno())
                if leer_base(temporal)!=nueva: raise ValueError('Base provisional no verificable')
                # La galería derivada queda lista ANTES de publicar la nueva base.
                version=inventario(temporal,self.evaluacion)[2]
                galeria_nueva=self.galerias/('v1_'+version);propia=not galeria_nueva.exists()
                self._anotar('galeria_preparacion')
                preparar(temporal,self.evaluacion,self.galerias)
                self._anotar('galeria_verificacion')
                if cargar(temporal,self.evaluacion,self.galerias)[2]!=version:
                    raise ValueError('Galeria nueva no verificable')
                if self.db.read_bytes()!=antes: raise ValueError('La base cambio durante el guardado')
                if cancelar(): raise InterruptedError('Registro cancelado')
                self._anotar('respaldo')
                with respaldo.open('xb') as f:
                    respaldo_creado=True
                    f.write(antes);f.flush();os.fsync(f.fileno())
                if respaldo.read_bytes()!=antes: raise ValueError('Respaldo no verificable')
                if cancelar(): raise InterruptedError('Registro cancelado')
                # Punto de confirmación: ambos recursos existen y se verificaron.
                self._anotar('publicacion')
                os.replace(temporal,self.db)
                publicado=True
            finally:
                if not publicado:
                    if temporal_creado: temporal.unlink(missing_ok=True)
                    if respaldo_creado: respaldo.unlink(missing_ok=True)
                    if fotos_creadas: eliminar_propio(fotos,self.fotos)
                    if refs_creadas: eliminar_propio(referencias,self.evaluacion/'referencias')
                    if propia and galeria_nueva is not None:
                        eliminar_propio(galeria_nueva,self.galerias)
            self._anotar('confirmado',evento='guardado',galeria=version)
            return dict(estado='guardado',persona=nombre,galeria=version,mensaje='Registro guardado correctamente')
