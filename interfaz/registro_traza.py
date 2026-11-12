"""Traza local de etapas y scores; nunca escribe imágenes ni embeddings."""
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback


def registrar(db_path, intento, etapa, **datos):
    ruta=Path(db_path).parent/'registro_sface.log'
    entrada=dict(fecha_utc=datetime.now(timezone.utc).isoformat(),intento=intento,etapa=etapa,**datos)
    try:
        with ruta.open('a',encoding='utf-8') as f:
            f.write(json.dumps(entrada,ensure_ascii=True,allow_nan=False)+'\n')
    except Exception as error:
        print(f'No se pudo escribir la traza de registro: {type(error).__name__}: {error}')


def error_registro(db_path,intento,etapa,error):
    registrar(db_path,intento,etapa,evento='error',tipo=type(error).__name__,detalle=str(error),
              traza=traceback.format_exc())


def indicacion(etapa):
    if etapa in ('galeria_preparacion','galeria_verificacion'):
        return 'No se pudo preparar el reconocimiento. Intenta de nuevo.'
    if etapa in ('guardado_fotos','base_provisional','respaldo','publicacion'):
        return 'No se pudo guardar el registro. Intenta de nuevo.'
    return 'No se pudo verificar la captura. Repite los cinco pasos.'
