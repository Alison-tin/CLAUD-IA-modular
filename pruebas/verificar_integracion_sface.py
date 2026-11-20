"""Runner con entrada real para multiprocessing spawn de Windows; no usar stdin."""
import contextlib
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ejecutar(salida=None):
    from configuracion.config import DB_PATH,FOTOS_DIR
    from interfaz.duplicados import preparacion_registro
    preparacion_inicial = preparacion_registro(DB_PATH,FOTOS_DIR)
    protegidos = set(ROOT.rglob('*.json')) | set(ROOT.parent.glob('*.json'))
    for nombre in ('configuracion','hardware','movimientos','secuencias','respaldos',
                   'respaldo_configuracion_20261026_190346','evaluacion_sface/referencias','evaluacion_sface/sesiones',
                   'datos/galerias_sface','evaluacion_duplicados','evaluacion_duplicados_sface',
                   'web_local/static/imagenes/posturas'):
        protegidos.update(p for p in (ROOT/nombre).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    huellas = {p:hashlib.sha256(p.read_bytes()).digest() for p in protegidos if p.is_file()}
    salida = salida or ROOT/'pruebas/VERIFICACION_SFACE_WEB.txt'
    with salida.open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        r = unittest.TextTestRunner(verbosity=2).run(suite)
        assert all(hashlib.sha256(p.read_bytes()).digest()==h for p,h in huellas.items())
        assert preparacion_registro(DB_PATH,FOTOS_DIR) == preparacion_inicial
        print('Huellas sin cambios: base, fotos, respaldos, JSON, hardware, movimientos, secuencias y capturas de evaluacion.')
        print('Disponibilidad del verificador legado sin cambios; altas SFace probadas solo con datos temporales. Sin hardware real.')
    print(f'{r.testsRun} pruebas; fallos={len(r.failures)}; errores={len(r.errors)}; {salida}')
    return not r.wasSuccessful()


if __name__ == '__main__':
    raise SystemExit(ejecutar())
