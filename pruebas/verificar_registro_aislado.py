"""Regresion e integridad; datos de alta nuevos solo sinteticos y temporales."""
from pathlib import Path
from verificar_integracion_sface import ejecutar

if __name__=='__main__':
    raise SystemExit(ejecutar(Path(__file__).with_name('VERIFICACION_REGISTRO_AISLADO.txt')))
