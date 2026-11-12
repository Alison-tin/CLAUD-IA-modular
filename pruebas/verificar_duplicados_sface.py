from pathlib import Path
from verificar_integracion_sface import ejecutar

if __name__ == '__main__':
    raise SystemExit(ejecutar(Path(__file__).with_name('VERIFICACION_DUPLICADOS_SFACE.txt')))
