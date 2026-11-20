"""Suite completa y después inferencia real por IPC con una imagen sintética."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from verificar_integracion_sface import ejecutar
    if ejecutar(ROOT/'pruebas/VERIFICACION_EMOCIONES_SEPARADAS_SUITE.log'):
        return 1
    import numpy as np
    from experimental.emociones_proceso import EmocionesProceso
    imagen=np.full((100,100,3),128,np.uint8)
    cliente=EmocionesProceso(inicio_timeout=45,timeout=10,vigencia=15,intervalo=1)
    try:
        fin=time.monotonic()+50
        while time.monotonic()<fin:
            resultado=cliente.actualizar(imagen,objetivo='sintetico')
            if resultado:
                assert 'deepface' not in sys.modules
                assert 'retinaface' not in sys.modules
                print('INFERENCIA REAL DeepFace ONNX por IPC: salida finita sobre imagen sintetica. No valida emociones humanas.')
                print('Proceso principal sin DeepFace; interprete hijo:',cliente.interprete)
                break
            if cliente.fallo:
                raise RuntimeError(cliente.estado())
            time.sleep(.02)
        else:
            raise RuntimeError('Sin resultado real dentro del tiempo de prueba')
    finally:
        cliente.cerrar()
    assert cliente.proceso.poll() is not None
    print('Proceso real cerrado y recogido. No se abrio camara ni Arduino.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
