"""Apertura comun de la camara, sin guardar imagenes ni cambiar registros."""
import cv2
from hardware.exclusion import Exclusivo

from configuracion.config import (
    CAMARA_BACKEND, CAMARA_AJUSTES, CAMARA_PERFIL, CAMARA_AJUSTES_ANTERIORES,
)


def informar_camara(cap):
    """Lectura informativa; no escribe propiedades."""
    for nombre, _ in CAMARA_AJUSTES_ANTERIORES:
        print(f" {nombre}: leido={cap.get(getattr(cv2, nombre)):g}")
    print(" Los valores leidos dependen del controlador; 0 o -1 pueden "
          "ser valores no disponibles. No confirman el modo automatico.")


class CamaraExclusiva:
    def __init__(self, captura, bloqueo):
        self.captura = captura
        self.bloqueo = bloqueo

    def __getattr__(self, nombre):
        return getattr(self.captura, nombre)

    def release(self):
        try:
            self.captura.release()
        finally:
            self.bloqueo.liberar()


def abrir_camara(cam_idx=0):
    bloqueo = Exclusivo("camara").adquirir()
    try:
        cap = _abrir_camara(cam_idx)
        if not cap.isOpened():
            bloqueo.liberar()
        return CamaraExclusiva(cap, bloqueo)
    except BaseException:
        bloqueo.liberar()
        raise


def _abrir_camara(cam_idx=0):
    cap = cv2.VideoCapture(cam_idx, getattr(cv2, CAMARA_BACKEND))
    if not cap.isOpened():
        print(f" Camara {cam_idx}: no disponible con {CAMARA_BACKEND}")
        return cap

    resultados = []
    for nombre, solicitado in CAMARA_AJUSTES:
        propiedad = getattr(cv2, nombre)
        aceptado = cap.set(propiedad, solicitado)
        resultados.append((nombre, propiedad, solicitado, aceptado))

    print(f" Camara {cam_idx}: perfil={CAMARA_PERFIL}, backend solicitado={CAMARA_BACKEND}, "
          f"backend informado={cap.getBackendName()}")
    for nombre, propiedad, solicitado, aceptado in resultados:
        print(f" {nombre}: solicitado={solicitado}, "
              f"set={aceptado}, leido={cap.get(propiedad):g}")
    informar_camara(cap)
    return cap


if __name__ == "__main__":
    # Vista previa sin acceso a la base de personas ni escritura de fotos.
    import argparse
    parser = argparse.ArgumentParser(description="Prueba de iluminacion de camara")
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()
    cap = abrir_camara(args.cam)
    try:
        if cap.isOpened():
            print(" Q=salir | C=solicitar panel del controlador | L=leer ajustes")
            print(" El panel puede no estar disponible. No se restablecen ajustes previos.")
            primera = True
            while True:
                ok, frame = cap.read()
                if not ok:
                    print(" No se pudo leer un fotograma.")
                    break
                if primera:
                    print(f" Fotograma recibido: {frame.shape[1]}x{frame.shape[0]}")
                    primera = False
                cv2.imshow("Prueba camara - Q salir, C controles, L lectura", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("c"):
                    resultado = cap.set(cv2.CAP_PROP_SETTINGS, 1)
                    print(f" Solicitud de panel: {resultado}; comprueba si se abrio.")
                    informar_camara(cap)
                if tecla == ord("l"):
                    informar_camara(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
