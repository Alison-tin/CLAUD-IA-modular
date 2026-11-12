import os
import urllib.request
from configuracion.config import MODELO_CARA_URL, MODELO_CARA_PATH, MODELO_MANO_URL, MODELO_MANO_PATH

def descargar_modelos():
    for url, path, desc in [
        (MODELO_CARA_URL, MODELO_CARA_PATH, "cara (~1 MB)"),
        (MODELO_MANO_URL, MODELO_MANO_PATH, "mano (~24 MB)"),
    ]:
        if not os.path.exists(path):
            print(f" Descargando modelo {desc}...")
            urllib.request.urlretrieve(url, path)
            print(f" {path}")
        else:
            print(f" Modelo: {path}")

