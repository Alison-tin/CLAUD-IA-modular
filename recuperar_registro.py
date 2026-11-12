"""Recuperar un retiro sin reemplazar otras personas. No abre camara ni Arduino."""
import argparse
import json

from configuracion.config import DB_PATH, FOTOS_DIR
from interfaz.bajas import carpeta_respaldos, recuperar_persona


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--listar", action="store_true")
    grupo.add_argument("--respaldo", help="Identificador retiro_... mostrado con --listar")
    args = parser.parse_args()
    if args.listar:
        for ruta in sorted(carpeta_respaldos(DB_PATH).glob("retiro_*/registro.json")):
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            print(f"{ruta.parent.name} | {datos['nombre']} | {datos['cantidad']} fotos")
    else:
        nombre = recuperar_persona(args.respaldo, DB_PATH, FOTOS_DIR)
        print(f"Recuperado: {nombre}. Inicia de nuevo el sistema para entrenar el reconocedor.")


if __name__ == "__main__":
    main()
