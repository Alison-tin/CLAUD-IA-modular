"""Entrada web local. No inicia camara ni Arduino al arrancar el servidor."""
import argparse
import getpass
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path

from hardware.exclusion import Exclusivo
from web_local.servidor import Servidor, crear_acceso, direcciones_locales

RUTA_ACCESO = Path(__file__).resolve().parent / "web_local" / ".privado" / "acceso.json"


def configurar(cambiar=False):
    if RUTA_ACCESO.exists() and not cambiar:
        return json.loads(RUTA_ACCESO.read_text(encoding="utf-8"))
    print("Configura una clave exclusiva para CLAUD-IA (12 a 128 caracteres).")
    clave = getpass.getpass("Clave: ")
    if clave != getpass.getpass("Repite la clave: "):
        raise ValueError("Las claves no coinciden.")
    acceso = crear_acceso(clave)
    RUTA_ACCESO.parent.mkdir(parents=True, exist_ok=True)
    temporal = RUTA_ACCESO.with_suffix(".tmp")
    with temporal.open("x", encoding="utf-8") as archivo:
        json.dump(acceso, archivo)
    temporal.replace(RUTA_ACCESO)
    return acceso


def main():
    mp.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="Solo este computador; sin acceso Wi-Fi")
    parser.add_argument("--puerto", type=int, default=8765)
    parser.add_argument("--cambiar-clave", action="store_true")
    parser.add_argument("--diagnostico-sface", action="store_true", help="Lecturas SFace por fotograma, sin guardar imagenes")
    args = parser.parse_args()
    if args.diagnostico_sface:
        os.environ['CLAUD_IA_DIAGNOSTICO_SFACE'] = '1'
    else:
        os.environ.pop('CLAUD_IA_DIAGNOSTICO_SFACE', None)
    if not 1024 <= args.puerto <= 65535:
        parser.error("El puerto debe estar entre 1024 y 65535.")
    logging.basicConfig(level=logging.WARNING)
    with Exclusivo("servidor_web"):
        acceso = configurar(args.cambiar_clave)
        from web_local.control import Control
        control = Control()
        servidor = Servidor(("127.0.0.1" if args.local else "0.0.0.0", args.puerto), control, acceso)
        print(f"\nCLAUD-IA · Demo o cabeza fisica (con confirmacion en la web)\nEn este computador: http://localhost:{args.puerto}")
        if not args.local:
            for ip in direcciones_locales():
                print(f"Posible direccion Wi-Fi: http://{ip}:{args.puerto}")
        print("Usa una red local de confianza. No abras puertos del router ni uses tuneles.")
        print("Ctrl+C para detener el servidor y cerrar la camara.\n")
        try:
            servidor.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            print("\nCerrando CLAUD-IA...")
        finally:
            control.cerrar()
            servidor.server_close()


if __name__ == "__main__":
    main()
