"""Preparacion offline explicita. Sin camara, Arduino, altas ni cambios de JSON."""
from vision.galeria_sface import preparar, cargar

if __name__ == '__main__':
    try:
        carpeta = preparar()
        grupos, _, version = cargar()
        print('Galeria SFace:', carpeta)
        print('Version:', version)
        print('Referencias:', {n: len(fs) for n, fs in grupos.items()})
        print('Se conservan diferencias de cantidad. Umbral coseno 0.363 provisional.')
    except ValueError as e:
        raise SystemExit(str(e))
