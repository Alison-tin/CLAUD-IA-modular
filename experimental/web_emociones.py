"""Web experimental Demo: captura/gestos en contrib y emociones en headless."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]


def trabajador(accion, parametros, eventos, cuadros, comandos, parada, video_limpio=False):
    from interfaz.trabajador import Puente
    from hardware.exclusion import Exclusivo
    puente = Puente(eventos, cuadros, comandos, parada)
    puente.video_limpio = video_limpio
    cliente = None
    try:
        if accion != 'sistema' or not parametros.get('demo'):
            raise ValueError('Esta entrada experimental solo permite el sistema en Demo; sin altas ni modo fisico')
        from experimental.emociones_proceso import EmocionesProceso
        from aplicacion.sistema import ejecutar
        cliente = EmocionesProceso()
        with Exclusivo('escritorio'):
            ejecutar(interfaz=puente, emociones_externas=cliente, **parametros)
    except Exception as error:
        puente.emitir('error', f'{type(error).__name__}: {error}')
    finally:
        if cliente:
            cliente.cerrar()
        puente.emitir('finalizado', accion)


def crear_control():
    from web_local.control import Control

    class ControlExperimental(Control):
        def __init__(self):
            super().__init__(trabajador=trabajador)

        def iniciar(self, actividad, *args, **kwargs):
            if actividad != 'sistema' or kwargs.get('modo', 'demo') != 'demo':
                raise ValueError('Prueba separada: solo reconocimiento y gestos en Demo; sin registro ni modo fisico')
            return super().iniciar(actividad, *args, **kwargs)

        def eliminar(self, nombre):
            raise ValueError('Las bajas no estan disponibles en esta prueba separada')

        def evento(self, tipo, valor):
            super().evento(tipo, valor)
            if tipo == 'deteccion' and valor['deepface'].get('externo'):
                with self.lock:
                    detalle = valor['deepface']['detalle']
                    self.estado['detalle_emociones'] = detalle
                    self.estado['emociones_estado'] = ('listo' if valor['deepface']['ok'] else
                        'cargando' if detalle.startswith('cargando') else 'no disponible')
                    if not valor['deepface']['ok']:
                        self.estado['expresion'] = None

        def detener(self):
            super().detener()
            with self.lock:
                self.estado['expresion'] = None
                self.estado['emociones_estado'] = 'detenido'

    return ControlExperimental()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--puerto', type=int, default=8766)
    parser.add_argument('--diagnostico-sface', action='store_true',
                        help='Imprime frame, dos candidatas y margen; no guarda imagenes')
    args = parser.parse_args()
    if args.diagnostico_sface:
        os.environ['CLAUD_IA_DIAGNOSTICO_SFACE'] = '1'
    else:
        os.environ.pop('CLAUD_IA_DIAGNOSTICO_SFACE', None)
    if not 1024 <= args.puerto <= 65535:
        parser.error('Puerto invalido')
    esperado = (ROOT/'.venv_prueba_opencv/Scripts/python.exe').resolve()
    if Path(sys.executable).resolve() != esperado:
        parser.error('Ejecutar con .venv_prueba_opencv/Scripts/python.exe')
    hijo = ROOT/'.venv_prueba_emociones/Scripts/python.exe'
    if not hijo.is_file():
        parser.error('Falta .venv_prueba_emociones; no se sustituye por otro interprete')
    from web import RUTA_ACCESO
    if not RUTA_ACCESO.is_file():
        parser.error('Falta la configuracion de acceso existente. Esta prueba no crea ni modifica JSON')
    from hardware.exclusion import Exclusivo
    from web_local.servidor import Servidor
    with Exclusivo('servidor_web'):
        control = crear_control()
        servidor = Servidor(('127.0.0.1', args.puerto), control,
                            json.loads(RUTA_ACCESO.read_text(encoding='utf-8')))
        print(f'Prueba Demo con emociones separadas: http://localhost:{args.puerto}')
        print('No se abre camara hasta Iniciar. Ctrl+C cierra ambos procesos.')
        try:
            servidor.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass
        finally:
            control.cerrar()
            servidor.server_close()


if __name__ == '__main__':
    mp.freeze_support()
    main()
