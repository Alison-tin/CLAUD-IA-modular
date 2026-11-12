"""Arranca el CMD habitual y comprueba solo acceso HTTP, sin iniciar actividades."""
import hashlib
import http.client
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    acceso=ROOT/'web_local/.privado/acceso.json'
    if not acceso.is_file():
        raise RuntimeError('Falta el acceso existente; no se crea ni cambia clave')
    huella=hashlib.sha256(acceso.read_bytes()).digest()
    with socket.socket() as prueba:
        try: prueba.bind(('127.0.0.1',8765))
        except OSError:
            raise RuntimeError('8765 ya esta ocupado; no se toca el servidor existente')
    with (ROOT/'documentacion/VERIFICACION_LANZADOR_8765.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen(['cmd.exe','/d','/c',str(ROOT/'Iniciar_CLAUD_IA_Web.cmd')],
            cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW)
        log.write(f'CMD de esta prueba: PID {p.pid}\n');log.flush()
        try:
            for _ in range(100):
                try:
                    conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=1)
                    conn.request('GET','/');r=conn.getresponse();pagina=r.read().decode('utf-8');conn.close()
                    assert r.status==200
                    assert 'id="recognition-engine" type="hidden" value="sface"' in pagina
                    assert 'id="evaluation-engine" type="hidden" value="sface"' in pagina
                    assert 'id="registration-verification"' in pagina
                    break
                except (ConnectionError,OSError):
                    time.sleep(.1)
            else: raise RuntimeError('El CMD no sirvio la pagina en 8765')
            conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=2)
            conn.request('GET','/api/estado');r=conn.getresponse();r.read();conn.close()
            assert r.status==401  # La contraseña existente sigue siendo necesaria.
            for imagen in ('logo-claud-ia.png','posturas/frente.png'):
                conn=http.client.HTTPConnection('127.0.0.1',8765,timeout=2)
                conn.request('GET','/imagenes/'+imagen);r=conn.getresponse();contenido=r.read();conn.close()
                assert r.status==200 and contenido.startswith(b'\x89PNG\r\n\x1a\n')
            log.write('\nCMD habitual: HTTP 8765 OK, motores fijos SFace y acceso protegido. Sin iniciar actividades.\n')
        finally:
            # Solo el árbol del CMD creado por esta prueba; nunca otro servidor.
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=10,check=True)
            p.wait(timeout=10)
            p.stdin.close()
    assert hashlib.sha256(acceso.read_bytes()).digest()==huella
    print('Lanzador habitual 8765 verificado sin camara ni Arduino. Acceso existente intacto. Arbol propio cerrado.')


if __name__=='__main__': main()
