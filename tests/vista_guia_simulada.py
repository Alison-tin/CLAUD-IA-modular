"""Verificacion visual local con Edge headless, datos sinteticos y sin hardware.

Sirve solo el CSS, los SVG y un dialogo de prueba. Nunca inicia la aplicacion,
lee su clave ni accede a camara/Arduino. Escribe capturas en pruebas/.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import base64
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT/'web_local/static'
EDGE = Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')


class DevTools:
    """Cliente local minimo para emular el viewport; no requiere instalar paquetes."""
    def __init__(self, url):
        u = urlsplit(url)
        self.socket = socket.create_connection((u.hostname, u.port), timeout=10)
        self.numero = 0
        clave = base64.b64encode(os.urandom(16)).decode()
        self.socket.sendall((f'GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {clave}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        cabecera = b''
        while not cabecera.endswith(b'\r\n\r\n'):
            cabecera += self.leer(1)
        if cabecera.split(b' ', 2)[1] != b'101':
            raise RuntimeError(cabecera.decode(errors='replace'))

    def leer(self, cantidad):
        datos = b''
        while len(datos) < cantidad:
            nuevo = self.socket.recv(cantidad-len(datos))
            if not nuevo:
                raise EOFError('DevTools cerro la conexion')
            datos += nuevo
        return datos

    def enviar(self, datos, opcode=1):
        mascara = os.urandom(4)
        n = len(datos)
        cabecera = bytes([0x80|opcode, 0x80|n]) if n < 126 else bytes([0x80|opcode, 0x80|126])+struct.pack('!H', n)
        self.socket.sendall(cabecera+mascara+bytes(v ^ mascara[i % 4] for i, v in enumerate(datos)))

    def recibir(self):
        datos = b''
        while True:
            primero, segundo = self.leer(2)
            longitud = segundo & 127
            if longitud == 126:
                longitud = struct.unpack('!H', self.leer(2))[0]
            elif longitud == 127:
                longitud = struct.unpack('!Q', self.leer(8))[0]
            mascara = self.leer(4) if segundo & 128 else None
            contenido = self.leer(longitud)
            if mascara:
                contenido = bytes(v ^ mascara[i % 4] for i, v in enumerate(contenido))
            opcode = primero & 15
            if opcode == 8:
                raise EOFError('DevTools cerro la conexion')
            if opcode == 9:
                self.enviar(contenido, 10)
                continue
            if opcode in (0, 1):
                datos += contenido
                if primero & 128:
                    return json.loads(datos)

    def llamar(self, metodo, parametros=None):
        self.numero += 1
        self.enviar(json.dumps({'id': self.numero, 'method': metodo, 'params': parametros or {}}).encode())
        while True:
            respuesta = self.recibir()
            if respuesta.get('id') == self.numero:
                if 'error' in respuesta:
                    raise RuntimeError(respuesta['error'])
                return respuesta.get('result', {})


def main():
    if not EDGE.is_file():
        raise RuntimeError('Edge headless no esta disponible; no se instala ningun navegador.')
    html = (STATIC/'index.html').read_text(encoding='utf-8')
    a = html.index('<dialog id="register-dialog">')
    dialogo = html[a:html.index('</dialog>', a)+9]
    js = (STATIC/'app.js').read_text(encoding='utf-8')
    prefijo = js[:js.index('let csrf =')]
    resultados = []
    codigo = prefijo + '''
    document.getElementById('register-dialog').showModal();
    document.getElementById('register-dialog').classList.add('is-running');
    ['registration-intro','registration-preparation','check-registration','register-form','test-registration'].forEach(id=>$(id).hidden=true);
    $('registration-active').hidden=false;
    $('registration-name').textContent='Prueba visual simulada';
    $('registration-count').textContent='Paso 2 de 5';
    $('registration-progress').value=1;
    $('registration-summary').textContent='';
    $('registration-confirmation').textContent='Captura del paso 1 confirmada.';
    $('registration-wait').textContent='CAMARA SIMULADA · SIN PERSONAS';
    $('registration-instruction').textContent='Postura alcanzada. Mantén el rostro quieto para capturar.';
    renderPosture({paso:2,completados:1,postura_ok:true,calidad_ok:true,progreso_postura:0.5},true);
    setTimeout(()=>{
      const a=document.querySelector('.registration-preview').getBoundingClientRect(), b=$('pose-guide').getBoundingClientRect();
      const d=$('register-dialog');
      const cancel=$('cancel-register').getBoundingClientRect(), dialog=d.getBoundingClientRect();
      const result={width:innerWidth,dialogWidth:d.clientWidth,horizontalOverflow:d.scrollWidth>d.clientWidth,
        overlap:a.left<b.right&&a.right>b.left&&a.top<b.bottom&&a.bottom>b.top,
        cancelVisible:cancel.bottom<=dialog.bottom&&cancel.top>=dialog.top,
        cameraAboveGuide:b.top>=a.bottom,postureGreen:$('pose-guide').classList.contains('is-aligned'),
        progress:$('pose-hold-progress').value,icon:$('pose-current').getAttribute('src')};
      fetch('/result',{method:'POST',body:JSON.stringify(result)});
    },500);
    '''
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            ruta = self.path.split('?')[0]
            if ruta == '/':
                tipo = 'text/html; charset=utf-8'
                datos = ('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/app.css"><title>Guia simulada sin camara</title>'+dialogo+'<script src="/harness.js"></script></html>').encode()
            elif ruta == '/harness.js':
                tipo, datos = 'text/javascript; charset=utf-8', codigo.encode()
            elif ruta in ('/app.css', '/posturas.svg') or ruta.startswith('/imagenes/posturas/'):
                tipo = 'text/css' if ruta.endswith('css') else 'image/png' if ruta.endswith('.png') else 'image/svg+xml'
                datos = (STATIC/ruta[1:]).read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', tipo)
            self.send_header('Content-Length', str(len(datos)))
            self.end_headers()
            self.wfile.write(datos)
        def do_POST(self):
            resultados.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(204)
            self.end_headers()
    servidor = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    try:
        for nombre, ancho, alto in [('escritorio', 1200, 1000), ('celular', 390, 844)]:
            perfil = Path(tempfile.mkdtemp(prefix='.guia_edge_', dir=ROOT/'pruebas')).resolve()
            assert perfil.parent == (ROOT/'pruebas').resolve()
            destino = ROOT/'pruebas'/f'guia_{nombre}_simulada.png'
            try:
                proceso = subprocess.Popen([str(EDGE), '--headless=new', '--disable-gpu', '--no-first-run',
                    '--no-default-browser-check', '--disable-extensions', '--disable-background-networking',
                    f'--user-data-dir={perfil}', '--force-device-scale-factor=1', '--hide-scrollbars',
                    '--remote-debugging-port=0', 'about:blank'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                cliente = None
                try:
                    limite = time.monotonic()+15
                    while True:
                        try:
                            puerto = (perfil/'DevToolsActivePort').read_text().splitlines()[0]
                            int(puerto)
                            break
                        except (OSError, IndexError, ValueError):
                            pass
                        if proceso.poll() is not None or time.monotonic() > limite:
                            raise RuntimeError('Edge no pudo iniciar el entorno de prueba local.')
                        time.sleep(.1)
                    with urllib.request.urlopen(f'http://127.0.0.1:{puerto}/json', timeout=10) as r:
                        paginas = json.load(r)
                    cliente = DevTools(next(p['webSocketDebuggerUrl'] for p in paginas if p['type'] == 'page'))
                    cliente.llamar('Runtime.enable')
                    cliente.llamar('Emulation.setDeviceMetricsOverride', {'width': ancho, 'height': alto, 'deviceScaleFactor': 1, 'mobile': nombre == 'celular'})
                    cliente.llamar('Page.navigate', {'url': f'http://127.0.0.1:{servidor.server_port}/'})
                    limite = time.monotonic()+10
                    while len(resultados) < (1 if nombre == 'escritorio' else 2):
                        if time.monotonic() > limite:
                            raise RuntimeError('No se recibio el resultado JavaScript del dialogo.')
                        time.sleep(.1)
                    compilacion = cliente.llamar('Runtime.compileScript', {'expression': js, 'sourceURL': 'app.js', 'persistScript': False})
                    assert 'exceptionDetails' not in compilacion, compilacion
                    imagen = cliente.llamar('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
                    destino.write_bytes(base64.b64decode(imagen['data']))
                    comprobaciones = cliente.llamar('Runtime.evaluate', {'returnByValue': True, 'expression': '''(()=>{
                      const base={paso:2,completados:1,postura_ok:true,calidad_ok:true,lecturas_estables:3};
                      renderPosture({...base,pausado:true},true);
                      const pausa=!$('pose-guide').classList.contains('is-aligned')&&$('pose-hold-progress').value===0;
                      renderPosture({...base,calidad_ok:false,lecturas_estables:0},true);
                      const calidad=$('pose-guide').classList.contains('is-aligned')&&$('pose-hold-progress').value===0&&$('pose-hold-text').textContent.includes('calidad');
                      renderPosture(base,false);
                      const sinLectura=!$('pose-guide').classList.contains('is-aligned')&&$('pose-hold-progress').value===0;
                      renderPosture({...base,cancelado:true},true);
                      const cancelacion=$('pose-hold-progress').value===0&&$('pose-status').textContent.includes('cancelado');
                      renderPosture({...base,completo:true},true);
                      const finalizacion=$('pose-hold-progress').value===Number($('pose-hold-progress').max)&&$('pose-hold-text').textContent.includes('completadas');
                      renderPosture({...base,progreso_postura:.5},true);
                      const reintento=$('pose-guide').classList.contains('is-aligned');
                      const diagnosticoOculto=$('registration-diagnostics').hidden;
                      const imagenes=[...document.querySelectorAll('.pose-steps img,#pose-current')].every(i=>i.complete&&i.naturalWidth===1254&&getComputedStyle(i).objectFit==='contain');
                      return {pausa,calidad,sinLectura,cancelacion,finalizacion,reintento,diagnosticoOculto,imagenes};
                    })()'''})
                    assert all(comprobaciones['result']['value'].values()), comprobaciones
                    resultados[-1]['estados_js_verificados'] = comprobaciones['result']['value']
                finally:
                    if cliente is not None:
                        try:
                            cliente.llamar('Browser.close')
                        except (EOFError, OSError):
                            pass
                        cliente.socket.close()
                    try:
                        proceso.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proceso.kill()
                        proceso.wait()
            finally:
                assert perfil.parent == (ROOT/'pruebas').resolve()
                shutil.rmtree(perfil)
        assert len(resultados) == 2, resultados
        for resultado in resultados:
            assert not resultado['horizontalOverflow'] and not resultado['overlap'], resultado
            assert resultado['cancelVisible'], resultado
            assert resultado['postureGreen'] and resultado['progress'] == .5, resultado
            assert resultado['icon'] == '/imagenes/posturas/derecha.png', resultado
        assert resultados[1]['cameraAboveGuide'], resultados[1]
        assert resultados[1]['width'] == 390, resultados[1]
        (ROOT/'pruebas/verificacion_visual_guia.json').write_text(json.dumps(resultados, indent=2), encoding='utf-8')
        print(json.dumps(resultados))
    finally:
        servidor.shutdown()
        servidor.server_close()
        hilo.join()


if __name__ == '__main__':
    main()
