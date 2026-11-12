"""Web completa con API simulada; nunca ejecuta el trabajador ni hardware."""
import base64
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request

from tests.vista_guia_simulada import DevTools, EDGE, ROOT, STATIC
from web_local.control import Control


def main():
    estado=Control.vacio()
    peticiones=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def responder(self,datos,tipo='application/json'):
            if isinstance(datos,dict): datos=json.dumps(datos,ensure_ascii=False).encode()
            self.send_response(200); self.send_header('Content-Type',tipo)
            self.send_header('Content-Length',str(len(datos))); self.end_headers(); self.wfile.write(datos)
        def do_GET(self):
            ruta=self.path.split('?')[0]
            if ruta=='/api/estado': return self.responder({'estado':estado,'csrf':'simulado'})
            if ruta=='/api/personas': return self.responder({'personas':[{'nombre':'Persona Exacta','fotos':5},{'nombre':'Otra Persona','fotos':5}]})
            if ruta=='/api/registro/preparacion': return self.responder({'permitido':False,'mensaje':'Registro bloqueado para la evaluación.'})
            if ruta=='/api/camara.jpg':
                self.send_response(204); self.end_headers(); return
            p=(STATIC/('index.html' if ruta=='/' else ruta.lstrip('/'))).resolve()
            if not p.is_relative_to(STATIC.resolve()) or not p.is_file(): self.send_error(404); return
            tipo='image/png' if p.suffix=='.png' else 'text/javascript' if p.suffix=='.js' else 'text/css' if p.suffix=='.css' else 'text/html' if p.suffix=='.html' else 'image/svg+xml'
            self.responder(p.read_bytes(),tipo+'; charset=utf-8' if not tipo.startswith('image') else tipo)
        def do_POST(self):
            datos=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            peticiones.append((self.path,datos))
            if self.path=='/api/probar-registro':
                estado.update(fase='iniciando',actividad='registro',camara=False,prueba_registro=True,registro_nombre='Prueba sin alta',recorrido={})
            if self.path=='/api/evaluar-duplicados':
                estado.update(fase='activo',actividad='registro',camara=True,evaluacion_registro=True,
                    prueba_registro=False,resultado_evaluacion=None,registro_nombre='Evaluación: '+datos['evaluacion']['identificador'],
                    recorrido={'paso':1,'completados':0,'progreso_postura':0})
            if self.path=='/api/detener':
                estado.update(fase='detenido',actividad=None,camara=False,recorrido={'cancelado':True})
            self.responder({'ok':True})
    servidor=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    hilo=threading.Thread(target=servidor.serve_forever,daemon=True); hilo.start()
    resultados=[]
    try:
        for nombre,ancho,alto in [('escritorio',1200,1000),('celular',390,844)]:
            estado.clear();estado.update(Control.vacio())
            perfil=Path(tempfile.mkdtemp(prefix='.eval_edge_',dir=ROOT/'pruebas')).resolve()
            proceso=subprocess.Popen([str(EDGE),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-extensions','--disable-background-networking',f'--user-data-dir={perfil}','--remote-debugging-port=0','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            cliente=None
            try:
                limite=time.monotonic()+15
                while not (perfil/'DevToolsActivePort').exists():
                    if time.monotonic()>limite: raise RuntimeError('No inició Edge')
                    time.sleep(.1)
                puerto=(perfil/'DevToolsActivePort').read_text().splitlines()[0]
                with urllib.request.urlopen(f'http://127.0.0.1:{puerto}/json') as r: paginas=json.load(r)
                cliente=DevTools(next(p['webSocketDebuggerUrl'] for p in paginas if p['type']=='page'))
                cliente.llamar('Runtime.enable')
                cliente.llamar('Emulation.setDeviceMetricsOverride',{'width':ancho,'height':alto,'deviceScaleFactor':1,'mobile':nombre=='celular'})
                cliente.llamar('Page.navigate',{'url':f'http://127.0.0.1:{servidor.server_port}/'})
                limite=time.monotonic()+10
                while True:
                    listo=cliente.llamar('Runtime.evaluate',{'expression':"typeof state !== 'undefined' && authenticated",'returnByValue':True})
                    if listo.get('result',{}).get('value'): break
                    if time.monotonic()>limite: raise RuntimeError('No cargó app.js')
                    time.sleep(.1)
                def evaluar(codigo):
                    r=cliente.llamar('Runtime.evaluate',{'expression':codigo,'returnByValue':True,'awaitPromise':True})
                    if 'exceptionDetails' in r: raise AssertionError(r)
                    return r['result'].get('value')
                comprobaciones=evaluar('''(async()=>{
                    await openRegister();
                    const bloqueoInicial=$('begin-register').disabled;
                    $('test-registration').click();await new Promise(r=>setTimeout(r,400));
                    await Promise.all([...document.querySelectorAll('.pose-steps img,#pose-current')].map(i=>i.decode()));
                    const robotsAlIniciar=state.fase==='iniciando' && !$('registration-active').hidden && [...document.querySelectorAll('.pose-steps img,#pose-current')].every(i=>i.naturalWidth===1254 && i.getBoundingClientRect().width>0 && getComputedStyle(i).objectFit==='contain');
                    const pruebaExplicita=$('registration-heading').textContent==='Prueba de captura sin guardar' && $('registration-summary').textContent.includes('no se conservar\u00e1n');
                    const basePrueba=state;
                    render({...basePrueba,fase:'detenido',actividad:null,recorrido:{completo:true,completados:5,paso:5}});
                    const finalSinAlta=$('registration-summary').textContent.includes('No se guardaron fotograf\u00edas ni se cre\u00f3 un registro.');
                    render({...basePrueba,prueba_registro:false});
                    const registroDistinto=$('registration-heading').textContent==='Registro en cinco pasos';
                    await action('/api/detener');
                    const modos={bloqueoInicial,robotsAlIniciar,pruebaExplicita,finalSinAlta,registroDistinto};
                    if (!Object.values(modos).every(Boolean)) throw Error(JSON.stringify(modos));
                    $('open-evaluation').click();
                    await new Promise(r=>setTimeout(r,300));
                    const aviso=!$('evaluation-setup').hidden && $('evaluation-setup').textContent.includes('Se conservarán cinco');
                    const consentimiento=!$('evaluation-form').checkValidity();
                    $('evaluation-known').value='Persona Exacta';$('evaluation-consent').checked=true;
                    const nombreExacto=$('evaluation-form').checkValidity();
                    $('evaluation-form').requestSubmit();await new Promise(r=>setTimeout(r,400));
                    const activo=state.evaluacion_registro && !$('registration-active').hidden && $('register-form').hidden && $('registration-tip').hidden && $('registration-heading').textContent==='Evaluación de duplicados';
                    const diagnosticoOculto=$('registration-diagnostics').hidden;
                    const progreso={...state,recorrido:{paso:2,completados:1,progreso_postura:.5,postura_ok:true,calidad_ok:true,confirmacion:'Captura del paso 1 confirmada.'}};
                    render(progreso);const captura=$('pose-hold-progress').value===.5 && $('registration-confirmation').textContent.includes('confirmada');
                    render({...progreso,recorrido:{...progreso.recorrido,pausado:true}});
                    const pausa=!$('retry-registration').hidden;
                    $('retry-registration').click();await new Promise(r=>setTimeout(r,300));
                    $('cancel-register').click();await new Promise(r=>setTimeout(r,300));
                    const cancelado=$('registration-summary').textContent.includes('cancelado');
                    $('open-evaluation').click();await new Promise(r=>setTimeout(r,300));
                    $('evaluation-type').value='no_registrada';evaluationFields();$('evaluation-id').value='PRUEBA_001';$('evaluation-consent').checked=true;
                    const noRegistrada=$('evaluation-known-field').hidden && $('evaluation-form').checkValidity();
                    $('evaluation-form').requestSubmit();await new Promise(r=>setTimeout(r,400));
                    return {...modos,aviso,consentimiento,nombreExacto,activo,diagnosticoOculto,captura,pausa,cancelado,noRegistrada};
                })()''')
                assert all(comprobaciones.values()),comprobaciones
                # Finalización sintética: la persistencia y las distancias reales se prueban en unittest.
                estado.update(fase='detenido',actividad=None,camara=False,recorrido={'paso':5,'completados':5,'completo':True},resultado_evaluacion={
                    'carpeta':'evaluacion_duplicados/recorrido_SIMULADO','participante_declarado':{'tipo':'no_registrada','identificador':'PRUEBA_001'},
                    'personas':[{'nombre':n,'posturas':dict.fromkeys(['frente','derecha','izquierda','arriba','abajo'],d),'minima':d,'media':d,'maxima':d} for n,d in [('Persona Exacta',71.2),('Otra Persona',83.4)]]})
                evaluar('sync()')
                fin=evaluar('''(()=>{const d=$('register-dialog');return {resultado:!$('evaluation-result').hidden,filas:$('evaluation-rows').children.length===2,finalizacion:$('registration-summary').textContent.includes('solo en evaluación'),sinAlta:$('begin-register').disabled,modoSeparado:$('register-form').hidden && $('registration-heading').textContent==='Evaluación de duplicados',tipOculto:$('registration-tip').hidden,sinDesbordamiento:d.scrollWidth===d.clientWidth};})()''')
                assert all(fin.values()),fin
                captura=cliente.llamar('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
                (ROOT/'pruebas'/f'inicio_modos_{nombre}_simulada.png').write_bytes(base64.b64decode(captura['data']))
                vuelta=evaluar("(()=>{$('exit-evaluation').click();return !$('register-form').hidden && $('evaluation-result').hidden && $('begin-register').disabled;})()")
                assert vuelta
                resultados.append({'viewport':[ancho,alto],**comprobaciones,**fin,'volverAlRegistro':vuelta})
            finally:
                if cliente:
                    try: cliente.llamar('Browser.close')
                    except (OSError,EOFError): pass
                    cliente.socket.close()
                try: proceso.wait(timeout=5)
                except subprocess.TimeoutExpired: proceso.kill();proceso.wait()
                if perfil.parent==(ROOT/'pruebas').resolve(): shutil.rmtree(perfil)
        assert any(r=='/api/reintentar-registro' for r,_ in peticiones)
        assert all(d['evaluacion']['acepta_conservar'] for r,d in peticiones if r=='/api/evaluar-duplicados')
        assert not any(r=='/api/registrar' for r,_ in peticiones)
        (ROOT/'pruebas/verificacion_inicio_modos.json').write_text(json.dumps(resultados,indent=2),encoding='utf-8')
        print(json.dumps(resultados))
    finally:
        servidor.shutdown();servidor.server_close();hilo.join()


if __name__=='__main__': main()
