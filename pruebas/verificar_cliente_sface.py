"""Prueba DOM/JavaScript local en Edge headless. Sin servidor ni dispositivos."""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def verificar(ancho=1366, alto=768):
    html = (ROOT/'web_local/static/index.html').read_text(encoding='utf-8')
    js = (ROOT/'web_local/static/app.js').read_text(encoding='utf-8')
    # Ejecutar el JS real y su DOM; suprimir solo el arranque de polling/autenticacion.
    assert js.rstrip().endswith('restore();')
    js = js.rstrip()[:-len('restore();')]
    pruebas = r'''
window.fetch = async (url, options) => {
  window.requests.push([url, options]);
  return {ok:true, json:async()=>({ok:true})};
};
window.requests=[];
function check(ok,msg) { if(!ok) throw new Error(msg); }
async function pruebasSFace() {
  try {
    let s={fase:"detenido",actividad:null,modo:"demo",camara:false,rostros:0,
      motor_seleccionado:"sface",motor_activo:null,nombre:null,similitud_estimada:null,total:5,progreso:0,
      similitud_coseno:null,opciones:{reconocimiento:true,emociones:true,gestos:true},recorrido:{}};
    check($("recognition-engine").value==="sface","Default HTML SFace");
    render(s);
    check($("recognition-engine").value==="sface","Default servidor SFace");
    check($("registration-intro").textContent==="Escribe tu nombre y sigue los cinco pasos","Instruccion inicial unica");
    for (const id of ['test-registration','open-evaluation','registration-tip','registration-preparation','registration-confirmation']) check($(id).hidden,'Pantalla inicial sin extras: '+id);
    check(!$("recognition-engine").disabled,"Selector detenido");
    check($("recognition-engine").tagName==="INPUT","SFace fijo sin flecha");
    check($("evaluation-engine").tagName==="INPUT","Evaluacion SFace fija sin flecha");
    $("new-name").value="   "; $("new-name").dispatchEvent(new Event("input"));
    check($("begin-register").disabled,"Nombre vacio bloqueado");
    $("new-name").value="Persona_Prueba"; $("new-name").dispatchEvent(new Event("input"));
    check(!$("begin-register").disabled,"Captura habilitada sin comprobar configuracion");
    $("register-form").dispatchEvent(new Event("submit",{cancelable:true}));
    await new Promise(resolve=>setTimeout(resolve,50));
    check(window.requests.some(r=>r[0]==="/api/iniciar-recorrido"),"Inicio directo separado del alta");
    check(!window.requests.some(r=>r[0]==="/api/registro/preparacion"),"Sin paso manual de configuracion");
    s={...s,fase:"activo",actividad:"sistema",camara:true,rostros:1,
       motor_seleccionado:"sface",motor_activo:"sface",nombre:"Alison",similitud_coseno:.701};
    render(s);
    check($("recognition-engine").disabled,"Selector activo bloqueado");
    check($("result-similarity").textContent==="Similitud 70,1 %","Coseno correcto");
    check($("camera-recognition").textContent==="Alison \u2014 Similitud 70,1 %","Formato de video exacto");
    check($("recognition-engine-status").textContent.includes("SFace"),"Motor real");
    s={...s,nombre:"Desconocido",similitud_coseno:null};render(s);
    check($("result-name").textContent==="Desconocido","Rechazo");
    check($("result-similarity").hidden,"Limpieza coseno rechazo");
    s={...s,nombre:null,estado_reconocimiento:"varios_rostros",rostros:2};render(s);
    check($("person-note").textContent.includes("Una persona a la vez"),"Politica varios");
    s={...s,nombre:"Alexa",similitud_coseno:.5,rostros:0};render(s);
    check($("result-name").textContent==="Sin lectura","Sin rostro");
    check($("result-similarity").hidden,"No score sin rostro");
    s={...s,rostros:1,opciones:{...s.opciones,reconocimiento:false}};render(s);
    check($("result-name").textContent==="Desactivado","Desactivacion");
    check($("result-similarity").hidden,"No score desactivado");
    s={...s,motor_seleccionado:"lbph",motor_activo:"lbph",similitud_estimada:.45,
      opciones:{...s.opciones,reconocimiento:true}};render(s);
    check($("result-similarity").textContent==="","Motor ajeno sin formato SFace");
    s={...s,fase:"detenido",actividad:null,camara:false,motor_activo:null,nombre:null};render(s);
    check(!$("recognition-engine").disabled,"Retorno selector detenido");
    check($("result-similarity").hidden,"Limpieza parada");
    $('evaluation-engine').value='sface';
    $('evaluation-known').add(new Option('Alison','Alison'));
    $('evaluation-known').value='Alison';
    $('evaluation-type').value='registrada';
    evaluationFields();
    check($('evaluation-alias').required && !$('evaluation-alias').disabled,'Alias de prueba');
    $('evaluation-alias').value='Alison_Prueba';
    $('evaluation-consent').checked=true;
    $('evaluation-form').dispatchEvent(new Event('submit',{cancelable:true}));
    await new Promise(resolve=>setTimeout(resolve,50));
    let ultima=window.requests.at(-1);
    let datos=JSON.parse(ultima[1].body).evaluacion;
    check(ultima[0]==='/api/evaluar-duplicados','Ruta evaluacion separada');
    check(datos.motor==='sface' && datos.identificador==='Alison' && datos.nombre_propuesto==='Alison_Prueba','Datos Alison alias');
    evaluationView=true;
    s={...s,evaluacion_registro:true,resultado_evaluacion:{motor:'sface',carpeta:'solo-prueba-dom',
      participante_declarado:datos,analisis:{validas:5,descripcion:'Sin criterio validado. Sin decision de alta.'},
      personas:[{nombre:'Alison',referencias:5,posturas:{frente:.65,derecha:.65,izquierda:.65,arriba:.65,abajo:.65},minima:.65,media:.65,maxima:.65}],
      lecturas:['frente','derecha','izquierda','arriba','abajo'].map(postura=>({postura,primera:'Alison',segunda:'Alexa',separacion:.2,error:''}))}};
    render(s);
    check($('evaluation-result-heading').textContent==='Similitud por postura','Titulo simple');
    check($('evaluation-rows').textContent.includes('0.6500'),'Cuatro decimales sin porcentaje');
    check($('evaluation-rows').querySelectorAll('td').length===9,'Identidad, cinco posturas y tres estadisticos');
    check($('evaluation-folder').textContent.includes('puntuaciones.csv') && $('evaluation-folder').textContent.includes('resumen.md'),'Archivos de evaluacion indicados');
    check($('evaluation-analysis').textContent.includes('5/5') && $('evaluation-analysis').textContent.includes('Alison'),'Resumen breve');
    check($('evaluation-ranking').textContent.includes('Alison: 5'),'Referencias visibles');
    $('evaluation-type').value='no_registrada';evaluationFields();
    check($('evaluation-alias').disabled && !$('evaluation-alias').required,'Externo sin alias');
    $('evaluation-id').value='EXTERNA_001';
    $('evaluation-form').dispatchEvent(new Event('submit',{cancelable:true}));
    await new Promise(resolve=>setTimeout(resolve,50));
    datos=JSON.parse(window.requests.at(-1)[1].body).evaluacion;
    check(datos.tipo==='no_registrada' && datos.nombre_propuesto==='','Datos externo');
    $("app").hidden=false;
    if (!$("register-dialog").open) $("register-dialog").showModal();
    s={...s,fase:"iniciando",actividad:"registro",prueba_registro:true,evaluacion_registro:false,resultado_evaluacion:null,recorrido:{paso:1},registro_nombre:"Persona Nueva"};
    render(s);
    check(!$("registration-active").hidden,"Guia visible desde el inicio");
    const video=document.querySelector('.registration-preview').getBoundingClientRect();
    const guia=$("pose-guide").getBoundingClientRect();
    check(video.width>guia.width*1.4,"Camara grande a la izquierda");
    check(video.left<guia.left,"Robot a la derecha");
    const antes=$("register-dialog").getBoundingClientRect();
    for (let paso=1;paso<=5;paso++) {
      render({...s,recorrido:{paso}});
      const ahora=$("register-dialog").getBoundingClientRect();
      check(Math.abs(antes.top-ahora.top)<1 && Math.abs(antes.height-ahora.height)<1,"Ventana fija sin saltos");
      const videoAhora=document.querySelector('.registration-preview').getBoundingClientRect();
      check(Math.abs(videoAhora.top-video.top)<1 && Math.abs(videoAhora.height-video.height)<1,"Video estable entre posturas");
      const cancelar=$("cancel-register").getBoundingClientRect();
      check(cancelar.top>=0 && cancelar.bottom<=window.innerHeight,`Cancelar completo sin rueda: ${cancelar.top},${cancelar.bottom},viewport=${window.innerHeight},dialog=${ahora.top}/${ahora.bottom},active=${$("registration-active").clientHeight}`);
      for (const item of document.querySelectorAll('[data-pose]')) {
        const r=item.getBoundingClientRect();
        check(r.top>=0 && r.bottom<=window.innerHeight,"Cinco pasos completos");
      }
      check(getComputedStyle($("register-dialog")).overflowY==="hidden","Ventana sin rueda");
      const instruccion=$("registration-instruction").getBoundingClientRect();
      check(instruccion.top>=ahora.top && instruccion.bottom<=ahora.bottom,"Instruccion completa");
      check(video.height>100 && video.top>=ahora.top && video.bottom<=ahora.bottom,"Video visible dentro del espacio disponible");
    }
    renderPosture({paso:1,postura_ok:true,calidad_ok:true,progreso_postura:.5},true);
    check($("pose-hold-progress").max===1 && $("pose-hold-progress").value===.5,"Barra con progreso real");
    const barra=$("pose-hold-progress").getBoundingClientRect();
    check(barra.height>=10 && barra.width>20,"Barra visible sin comprimirse");
    check(barra.bottom<=$("pose-guide").getBoundingClientRect().bottom,"Barra dentro de guia visible");
    renderPosture({paso:1,postura_ok:false,calidad_ok:true,progreso_postura:.5},true);
    check($("pose-hold-progress").value===0,"Perder postura reinicia barra");
    renderPosture({paso:1,postura_ok:true,calidad_ok:false,progreso_postura:.5},true);
    check($("pose-hold-progress").value===0,"Calidad insuficiente reinicia barra");
    render({...s,fase:"detenido",actividad:null,resultado_registro:{estado:"guardado",galeria:"temporal-ui",mensaje:"Registro guardado correctamente"}});
    check($("registration-summary").hidden,"Resultado no se muestra debajo del registro");
    const duplicado="Esta persona ya est\u00e1 registrada como Alison. No se guard\u00f3 un nuevo registro";
    render({...s,fase:"detenido",actividad:null,resultado_registro:{estado:"duplicado",persona:"Alison",mensaje:duplicado}});
    check($("registration-summary").hidden,"Sin resultado residual");
    const ventana=$("registration-verification");
    let altaState={...s,alta_sface:true,prueba_registro:false,evaluacion_registro:false,
      registro_id:'modal-ui-1',registro_nombre:'Alison Tinoco',fase:'activo',actividad:'registro',
      resultado_registro:null,recorrido:{paso:4,completo:false}};
    render(altaState);
    check(!ventana.open,'No verificar antes de la quinta captura');
    altaState={...altaState,recorrido:{paso:5,completo:true}};
    render(altaState);
    check(ventana.open,'Ventana encima del registro tras quinta captura');
    check($('verification-message').textContent==='Verificando el registro\u2026','Carga exacta');
    check(!$('verification-loading').hidden && $('verification-understood').hidden,'Carga sin confirmacion prematura');
    check($('verification-robot').getAttribute('src')==='/imagenes/posturas/frente.png','Robot existente');
    const caja=ventana.getBoundingClientRect();
    check(Math.abs(caja.left+caja.width/2-window.innerWidth/2)<2,'Ventana centrada horizontal');
    check(Math.abs(caja.top+caja.height/2-window.innerHeight/2)<2,'Ventana centrada vertical');
    const escape=new Event('cancel',{cancelable:true});ventana.dispatchEvent(escape);
    check(escape.defaultPrevented && ventana.open,'Escape no cierra la verificacion');
    $('verification-understood').click();
    check(ventana.open,'No cerrar mientras procesa');
    render({...altaState,instruccion:'Registro guardado correctamente'});
    check($('verification-message').textContent==='Verificando el registro\u2026','Texto suelto no confirma guardado');
    altaState={...altaState,resultado_registro:{estado:'guardado',persona:'Alison Tinoco',galeria:'v-confirmada'}};
    render(altaState);
    check($('verification-message').textContent==='Alison Tinoco qued\u00f3 registrada correctamente','Nombre realmente guardado');
    check($('verification-loading').hidden && !$('verification-understood').hidden,'Carga sustituida por resultado');
    const icono=$('verification-robot').getBoundingClientRect();
    const texto=$('verification-message').getBoundingClientRect();
    check(icono.right<=texto.left,'Robot a la izquierda del mensaje');
    check($('verification-understood').getBoundingClientRect().top>=texto.bottom,'Entendido debajo del mensaje');
    check(getComputedStyle($('verification-loading')).display==='none','Animacion retirada del resultado');
    render({...altaState,resultado_registro:null,fase:'detenido',actividad:null});
    check(ventana.open && $('verification-message').textContent==='Alison Tinoco qued\u00f3 registrada correctamente','Resultado permanece entre actualizaciones');
    $('verification-understood').click();render(altaState);
    check(!ventana.open,'Entendido cierra y el polling no reabre');
    altaState={...altaState,registro_id:'modal-ui-2',resultado_registro:null};render(altaState);
    check(ventana.open && !$('verification-loading').hidden,'Otro intento inicia su propia carga');
    render({...altaState,resultado_registro:{estado:'duplicado',persona:'Alison',nombre_solicitado:'Alison Tinoco'}});
    check($('verification-message').textContent==='La persona que intentas registrar como Alison Tinoco ya est\u00e1 registrada como Alison. No se puede registrar otra vez','Ambos nombres en duplicado');
    $('verification-understood').click();
    for (const estado of ['dudoso','error','cancelado']) {
      altaState={...altaState,registro_id:'modal-ui-'+estado,resultado_registro:{estado}};
      render(altaState);
      check(ventana.open && !$('verification-title').hidden && $('verification-title').textContent==='Registro no completado','Fallo con titulo claro');
      $('verification-understood').click();render(altaState);
      check(!ventana.open,'Fallo reconocido no reaparece');
    }
    altaState={...altaState,registro_id:'modal-ui-incompleto',resultado_registro:{estado:'guardado',persona:'Alison Tinoco'}};
    render(altaState);
    check(!$('verification-title').hidden && $('verification-message').textContent.includes('Revisa Personas'),'Sin confirmacion completa no hay exito');
    $('verification-understood').click();
    altaState={...altaState,registro_id:'revision-manual',fase:'activo',resultado_registro:{estado:'dudoso',revision_manual:true,
      indicacion:'Revisa y confirma',puntuaciones:Array.from({length:5},()=>({Alison:.42,Alexa:.21}))}};
    render(altaState);
    check(!$('verification-review').hidden && $('verification-understood').hidden,'Revision solo si servidor permite');
    check($('verification-approve').disabled,'Confirmacion expresa requerida');
    check($('verification-scores').textContent.includes('Alison: 0.4200'),'Puntuaciones de cinco fotos visibles');
    $('verification-confirm').checked=true;$('verification-confirm').dispatchEvent(new Event('change'));
    check(!$('verification-approve').disabled,'Puede confirmar persona nueva');
    render(altaState);
    check($('verification-confirm').checked,'Polling no borra confirmacion pendiente');
    render({...altaState,resultado_registro:{estado:'cancelado',indicacion:'Revision vencida'}});
    check($('verification-review').hidden && !$('verification-understood').hidden,'Vencimiento reemplaza revision');
    $('verification-understood').click();
    altaState={...altaState,registro_id:'repetir-postura',resultado_registro:{estado:'dudoso',repetir_postura:3,indicacion:'Repite solo postura 3'}};
    render(altaState);
    check(!$('verification-retake').hidden && $('verification-review').hidden,'Repeticion independiente de revision');
    check(!$('verification-cancel').hidden,'Cancelacion disponible mientras espera');
    render({...altaState,resultado_registro:{estado:'duplicado',persona:'Alison'}});
    check($('verification-review').hidden && $('verification-retake').hidden,'Duplicado sin aprobacion manual');
    $('verification-understood').click();
    document.documentElement.dataset.sfaceTest="OK";
  } catch(error) {
    document.documentElement.dataset.sfaceTest="ERROR";
    document.documentElement.dataset.error=error.stack;
  }
}
pruebasSFace();
'''
    html = html.replace('<script src="/app.js" defer></script>', '')
    html = html.replace('</head>', '<style>'+(ROOT/'web_local/static/app.css').read_text(encoding='utf-8')+'</style></head>')
    html = html.replace('</body>', '<script>'+js+'\n'+pruebas+'</script></body>')
    edge = Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    if not edge.exists():
        raise RuntimeError('Edge no disponible; no se ha realizado la prueba DOM')
    with tempfile.TemporaryDirectory(prefix='claudia_sface_dom_',ignore_cleanup_errors=True) as td:
        p = Path(td)/'test.html'; p.write_text(html,encoding='utf-8')
        r = subprocess.run([str(edge),'--headless','--disable-gpu','--no-first-run',
            '--disable-background-networking','--disable-sync',
            '--user-data-dir='+str(Path(td)/'perfil'), '--virtual-time-budget=1500',
            f'--window-size={ancho},{alto}', '--dump-dom',p.as_uri()],capture_output=True,text=True,encoding='utf-8',errors='replace',
            timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
        if not re.search(r'<html[^>]*data-sface-test="OK"',r.stdout):
            raise AssertionError('DOM: '+str(re.findall(r'data-error="([^"]*)',r.stdout))+' '+r.stderr[-600:])
    informe = 'Edge headless: HTML, CSS y JS reales OK. Pantalla inicial simplificada, barra visible con progreso real y reinicios. Ventana fija y robot animado a la izquierda del aviso: exito solo tras resultado confirmado, ambos nombres del duplicado, titulo Registro no completado e indicacion breve, Entendido, Escape bloqueado y sin reapertura por polling. SFace y tabla conservados. Fetch y resultados simulados, sin camara ni Arduino. No valida fluidez real.\n'
    destino=ROOT/'pruebas/VERIFICACION_CLIENTE_SFACE.txt'
    with destino.open('w' if (ancho,alto)==(1366,768) else 'a',encoding='utf-8') as archivo:
        archivo.write(f'{ancho}x{alto}: '+informe)
    print(informe)


if __name__=='__main__':
    for dimensiones in ((1366,768),(1024,600),(800,600),(540,720)):
        verificar(*dimensiones)
