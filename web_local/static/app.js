"use strict";
const $ = id => document.getElementById(id);
const postureGuide = [
  ["frente", "Mira de frente"], ["derecha", "Gira hacia TU derecha"],
  ["izquierda", "Gira hacia TU izquierda"], ["arriba", "Levanta un poco el rostro"],
  ["abajo", "Baja un poco el rostro"]
];
function renderPosture(r, active) {
  const index = Math.max(0, Math.min(4, (r.paso || 1) - 1));
  const [symbol, title] = postureGuide[index];
  $("pose-current").src = `/imagenes/posturas/${symbol}.png`;
  $("pose-current").alt = title;
  $("pose-title").textContent = title;
  const aligned = active && !r.cancelado && !r.pausado && (r.postura_ok || r.completo);
  $("pose-guide").classList.toggle("is-aligned", Boolean(aligned));
  $("pose-status").textContent = !active ? "Sin lectura" : r.cancelado ? "Recorrido cancelado" : r.pausado ? "Paso pausado" : r.completo ? "Recorrido completo" : aligned ? "Postura alcanzada" : "Ajusta suavemente la postura";
  const required = Math.max(1, r.lecturas_necesarias || 6);
  const ratio = Number.isFinite(r.progreso_postura) ? r.progreso_postura : (r.lecturas_estables || 0)/required;
  const stable = active && !r.pausado && !r.cancelado ? (r.completo ? 1 : r.postura_ok && r.calidad_ok ? Math.max(0,Math.min(1,ratio)) : 0) : 0;
  $("pose-hold-progress").max = 1;
  $("pose-hold-progress").value = stable;
  $("pose-hold-text").textContent = r.completo ? "Capturas completadas" : aligned && !r.calidad_ok ? "La captura espera una imagen de calidad." : "Mantén el rostro quieto";
  document.querySelectorAll("[data-pose]").forEach(item => {
    const step = Number(item.dataset.pose);
    item.classList.toggle("current", step === index && !r.completo);
    item.classList.toggle("done", step < (r.completados || 0));
    if (step === index && !r.completo) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  });
  $("registration-diagnostics").hidden = !r.diagnostico;
  $("registration-debug").textContent = r.diagnostico ? JSON.stringify(r.diagnostico, null, 2) : "Todavia no hay lecturas.";
}
let csrf = "", authenticated = false, state = null, lastNotice = 0, imageUrl = null;
let toastTimer, peopleTimer = 0, lastFrameAt = 0, offline = false, actionBusy = false;
let deleteName = null;
let pendingPhysical = null;
let evaluationSetup = false;
let evaluationView = false;
let evaluationResultRendered = null;
let registroActualizado = null;
let verificacionVisible = null;
const verificacionesEntendidas = new Set();

function renderSimilitudesRegistro(lecturas) {
  const destino=$("verification-scores");
  destino.replaceChildren();
  if (!Array.isArray(lecturas) || lecturas.length!==5 || !lecturas[0] || typeof lecturas[0]!=='object') return false;
  const nombres=Object.keys(lecturas[0]);
  if (!nombres.length || !lecturas.every(s=>s && typeof s==='object' && Object.keys(s).length===nombres.length &&
      nombres.every(n=>Number.isFinite(s[n]) && s[n]>=-1 && s[n]<=1))) return false;
  const posturas=['Frente','Derecha','Izquierda','Arriba','Abajo'];
  nombres.sort((a,b)=>lecturas.reduce((total,s)=>total+s[b]-s[a],0) || a.localeCompare(b));
  function comparar(nombre) {
    const bloque=document.createElement('section');bloque.className='comparison-person';
    const titulo=document.createElement('h3');titulo.textContent=`Similitud con ${nombre}`;bloque.append(titulo);
    lecturas.forEach((scores,i)=>{
      const fila=document.createElement('div');fila.className='comparison-pose';
      const postura=document.createElement('span');postura.textContent=posturas[i];
      const valor=document.createElement('span');valor.className='comparison-value';
      const porcentaje=(scores[nombre]*100).toFixed(1).replace('.',',')+' %';valor.textContent=porcentaje;
      const barra=document.createElement('progress');barra.max=100;barra.value=Math.max(0,scores[nombre]*100);
      barra.setAttribute('aria-label',`Similitud con ${nombre}, ${posturas[i]}: ${porcentaje}`);
      fila.append(postura,barra,valor);bloque.append(fila);
    });
    return bloque;
  }
  const resumen=document.createElement('p');resumen.className='comparison-summary';
  resumen.textContent='5/5 posturas. Primero se muestra la mayor similitud media.';destino.append(resumen,comparar(nombres[0]));
  if (nombres.length>1) {
    const detalles=document.createElement('details');const titulo=document.createElement('summary');titulo.textContent='Ver detalles';
    detalles.append(titulo);nombres.slice(1).forEach(n=>detalles.append(comparar(n)));destino.append(detalles);
  }
  const nota=document.createElement('p');nota.className='comparison-summary';nota.textContent='Similitud, no probabilidad de identidad.';destino.append(nota);
  return true;
}

function renderVerificacionRegistro(s) {
  const dialogo = $("registration-verification");
  // El resultado ya mostrado se conserva aunque lleguen más actualizaciones.
  if (dialogo.open && verificacionVisible?.final &&
      (!verificacionVisible.recuperable || verificacionVisible.firma === JSON.stringify(s.resultado_registro))) return;
  if (!s.alta_sface || s.evaluacion_registro || s.prueba_registro) return;
  const resultado = s.resultado_registro;
  const id = resultado?.registro_id || s.registro_id;
  if (!id || verificacionesEntendidas.has(id)) return;
  if (!resultado && !s.recorrido?.completo && s.fase !== 'error' && !dialogo.open) return;
  if (verificacionVisible && verificacionVisible.id !== id && dialogo.open) return;
  verificacionVisible ||= {id, final:false};
  if (verificacionVisible.id !== id) verificacionVisible = {id, final:false};
  const terminado = Boolean(resultado) || s.fase === 'error' || s.fase === 'detenido';
  let mensaje = 'Verificando el registro…';
  let fallo = false;
  if (terminado) {
    fallo = true;
    mensaje = resultado?.indicacion || 'Intenta de nuevo.';
    if (resultado?.estado === 'guardado' && resultado.persona && resultado.galeria) {
      mensaje = `${resultado.persona} quedó registrada correctamente`;
      fallo = false;
    } else if (resultado?.estado === 'duplicado' && resultado.persona) {
      const solicitado = resultado.nombre_solicitado || s.registro_nombre;
      mensaje = `La persona que intentas registrar como ${solicitado} ya está registrada como ${resultado.persona}. No se puede registrar otra vez`;
      fallo = false;
    } else if (resultado?.estado === 'guardado') {
      mensaje = 'No se recibió la confirmación completa. Revisa Personas antes de intentarlo de nuevo.';
    }
  }
  verificacionVisible.final = terminado;
  const revision = Boolean(resultado?.revision_manual);
  const repetir = Boolean(resultado?.repetir_postura);
  verificacionVisible.recuperable = revision || repetir;
  verificacionVisible.firma = JSON.stringify(resultado);
  $("verification-review").hidden = !revision;
  $("verification-retake").hidden = !repetir;
  $("verification-cancel").hidden = !(revision || repetir);
  $("verification-confirm").checked = false;
  $("verification-approve").disabled = true;
  $("verification-comparison").hidden = !renderSimilitudesRegistro(resultado?.puntuaciones);
  $("verification-message").textContent = mensaje;
  $("verification-title").hidden = !fallo;
  $("verification-loading").hidden = terminado;
  $("verification-understood").hidden = !terminado || revision || repetir;
  dialogo.setAttribute('aria-busy', String(!terminado));
  if (!dialogo.open) dialogo.showModal();
  if (terminado && !revision && !repetir) $("verification-understood").focus({preventScroll:true});
}

$("verification-confirm").addEventListener('change',()=> {
  $("verification-approve").disabled = !$("verification-confirm").checked;
});
async function resolverRegistro(manual) {
  const id=verificacionVisible?.id;
  if (!id || (manual && !$("verification-confirm").checked)) return;
  const ok=await action(manual ? '/api/registro/confirmar-nueva' : '/api/registro/repetir-postura',
    {registro_id:id,confirmacion:manual});
  if (ok) {
    verificacionVisible=null;
    if (manual) renderVerificacionRegistro({alta_sface:true,registro_id:id,recorrido:{completo:true},fase:'activo'});
    else $("registration-verification").close();
  }
}
$("verification-approve").addEventListener('click',()=>resolverRegistro(true));
$("verification-retake").addEventListener('click',()=>resolverRegistro(false));
$("verification-cancel").addEventListener('click',async()=> {
  if (await action('/api/detener')) {
    verificacionVisible=null;
    $("registration-verification").close();
  }
});

$("registration-verification").addEventListener('cancel', event => event.preventDefault());
$("verification-understood").addEventListener('click', () => {
  if (!verificacionVisible?.final) return;
  verificacionesEntendidas.add(verificacionVisible.id);
  if (verificacionesEntendidas.size > 20) verificacionesEntendidas.delete(verificacionesEntendidas.values().next().value);
  verificacionVisible = null;
  $("registration-verification").close();
});
function evaluationFields() {
  const known = $('evaluation-type').value === 'registrada';
  const sface = $('evaluation-engine').value === 'sface';
  $('evaluation-alias-field').hidden = !known || !sface;
  $('evaluation-alias').disabled = !known || !sface;
  $('evaluation-alias').required = known && sface;
  $('evaluation-description').textContent = 'Sigue las cinco posturas. La evaluacion no crea registros.';
  $('evaluation-known-field').hidden = !known;
  $('evaluation-unknown-field').hidden = known;
  $('evaluation-known').disabled = !known;
  $('evaluation-known').required = known;
  $('evaluation-id').disabled = known;
  $('evaluation-id').required = !known;
}
function renderEvaluation(s, registering) {
  const evaluating = Boolean(s.evaluacion_registro);
  if (evaluating && registering) evaluationView = true;
  $('registration-heading').textContent = evaluationView ? 'Evaluaci\u00f3n de duplicados' : 'Registro en cinco pasos';
  $('evaluation-setup').hidden = registering || !evaluationSetup;
  $('open-evaluation').hidden = true;
  for (const id of ['registration-intro','registration-preparation','check-registration','test-registration','register-form','registration-tip']) {
    if (evaluationView) $(id).hidden = true;
    else if (id === 'registration-tip') $(id).hidden = true;
  }
  $('begin-evaluation').disabled = registering || actionBusy;
  const result = s.resultado_evaluacion;
  $('evaluation-result').hidden = !result || !evaluationView;
  $('exit-evaluation').hidden = !evaluationView || registering || evaluationSetup;
  $('registration-summary').hidden = true;
  if (result && result.carpeta !== evaluationResultRendered) {
    evaluationResultRendered = result.carpeta;
    const isSFace = result.motor === 'sface';
    $('evaluation-result-heading').textContent = 'Similitud por postura';
    const alias = result.participante_declarado.nombre_propuesto;
    $('evaluation-result-note').textContent = `Participante declarado: ${result.participante_declarado.identificador}${alias ? " como " + alias : ""}. ${isSFace ? "Mayor coseno indica mayor semejanza. Sin umbral de duplicados ni decision de alta." : "Menor distancia LBPH indica mayor semejanza. Sin umbrales ni decision de identidad."} No se creo un registro.`;
    $('evaluation-folder').textContent = `Carpeta: ${result.carpeta}. ${isSFace ? "Cinco PNG originales, puntuaciones.csv y resumen.md." : "Cinco JPEG, resumen.json y distancias.csv."}`;
    $('evaluation-analysis').textContent = `${result.lecturas.length}/5 posturas completadas (${result.analisis.validas ?? result.lecturas.filter(l => l.primera).length} con lectura). Mayor similitud media: ${result.personas.filter(p => Number.isFinite(p.media)).sort((a,b) => b.media-a.media)[0]?.nombre || 'Sin lectura'}.`;
    $('evaluation-ranking').replaceChildren();
    if (isSFace) {
      const refs = document.createElement('p');
      refs.textContent = 'Referencias por identidad: ' + result.personas.map(p => `${p.nombre}: ${p.referencias}`).join(', ');
      $('evaluation-ranking').append(refs);
      for (const lectura of result.lecturas) {
        const line = document.createElement('p');
        line.textContent = `${lectura.postura}: ${lectura.primera || 'Sin lectura'} / segunda: ${lectura.segunda || '-'} / separacion: ${typeof lectura.separacion === 'number' ? lectura.separacion.toFixed(4) : '-'}${lectura.error ? ' / ' + lectura.error : ''}`;
        $('evaluation-ranking').append(line);
      }
    }
    $('evaluation-rows').replaceChildren();
    for (const person of result.personas) {
      const row = document.createElement('tr');
      for (const value of [person.nombre, ...postureGuide.map(([pose]) => person.posturas[pose]), person.minima, person.media, person.maxima]) {
        const cell = document.createElement('td');
        cell.textContent = typeof value === 'number' ? value.toFixed(isSFace ? 4 : 2) : value == null ? 'Sin lectura' : value;
        row.append(cell);
      }
      $('evaluation-rows').append(row);
    }
  }
}
document.querySelector(".mode-card").after($("hardware-controls"));
$("hardware-controls").hidden = false;

async function refreshPorts() {
  try {
    const data = await request("/api/puertos"), previous = $("arduino-port").value;
    $("arduino-port").replaceChildren(new Option("Automático · identificar Arduino", ""));
    for (const port of data.puertos) $("arduino-port").add(new Option(`${port.puerto} · ${port.descripcion}`, port.puerto));
    if (data.puertos.some(p => p.puerto === previous)) $("arduino-port").value = previous;
    if (!data.puertos.length) toast("No se detectan puertos. Comprueba la conexión USB de Arduino y actualiza la lista.");
  } catch (error) { toast(error.message); }
}
function modeDisplay() {
  const physical = $("system-mode").value === "fisico";
  $("port-controls").hidden = !physical;
  document.querySelector('[data-option="emociones"]').disabled=false;
  document.querySelector(".mode-card strong").textContent = physical ? "Cabeza física" : "Modo demostración";
  document.querySelector(".mode-card div>span").textContent = physical ? "Usa la comunicación existente con Arduino" : "Sin órdenes a Arduino";
  document.querySelector(".control-card>p").textContent = physical ? "Revisa el puerto y confirma el arranque antes de conectar." : "Sin órdenes a Arduino";
  document.querySelector(".demo-tag").textContent = physical ? "Cabeza física" : "Modo demo";
}
function renderGuiaGestos(gestos) {
  const lista=$('gesture-guide');lista.replaceChildren();
  for (const gesto of gestos) {
    const item=document.createElement('div');item.className='gesture-item';
    item.dataset.gesto=gesto.gesto;
    const img=document.createElement('img');img.src=`/imagenes/${gesto.imagen}.png`;
    img.alt='';img.width=512;img.height=512;img.className='gesture-image';
    const nombre=document.createElement('strong');nombre.textContent=gesto.nombre;
    const instruccion=document.createElement('span');instruccion.textContent=gesto.instruccion;
    const movimiento=document.createElement('p');movimiento.className='gesture-movement';
    movimiento.textContent=`Robot: ${gesto.movimiento}`;
    item.append(img,nombre,instruccion,movimiento);
    lista.append(item);
  }
}
async function cargarGuiaGestos() {
  $('gesture-guide').textContent='Cargando movimientos...';
  try { const datos=await request('/api/guia-gestos');renderGuiaGestos(datos.gestos); }
  catch (error) { $('gesture-guide').textContent='No se pudo cargar la guía. Vuelve a abrirla.';toast(error.message); }
}
function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 6500);
}
function clearImage() {
  $("registration-image").hidden = true;
  renderPosture(state?.recorrido || {}, false);
  $("camera-image").hidden = true; $("camera-empty").hidden = false; $("live-label").hidden = true;
  $("registration-image").removeAttribute("src"); $("camera-image").removeAttribute("src");
  $("registration-wait").hidden = false;
  if (imageUrl) URL.revokeObjectURL(imageUrl); imageUrl = null; lastFrameAt = 0;
}
function loginScreen() {
  authenticated = false; csrf = ""; state = null; offline = false; clearImage();
  $("app").hidden = true; $("login").hidden = false;
  for (const id of ["registration-verification", "register-dialog", "guide-dialog", "delete-dialog", "physical-dialog"]) if ($(id).open) $(id).close();
  verificacionVisible = null;
  pendingPhysical = null;
  deleteName = null;
}
async function request(path, data) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 8000);
  let response;
  try {
    response = await fetch(path, {credentials:"same-origin", cache:"no-store", signal:controller.signal,
      ...(data !== undefined ? {method:"POST", headers:{"Content-Type":"application/json", "X-CSRF-Token":csrf}, body:JSON.stringify(data)} : {})});
  } catch (_) { throw new Error("No hay conexión con CLAUD-IA. Comprueba que el computador y esta red sigan disponibles."); }
  finally { clearTimeout(timer); }
  const body = await response.json();
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/entrar") loginScreen();
    throw new Error(body.error || "No se pudo completar la acción.");
  }
  return body;
}
async function action(path, data = {}) {
  if (actionBusy) return false;
  actionBusy = true;
  try { await request(path, data); await sync(); return true; }
  catch (error) { toast(error.message); return false; }
  finally { actionBusy = false; }
}
function view(name) {
  for (const n of ["laboratorio", "personas"]) $("view-"+n).hidden = name !== n;
  document.querySelectorAll("[data-view]").forEach(button => button.classList.toggle("selected", button.dataset.view === name));
  $("page-label").textContent = name === "personas" ? "Personas" : "Panel de control";
  if (name === "personas") loadPeople();
}
function similitudReconocimiento(s, nombre) {
  if (!nombre || nombre === "Desconocido" || s.motor_activo !== "sface") return "";
  return Number.isFinite(s.similitud_coseno) ? `Similitud ${(s.similitud_coseno * 100).toFixed(1).replace('.', ',')} %` : "";
}
function nombreCapturaValido() {
  const n = $("new-name").value.normalize('NFKC').trim();
  return n.length > 0 && n.length <= 60 && !/[\u0000-\u001f\u007f-\u009f]/.test(n);
}
function actualizarBotonCaptura() {
  $("begin-register").disabled = !nombreCapturaValido() || Boolean(state?.actividad) || actionBusy;
}

function render(s) {
  state = s;
  const busy = s.fase === "iniciando" || s.fase === "activo" || s.fase === "deteniendo";
  const registering = s.actividad === "registro";
  if (busy) $("system-mode").value = s.modo;
  $("system-mode").disabled = busy;
  $("arduino-port").disabled = busy;
  $("refresh-ports").disabled = busy;
  modeDisplay();
  const arduinoLabels = {demo:"Sin órdenes a Arduino", conectando:"Arduino: abriendo conexión…", conectado:`Arduino: puerto ${s.puerto || ""} abierto. Sin confirmación de movimientos.`, desconectado:"Arduino: sin conexión activa", error:"Arduino: error de comunicación. Revisa cable, puerto y otras aplicaciones."};
  $("arduino-state").textContent = arduinoLabels[s.arduino] || "Arduino: sin conexión activa";
  if ($("delete-dialog").open) {
    $("confirm-delete").disabled = busy || actionBusy;
    $("delete-status").textContent = busy ? "Primero detén el sistema o el registro desde el laboratorio y espera a que se cierre. Después podrás confirmar." : "El sistema está detenido. En el próximo inicio, el reconocedor se entrenará sin esta persona.";
  }
  $("start").disabled = busy; $("stop").disabled = !busy || s.fase === "deteniendo";
  $("start").textContent = s.fase === "iniciando" ? "Preparando sistema…" : "▶ Iniciar sistema";
  $("stop").textContent = s.fase === "deteniendo" ? "Cerrando cámara…" : "■ Detener sistema";
  $("camera-index").disabled = busy;
  $("recognition-engine").disabled = busy || Boolean(s.actividad) || actionBusy;
  $("recognition-engine").value = s.motor_seleccionado || "sface";
  const engineNames = {lbph:"LBPH", sface:"SFace"};
  const activeEngine = offline ? "sin confirmar" : engineNames[s.motor_activo] || "ninguno";
  $("recognition-engine-status").textContent = `Motor activo: ${activeEngine} | seleccionado: ${engineNames[s.motor_seleccionado] || "LBPH"}`;
  $("recognition-engine-help").hidden = true;
  const labels = {detenido:"Sistema detenido", iniciando:"Iniciando sistema…", activo:registering ? "Registro en curso" : "Sistema activo", deteniendo:"Deteniendo y cerrando cámara…", error:"Actividad no disponible"};
  $("system-status").textContent = offline ? "Conexión interrumpida" : labels[s.fase] || "En espera";
  $("system-dot").classList.toggle("green", s.fase === "activo" && !offline);
  document.querySelector(".checkmark").hidden = s.fase !== "activo" || offline;
  const placeholder = document.querySelector("#camera-empty h2");
  placeholder.textContent = offline ? "Conexión interrumpida" : s.fase === "iniciando" ? "Iniciando cámara…" : s.fase === "deteniendo" ? "Deteniendo cámara…" : s.camara ? "Esperando imagen de la cámara" : "Cámara detenida";
  document.querySelector("#camera-empty p").textContent = !offline && !busy ? "Pulsa Iniciar sistema para ver la cámara" : "";
  $("camera-tag").textContent = s.camara ? "Conectada" : s.fase === "iniciando" ? "Preparando…" : "Desconectada";
  $("camera-tag").classList.toggle("active", s.camara);
  $("camera-caption").textContent = s.camara ? "● Cámara conectada al computador de CLAUD-IA" : "● Cámara en espera";
  const reading = !offline && s.fase === "activo" && s.camara && s.rostros > 0 && s.opciones.reconocimiento;
  const name = reading ? s.nombre : null;
  const similarity = similitudReconocimiento(s, name);
  $("result-name").textContent = s.opciones.reconocimiento ? name || "Sin lectura" : "Desactivado";
  $("result-similarity").textContent = similarity;
  $("result-similarity").hidden = !similarity;
  $("camera-recognition").textContent = name ? `${name}${similarity ? " \u2014 " + similarity : ""}` : "";
  $("camera-recognition").hidden = !name;
  $("person-note").textContent = name === "Desconocido" ? "Sin coincidencia aceptada" : s.rostros > 1 ? "Una persona a la vez" : s.estado_reconocimiento === "error" ? "Reconocimiento no disponible" : !name ? "Esperando un rostro" : "";
  $("person-note").hidden = Boolean(name && name !== "Desconocido");
  const fisico=$("system-mode").value==='fisico';
  const emocionVigente=!offline && s.fase==='activo' && s.camara && s.rostros===1 && s.emociones_estado==='listo';
  $("result-expression").textContent = s.opciones.emociones ? emocionVigente && s.expresion || "Sin lectura" : "Desactivado";
  $("expression-note").textContent = !s.opciones.emociones ? '' : s.detalle_emociones || (s.emociones_estado === "cargando" ? "Preparando el análisis…" : s.emociones_estado === "no disponible" ? "Análisis no disponible por ahora" : "Una estimación visual");
  $("result-gesture").textContent = s.opciones.gestos ? s.gesto || "Sin gesto" : "Desactivado";
  document.querySelectorAll("[data-option]").forEach(input => {
    input.checked = s.opciones[input.dataset.option]; input.disabled = registering || s.fase === "deteniendo";
  });
  actualizarBotonCaptura();
  $("test-registration").disabled = busy;
  const recorrido = s.recorrido || {};
  renderPosture(recorrido, registering && s.fase === "activo" && !offline);
  $("registration-intro").hidden = registering;
  $("registration-preparation").hidden = true;
  $("check-registration").hidden = true;
  $("test-registration").hidden = true;
  const alta = s.resultado_registro;
  $("registration-summary").textContent = "";
  if (alta?.estado === 'guardado' && alta.galeria !== registroActualizado) {
    registroActualizado = alta.galeria;
    if (authenticated) loadPeople();
  }
  $("registration-confirmation").textContent = "";
  $("registration-confirmation").hidden = true;
  $("retry-registration").hidden = !registering || !recorrido.pausado;
  $("retry-registration").disabled = s.fase === "deteniendo";
  $("registration-active").hidden = !registering;
  $("register-dialog").classList.toggle("is-running", registering);
  document.body.classList.toggle('capture-open', registering);
  $("register-form").hidden = registering;
  if (registering) {
    $("registration-name").textContent = s.registro_nombre;
    $("registration-count").textContent = `Paso ${recorrido.paso || 1} de 5`;
    $("registration-progress").max = s.total; $("registration-progress").value = s.progreso;
    $("registration-instruction").textContent = s.instruccion || "Espera a que la cámara esté lista.";
    $("cancel-register").disabled = s.fase === "deteniendo";
  }
  renderEvaluation(s, registering);
  renderVerificacionRegistro(s);
  if (s.aviso_id && s.aviso_id !== lastNotice) {
    lastNotice = s.aviso_id;
    if (!s.alta_sface || !(s.recorrido?.completo || s.resultado_registro)) toast(s.aviso);
    if (s.progreso === s.total && !registering) loadPeople();
  }
  if (!s.camara) clearImage();
}
async function sync() {
  if (!authenticated) return;
  const data = await request("/api/estado"); csrf = data.csrf; offline = false; render(data.estado);
  if (Date.now()-peopleTimer > 5000) { peopleTimer = Date.now(); await loadPeople(); }
}
async function loadPeople() {
  if (!authenticated) return;
  try {
    const {personas} = await request("/api/personas");
    $("people-count").textContent = personas.length;
    const list = $("people-list"); list.replaceChildren();
    if (!personas.length) { const empty = document.createElement("p"); empty.className = "empty-people"; empty.textContent = "Aquí aparecerán las personas que registres."; list.append(empty); }
    for (const person of personas) {
      const row = document.createElement("article"); row.className = "person-row";
      const avatar = document.createElement("span"); avatar.className = "person-avatar"; avatar.textContent = person.nombre.charAt(0).toLocaleUpperCase();
      const text = document.createElement("div"), name = document.createElement("div"), photos = document.createElement("p"), status = document.createElement("span");
      name.className = "person-name"; name.textContent = person.nombre; photos.textContent = `${person.fotos} fotos de registro`; text.append(name, photos);
      status.className = "person-status"; status.textContent = "Registrada";
      const remove = document.createElement("button"); remove.className = "delete-person";
      remove.textContent = "Eliminar"; remove.setAttribute("aria-label", `Eliminar registro de ${person.nombre}`);
      remove.addEventListener("click", () => {
        deleteName = person.nombre; $("delete-name").textContent = person.nombre;
        $("delete-dialog").showModal(); if (state) render(state); $("cancel-delete").focus();
      });
      row.append(avatar, text, status, remove); list.append(row);
    }
  } catch (error) { toast(error.message); }
}
async function poll() {
  if (authenticated) {
    try { await sync(); }
    catch (error) { if (!offline) toast(error.message); offline = true; $("system-dot").classList.remove("green"); $("camera-tag").classList.remove("active"); $("camera-tag").textContent = "Sin conexión"; document.querySelector(".checkmark").hidden = true; document.querySelector("#camera-empty h2").textContent = "Conexión interrumpida"; document.querySelector("#camera-empty p").textContent = ""; $("system-status").textContent = "Sin conexión con CLAUD-IA"; clearImage(); }
  }
  setTimeout(poll, authenticated && state?.actividad === "registro" && !offline ? 200 : 650);
}
async function frames() {
  if (authenticated && state?.camara && !offline) {
    try {
      const response = await fetch("/api/camara.jpg", {cache:"no-store", credentials:"same-origin", signal:AbortSignal.timeout(5000)});
      if (response.status === 401) loginScreen();
      else if (response.status === 200) {
        const blob = await response.blob();
        if (authenticated && state?.camara) {
          const old = imageUrl; imageUrl = URL.createObjectURL(blob);
          $("camera-image").src = imageUrl; $("registration-image").src = imageUrl;
          $("registration-image").hidden = false;
          $("camera-image").hidden = false; $("camera-empty").hidden = true; $("live-label").hidden = false; $("registration-wait").hidden = true;
          if (old) URL.revokeObjectURL(old); lastFrameAt = Date.now();
        }
      } else if (Date.now()-lastFrameAt > 4000) clearImage();
    } catch (_) { if (Date.now()-lastFrameAt > 4000) clearImage(); }
  }
  setTimeout(frames, document.hidden ? 1000 : 120);
}
$("login-form").addEventListener("submit", async event => {
  event.preventDefault(); $("login-button").disabled = true; $("login-error").textContent = "";
  try {
    const data = await request("/api/entrar", {clave:$("password").value});
    csrf = data.csrf; $("password").value = ""; authenticated = true;
    $("login").hidden = true; $("app").hidden = false; await sync();
  } catch (error) { $("login-error").textContent = error.message; }
  finally { $("login-button").disabled = false; }
});
$("logout").addEventListener("click", async () => { try { await request("/api/salir", {}); loginScreen(); } catch (error) { toast(error.message); } });
document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => view(button.dataset.view)));
$("system-mode").addEventListener("change", () => { modeDisplay(); if ($("system-mode").value === "fisico") refreshPorts(); });
$("refresh-ports").addEventListener("click", refreshPorts);
$("start").addEventListener("click", () => {
  if ($("system-mode").value === "demo") return action("/api/iniciar", {modo:"demo", camara:Number($("camera-index").value)});
  pendingPhysical = {modo:"fisico", camara:Number($("camera-index").value), puerto:$("arduino-port").value || null, confirmar_fisico:true};
  $("physical-port-summary").textContent = `Puerto: ${pendingPhysical.puerto || "identificación automática"}. Se ejecutará el arranque físico existente.`;
  $("physical-dialog").showModal(); $("cancel-physical").focus();
});
for (const id of ["close-physical", "cancel-physical"]) $(id).addEventListener("click", () => { pendingPhysical = null; $("physical-dialog").close(); });
$("physical-dialog").addEventListener("cancel", () => { pendingPhysical = null; });
$("confirm-physical").addEventListener("click", async () => {
  if (!pendingPhysical || actionBusy) return;
  $("confirm-physical").disabled = true;
  try { if (await action("/api/iniciar", pendingPhysical)) { pendingPhysical = null; $("physical-dialog").close(); } }
  finally { $("confirm-physical").disabled = false; }
});
$("recognition-engine").addEventListener("change", () => action("/api/motor", {motor:$("recognition-engine").value}));
$("stop").addEventListener("click", () => action("/api/detener"));
document.querySelectorAll("[data-option]").forEach(input => input.addEventListener("change", () => action("/api/opciones", {opciones:{[input.dataset.option]:input.checked}})));
async function openRegister() {
  evaluationSetup = false;
  evaluationView = Boolean(state?.evaluacion_registro);
  if (state?.actividad && state.actividad !== "registro") { toast("Deten el sistema antes de comenzar un registro."); return; }
  $("register-dialog").showModal();
  actualizarBotonCaptura();
  if (state?.actividad !== "registro") $("new-name").focus();
}
$("new-name").addEventListener("input", actualizarBotonCaptura);
for (const id of ["quick-register", "new-person"]) $(id).addEventListener("click", openRegister);
$("register-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!nombreCapturaValido()) return;
  const ok = await action("/api/iniciar-recorrido", {nombre:$("new-name").value.normalize('NFKC').trim(), camara:Number($("camera-index").value)});
  if (ok) {
    $("registration-active").hidden = false;
    // El modal permanece fijo; nunca desplazarlo al comenzar o cambiar de paso.
  }
});
$("test-registration").addEventListener("click", () => action("/api/probar-registro", {camara:Number($("camera-index").value)}));
$('evaluation-type').addEventListener('change', evaluationFields);
$('evaluation-engine').addEventListener('change', evaluationFields);
$('open-evaluation').addEventListener('click', async () => {
  try {
    const {personas} = await request('/api/personas');
    $('evaluation-known').replaceChildren(new Option('Selecciona el nombre exacto', ''));
    for (const person of personas) $('evaluation-known').add(new Option(person.nombre, person.nombre));
    $('evaluation-consent').checked = false;
    evaluationSetup = true;
    evaluationView = true;
    evaluationFields();
    if (state) render(state);
    $('evaluation-type').focus();
  } catch (error) { toast(error.message); }
});
$('back-evaluation').addEventListener('click', () => { evaluationSetup = false; evaluationView = false; if (state) render(state); });
$('exit-evaluation').addEventListener('click', () => { evaluationView = false; if (state) render(state); });
$('evaluation-form').addEventListener('submit', async event => {
  event.preventDefault();
  const tipo = $('evaluation-type').value;
  const datos = {tipo, identificador: tipo === 'registrada' ? $('evaluation-known').value : $('evaluation-id').value.trim(), acepta_conservar: $('evaluation-consent').checked};
  if ($('evaluation-engine').value === 'sface') {
    datos.motor = 'sface';
    datos.nombre_propuesto = tipo === 'registrada' ? $('evaluation-alias').value.trim() : '';
  }
  $('begin-evaluation').disabled = true;
  const ok = await action('/api/evaluar-duplicados', {camara:Number($('camera-index').value), evaluacion:datos});
  if (ok) evaluationSetup = false;
  if (state) render(state);
});
$("retry-registration").addEventListener("click", () => action("/api/reintentar-registro"));
$("cancel-register").addEventListener("click", () => action("/api/detener"));
$("close-register").addEventListener("click", () => { $("register-dialog").close(); if (state?.actividad === "registro") toast("El registro sigue abierto. Puedes retomarlo desde Registrar persona o detenerlo desde el laboratorio."); });
$("guide-button").addEventListener("click", () => { $("guide-dialog").showModal(); cargarGuiaGestos(); });
$("close-guide").addEventListener("click", () => $("guide-dialog").close());
$("refresh-people").addEventListener("click", loadPeople);
for (const id of ["close-delete", "cancel-delete"]) $(id).addEventListener("click", () => { $("delete-dialog").close(); deleteName = null; });
$("confirm-delete").addEventListener("click", async () => {
  if (!deleteName || actionBusy) return;
  $("confirm-delete").disabled = true;
  if (await action("/api/eliminar", {nombre:deleteName, confirmacion:deleteName})) {
    $("delete-dialog").close(); deleteName = null; await loadPeople();
  } else if (state) render(state);
});
async function restore() {
  try {
    const data = await request("/api/estado"); csrf = data.csrf; authenticated = true;
    $("login").hidden = true; $("app").hidden = false; render(data.estado); await loadPeople();
  } catch (_) { loginScreen(); }
  poll(); frames();
}
restore();
