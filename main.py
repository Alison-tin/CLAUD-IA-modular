"""Punto de entrada modular de CLAUD-IA."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
import argparse
from datos.personas import cargar_db, guardar_db, listar_personas, registrar_persona
from aplicacion.sistema import ejecutar

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="InMoov Vision v2")
    ap.add_argument("--registrar",type=str,default=None,metavar="NOMBRE")
    ap.add_argument("--listar",   action="store_true")
    ap.add_argument("--borrar",   type=str,default=None,metavar="NOMBRE")
    ap.add_argument("--cam",      type=int,default=0)
    ap.add_argument("--port",     type=str,default=None)
    ap.add_argument("--demo",     action="store_true")
    ap.add_argument("--comparar-rostro", action="store_true",
                    help="Comparacion temporal: muestra seleccionada y entrada en vivo a LBPH")
    ap.add_argument("--persona-comparacion", choices=("Alexa", "daniela"), default="Alexa",
                    help="Persona de referencia para --comparar-rostro (por defecto: Alexa)")
    ap.add_argument("--motor", choices=("sface",), default="sface",
                    help="Motor facial SFace")
    args=ap.parse_args()

    if args.listar:
        listar_personas(cargar_db())
    elif args.borrar:
        db=cargar_db()
        idx=[i for i,n in enumerate(db["nombres"]) if n==args.borrar]
        if not idx: print(f"  '{args.borrar}' no encontrado.")
        else:
            keep=set(range(len(db["nombres"])))-set(idx)
            db["nombres"]  =[db["nombres"][i]   for i in sorted(keep)]
            db["imagenes"] =[db["imagenes"][i] for i in sorted(keep)]
            guardar_db(db); print(f"  '{args.borrar}' eliminado ({len(idx)} muestras; las fotos permanecen en disco).")
        listar_personas(db)
    elif args.registrar:
        print('El alta requiere una aprobacion todavia no definida. Inicia el recorrido sin guardar desde la web.')
    else:
        try:
            ejecutar(cam_idx=args.cam,port=args.port,demo=args.demo,
                     comparar_rostro=args.comparar_rostro, persona_comparacion=args.persona_comparacion, motor=args.motor)
        except Exception as e:
            import traceback; traceback.print_exc()
            input("\n Error  ENTER para cerrar...")

