"""Observacion del puerto existente: no genera ni modifica ordenes."""


class SerialObservado:
    def __init__(self, serial, informar, detener):
        self.original = serial
        self.informar = informar
        self.detener = detener
        self.fallo = False

    def __getattr__(self, nombre):
        return getattr(self.original, nombre)

    def notificar(self, estado):
        if not self.fallo:
            self.fallo = True
            self.informar("arduino", {"estado": estado, "puerto": self.original.port})
            self.detener.set()

    @property
    def is_open(self):
        abierto = self.original.is_open
        if not abierto:
            self.notificar("desconectado")
        return abierto

    def write(self, datos):
        try:
            resultado = self.original.write(datos)
            if resultado != len(datos):
                self.notificar("error")
            return resultado
        except Exception:
            self.notificar("error")
            raise


def preparar_conexion(puente, arduino, demo):
    if demo:
        puente.emitir("arduino", {"estado": "demo", "puerto": None})
        return
    if arduino.demo or arduino.ser is None or not arduino.ser.is_open:
        puente.emitir("arduino", {"estado": "error", "puerto": None})
        raise RuntimeError("No se pudo abrir Arduino. Comprueba el cable, el puerto y que otra aplicacion no lo use.")
    arduino.ser = SerialObservado(arduino.ser, puente.emitir, puente.parada)
    puente.serial_observado = arduino.ser
    puente.emitir("arduino", {"estado": "conectado", "puerto": arduino.ser.port})
