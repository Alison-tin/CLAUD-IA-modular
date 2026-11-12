import time
import threading

from configuracion.config import (
    PCA,
    SERVO,
    VEL_EXP,
    PARPADOS_ABIERTOS,
    PARPADOS_CERRADOS,
    MAXILAR_PCA,
    MAXILAR_CANAL,
    MAXILAR_ABRE,
    MAXILAR_CIERRA
)


# ======================================================
# MOVIMIENTO DE BOCA
# ======================================================

def mover_boca(arduino, ciclos=4, vel=8, pausa=0.13):
    """
    Anima el maxilar inferior simulando habla.
    """

    for _ in range(ciclos):

        arduino.enviar(
            MAXILAR_PCA,
            MAXILAR_CANAL,
            MAXILAR_ABRE,
            vel
        )

        time.sleep(pausa)

        arduino.enviar(
            MAXILAR_PCA,
            MAXILAR_CANAL,
            MAXILAR_CIERRA,
            vel
        )

        time.sleep(pausa)

    # Asegurar boca cerrada
    arduino.enviar(
        MAXILAR_PCA,
        MAXILAR_CANAL,
        MAXILAR_CIERRA,
        vel
    )


def hablar(arduino, ciclos=4, vel=8, pausa=0.13):
    """
    Ejecuta el movimiento de boca en un hilo
    para no bloquear el programa principal.
    """

    threading.Thread(
        target=mover_boca,
        args=(arduino, ciclos, vel, pausa),
        daemon=True
    ).start()


# ======================================================
# EXPRESIÓN DE SALUDO
# ======================================================

def expresion_saludo(arduino):

    for canal, posicion in [
        (3, 450),
        (4, 400),
        (5, 660),
        (8, 580),
        (9, 480),
        (10, 570),
        (11, 570)
    ]:

        arduino.enviar(
            PCA,
            canal,
            posicion,
            VEL_EXP
        )

    mover_boca(
        arduino,
        ciclos=3
    )


# ======================================================
# EXPRESIÓN NEUTRA
# ======================================================

def expresion_neutro(arduino):

    for canal in [3, 4, 5, 8, 9, 10, 11]:

        arduino.enviar(
            PCA,
            canal,
            SERVO[canal][2],
            VEL_EXP
        )

    # Boca cerrada en reposo
    arduino.enviar(
        MAXILAR_PCA,
        MAXILAR_CANAL,
        MAXILAR_CIERRA,
        VEL_EXP
    )


# ======================================================
# EXPRESIÓN AL RECONOCER UNA PERSONA
# ======================================================

def expresion_reconocido(arduino):

    expresion_saludo(
        arduino
    )

    time.sleep(0.3)

    arduino.enviar(
        PCA,
        12,
        PARPADOS_CERRADOS[12],
        3
    )

    arduino.enviar(
        PCA,
        13,
        PARPADOS_CERRADOS[13],
        3
    )

    time.sleep(0.25)

    arduino.enviar(
        PCA,
        12,
        PARPADOS_ABIERTOS[12],
        3
    )

    arduino.enviar(
        PCA,
        13,
        PARPADOS_ABIERTOS[13],
        3
    )

    mover_boca(
        arduino,
        ciclos=4
    )