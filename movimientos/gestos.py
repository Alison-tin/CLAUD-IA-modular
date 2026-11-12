import random
import time

from configuracion.config import (
    PCA,
    PCA0,
    SERVO,
    SERVO_CUELLO,
    VEL_EXP,
    MAXILAR_PCA,
    MAXILAR_CANAL,
    MAXILAR_ABRE,
    MAXILAR_CIERRA
)


# ======================================================
# MOVIMIENTOS DE OJOS
# ======================================================

def _ojos_h(arduino, pos1, vel=VEL_EXP):
    """Mueve los ojos horizontalmente."""

    pos1 = int(
        max(
            SERVO[1][0],
            min(SERVO[1][1], pos1)
        )
    )

    rh = (
        pos1 - SERVO[1][0]
    ) / (
        SERVO[1][1] - SERVO[1][0]
    )

    p6 = (
        SERVO[6][0]
        + rh * (SERVO[6][1] - SERVO[6][0])
    )

    arduino.enviar(PCA, 1, pos1, vel)
    arduino.enviar(PCA, 6, p6, vel)


def _ojos_v(arduino, pos0, vel=VEL_EXP):
    """Mueve los ojos verticalmente."""

    pos0 = int(
        max(
            SERVO[0][0],
            min(SERVO[0][1], pos0)
        )
    )

    rv = (
        pos0 - SERVO[0][0]
    ) / (
        SERVO[0][1] - SERVO[0][0]
    )

    p2 = (
        SERVO[2][1]
        - rv * (SERVO[2][1] - SERVO[2][0])
    )

    arduino.enviar(PCA, 0, pos0, vel)
    arduino.enviar(PCA, 2, p2, vel)


def _ojos_centro(arduino, vel=VEL_EXP):
    _ojos_h(
        arduino,
        SERVO[1][2],
        vel
    )

    _ojos_v(
        arduino,
        SERVO[0][2],
        vel
    )


def _cabeza_centro(arduino, vel=3):
    arduino.enviar(
        PCA0,
        15,
        SERVO_CUELLO[15][2],
        vel
    )

    arduino.enviar(
        PCA0,
        14,
        SERVO_CUELLO[14][2],
        vel
    )


# ======================================================
# GESTOS DE VIDA
# ======================================================

def gesto_bostezo(arduino):

    arduino.parpado(
        abrir=False,
        vel=2
    )

    arduino.enviar(
        PCA0,
        14,
        SERVO_CUELLO[14][2] - 50,
        2
    )

    arduino.enviar(
        MAXILAR_PCA,
        MAXILAR_CANAL,
        MAXILAR_ABRE,
        1
    )

    time.sleep(1.2)

    arduino.enviar(
        MAXILAR_PCA,
        MAXILAR_CANAL,
        MAXILAR_CIERRA,
        2
    )

    time.sleep(0.4)

    arduino.parpado(
        abrir=True,
        vel=2
    )

    _cabeza_centro(arduino)
    _ojos_centro(arduino)


def gesto_pensar(arduino):

    lado = random.choice([-1, 1])

    _ojos_v(
        arduino,
        SERVO[0][2]
        - int(
            (SERVO[0][2] - SERVO[0][0]) * 0.5
        )
    )

    _ojos_h(
        arduino,
        SERVO[1][2]
        + lado * int(
            (SERVO[1][1] - SERVO[1][0]) * 0.25
        )
    )

    arduino.enviar(
        PCA0,
        14,
        SERVO_CUELLO[14][2] + lado * 40,
        3
    )

    time.sleep(
        random.uniform(1.5, 2.5)
    )

    _cabeza_centro(arduino)
    _ojos_centro(arduino)


def gesto_mirar_alrededor(arduino):

    for _ in range(random.randint(3, 5)):

        _ojos_h(
            arduino,
            random.uniform(
                SERVO[1][0],
                SERVO[1][1]
            ),
            vel=6
        )

        _ojos_v(
            arduino,
            random.uniform(
                SERVO[0][0],
                SERVO[0][1]
            ),
            vel=6
        )

        time.sleep(
            random.uniform(0.4, 0.9)
        )

    _ojos_centro(arduino)


def gesto_curiosear(arduino):

    lado = random.choice([-1, 1])

    arduino.enviar(
        PCA0,
        14,
        SERVO_CUELLO[14][2] + lado * 55,
        3
    )

    arduino.enviar(
        PCA0,
        15,
        SERVO_CUELLO[15][2] + lado * 40,
        3
    )

    _ojos_h(
        arduino,
        SERVO[1][2]
        + lado * int(
            (SERVO[1][1] - SERVO[1][0]) * 0.2
        )
    )

    time.sleep(
        random.uniform(1.2, 2.0)
    )

    _cabeza_centro(arduino)
    _ojos_centro(arduino)


def gesto_estirar_cuello(arduino):

    arduino.enviar(
        PCA0,
        15,
        SERVO_CUELLO[15][0] + 60,
        2
    )

    time.sleep(1.4)

    arduino.enviar(
        PCA0,
        15,
        SERVO_CUELLO[15][1] - 60,
        2
    )

    time.sleep(1.4)

    arduino.enviar(
        PCA0,
        15,
        SERVO_CUELLO[15][2],
        3
    )

    time.sleep(0.5)


def gesto_doble_parpadeo(arduino):

    for _ in range(2):

        arduino.parpado(
            abrir=False,
            vel=3
        )

        time.sleep(0.12)

        arduino.parpado(
            abrir=True,
            vel=3
        )

        time.sleep(0.18)


def gesto_suspiro(arduino):

    _ojos_v(
        arduino,
        SERVO[0][2]
        + int(
            (SERVO[0][1] - SERVO[0][2]) * 0.4
        )
    )

    arduino.enviar(
        MAXILAR_PCA,
        MAXILAR_CANAL,
        int(
            (MAXILAR_ABRE + MAXILAR_CIERRA) / 2
        ),
        2
    )

    time.sleep(0.6)

    arduino.enviar(
        MAXILAR_PCA,
        MAXILAR_CANAL,
        MAXILAR_CIERRA,
        2
    )

    time.sleep(0.4)

    _ojos_centro(arduino)


# ======================================================
# LISTA DE GESTOS DE VIDA
# ======================================================

GESTOS_VIDA = [
    gesto_bostezo,
    gesto_pensar,
    gesto_mirar_alrededor,
    gesto_curiosear,
    gesto_estirar_cuello,
    gesto_doble_parpadeo,
    gesto_suspiro,
]