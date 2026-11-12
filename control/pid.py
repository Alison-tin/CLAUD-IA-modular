import time


# ======================================================
# CONTROL PID
# ======================================================

class PID:

    def __init__(self, kp, ki, kd, zm=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.zm = zm

        self.i = 0.0
        self.ep = 0.0
        self.tp = time.time()

    def update(self, e):

        if abs(e) < self.zm:
            self.i = 0
            return 0.0

        t = time.time()
        dt = max(
            t - self.tp,
            0.001
        )

        self.tp = t

        # Anti-windup
        self.i = max(
            -80,
            min(
                80,
                self.i + e * dt
            )
        )

        d = (
            e - self.ep
        ) / dt

        self.ep = e

        return (
            self.kp * e
            + self.ki * self.i
            + self.kd * d
        )

    def reset(self):
        self.i = 0.0
        self.ep = 0.0