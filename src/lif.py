import numpy as np


def lif_compute(I_Total, R, tau_V, Th, dt):
    EL = -70  # mV
    V_th = Th  # -50
    Iin = I_Total.copy()
    V = EL

    V_reset = -90.0

    k = 1
    total_time = len(Iin) - 1 / dt
    V_out = np.zeros(Iin.shape[-1])

    while k <= total_time:
        # LIF equation
        dq_dt = 1 / tau_V * (-1 * (V - EL) + R * Iin[k])
        V += dt

        # Spike
        if V >= V_th:
            V_out[k] = 50  # TODO: ???
            V = V_reset # TODO: What's the purpose of this
            V_out[k+1:int(k+1/dt)+1] = V_reset # TODO: ???
            k += int(1.0/dt) # TODO: ???
        # No spike
        else:
            V_out[k] = V
            k += 1
        V = V_out[k-1] # TODO: what's the purpose of this?

    return V_out
