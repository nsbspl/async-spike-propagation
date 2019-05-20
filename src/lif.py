import numpy as np

# VECTORIZE
def lif_compute(I_Total, R, tau_V, Th, dt):
    EL = -70.0  # mV
    V_th = Th  # -50
    V = EL

    V_reset = -90.0

    k = 0
    total_time = len(I_Total) - 1.0 / dt
    V_out = np.zeros(I_Total.shape[0])

    while k <= total_time:
        # LIF equation
        dq_dt = 1.0 / tau_V * (-1.0 * (V - EL) + R * I_Total[k])
        V += dt*dq_dt

        # Spike
        if V >= V_th:
            V_out[k] = 50.0  # TODO: ???
            V_out[k+1:int(k+1.0/dt)+1] = V_reset # TODO: ???
            k += int(1.0/dt) # TODO: ???
        # No spike
        else:
            V_out[k] = V
            k += 1
        V = V_out[k-1] # TODO: what's the purpose of this?

    return V_out


def spike_binary(V: np.array):
    thres = -20 # mv
    trial_num = V.shape[1]
    L = V.shape[0]
    F = np.zeros((V.shape[0]+1, V.shape[1]))
    TINY_VAL = -90.0 * np.ones((trial_num))

    # CONVERT SPIKE INTO DIRAC DELTA
    # Shift right a copy of V by 1; where copy intersects with V can be
    # used as a point representing the spike
    V_trial = np.concatenate((V, [TINY_VAL]))
    V_trial_shifted = np.concatenate(([TINY_VAL], V))

    spike_bool = np.logical_and(V_trial > thres, thres > V_trial_shifted)
    F = 1.0 * spike_bool
    
    return F[2:, :]


# TODO: VECTORIZE
def spike_binary_ghetto(V: np.array):
    thres = -20 # mv
    trial_num = V.shape[1]
    L = V.shape[0]
    F = np.zeros((V.shape[0]+1, V.shape[1]))
    TINY_VAL = -90.0

    for i in range(0, trial_num):
        # CONVERT SPIKE INTO DIRAC DELTA
        # Shift right a copy of V by 1; where copy intersects with V can be
        # used as a point representing the spike
        V_trial = np.concatenate((V[:,i], [TINY_VAL]))
        V_trial_shifted = np.concatenate(([TINY_VAL], V[:,]))

        spike_bool = V_trial > thres > V_trial_shifted
        F[:, i] = 1.0 * spike_bool

    return F[2:, :]
