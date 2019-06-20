import numpy as np

# VECTORIZE
def lif_compute(I_total, R, tau_V, Th, dt):
    EL = -70.0  # mV
    V_th = Th  # -50
    V = np.ones((I_total.shape[1]))*EL

    V_reset = -90.0

    k = 0
    total_time = len(I_total) - 1.0/dt
    V_out = np.zeros(I_total.shape)

    spike_reset_count = np.zeros((I_total.shape[1],1))

    while k <= total_time:
        # LIF equation
        dq_dt = 1.0 / tau_V * (-1.0 * (V - EL) + R * I_total[k])
        V += dt*dq_dt

        # Spike
        spike = 50.0*np.greater_equal(V, V_th) # Spike
        spike_reset_add = ((int(1.0/dt)+1)*np.greater_equal(V, V_th))
        spike_reset_count = np.add(spike_reset_count, spike_reset_add[:,None])

        # No spike
        subthresh = np.multiply(np.equal(spike_reset_count, 0.0),V[:,None])
        reset = np.multiply(np.greater(spike_reset_count, 0.0),V_reset)
        reset_or_subthres = reset + subthresh
        no_spike = np.multiply(reset_or_subthres, np.less(V, V_th)[:,None]).flatten()

        V_out[k] = spike + no_spike
        V = np.multiply(V_out[k,:], np.equal(spike_reset_count, 0.0).flatten()) + reset.flatten()

        spike_reset_count = spike_reset_count - 1.0*np.greater(spike_reset_count, 0.0)

        k += 1

        # # Spike
        # if np.greater_equal(V, V_th): # V >= V_th:
        #     V_out[k] = 50.0  # TODO: ???
        #     V_out[k+1:int(k+1.0/dt)+1] = V_reset # TODO: ???
        #     0:10+1 -> 11 steps

        #     k += int(1.0/dt) # TODO: ???
        # # No spike
        # else:
        #     V_out[k] = V
        #     k += 1
        # V = V_out[k-1]
    
    while k < V_out.shape[0]:
        V_out[k] = V
        k += 1
    
    return V_out


def spike_binary(V: np.array):
    thres = -20 # mv
    trial_num = V.shape[1]
    F = np.zeros((V.shape[0]+1, V.shape[1]))
    TINY_VAL = -90.0 * np.ones((trial_num))

    # CONVERT SPIKE INTO DIRAC DELTA
    # Shift right a copy of V by 1; where copy intersects with V can be
    # used as a point representing the spike
    V_trial = np.concatenate((V, [TINY_VAL]))
    V_trial_shifted = np.concatenate(([TINY_VAL], V))

    spike_bool = np.logical_and(V_trial > thres, thres > V_trial_shifted)
    F = 1.0 * spike_bool
    
    return F[1:, :]


# TODO: VECTORIZE
def spike_binary_ghetto(V: np.array):
    thres = -20 # mv
    trial_num = V.shape[1]
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

def id_synaptic_waveform(dt, t_end, tau_rise, tau_fall):
    t = np.arange(0.0, t_end, dt)
    
    tp = tau_rise*tau_fall / (tau_fall-tau_rise) * np.log(tau_fall/tau_rise)
    factor = -np.exp(-tp/tau_rise) + np.exp(-tp/tau_fall)
    
    return 1.0/factor * (np.exp(-t/tau_fall) - np.exp(-t/tau_rise))