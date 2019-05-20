import numpy as np
import seaborn

from src.lif import lif_compute, spike_binary
from src.ou_process import ouprocess_gaussian

if __name__ == '__main__':
    TRIAL_NUM = 100
    tau_V = 10
    R = 1 # MOhm
    EL = -70.0
    V_th = -40.0
    dt = 0.1 # msec
    t_stop = 50.0e3
    tt = np.arange(0.0, t_stop, dt)
    # p.dt = dt; p.tStop = t_stop; ????????
    tw = 100.0

    V = np.zeros((tt.shape[0], TRIAL_NUM)) # Membrane potential per neuron

    # Additive noise to individual neurons
    ETA = np.zeros(V.shape)
    for i in range(0, TRIAL_NUM):
        ETA[:,i], _ = ouprocess_gaussian(5, dt, t_stop)

    # Slow Signal
    input_slow, _ = ouprocess_gaussian(5, dt, t_stop)
    i_inj = 16.0 + 6.0*input_slow

    F_binary = np.zeros((tt.shape[0], TRIAL_NUM))
    avg_firing_rate = np.zeros(TRIAL_NUM)
    a2 = 25.0 # pA; std of noise

    for k in range(0, TRIAL_NUM):
        I_total = i_inj + a2*ETA[:,k]
        V[:,k] = lif_compute(I_total, R, tau_V, V_th, dt)
        F_binary[:, k] = spike_binary(V[:, k])
        avg_firing_rate[k] = np.sum(F_binary[:,k])/(t_stop/1e3)

