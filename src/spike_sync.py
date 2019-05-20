import numpy as np

def find_synced_spikes(psth_learning, percentage_sync, window_sync, TW, trial_num, dt, t_stop):
    min_num_spikes = percentage_sync*trial_num;
    kernel_rectangle = np.concat((np.zeros((int(1.0/dt), 1.0)), np.ones(T_sync/dt, 1.0) np.zeros(1.0/dt, 1.0)))
    L_k = len(kernel_rectangle)
    