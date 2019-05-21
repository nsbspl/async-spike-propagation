import numpy as np

def find_synced_spikes(psth_learning, percentage_sync, window_sync, TW, trial_num, dt, t_stop):
    min_num_spikes = percentage_sync*trial_num;
    kernel_rectangle = np.concat((np.zeros((int(1.0/dt), 1.0)), np.ones(T_sync/dt, 1.0) np.zeros(1.0/dt, 1.0)))
    L_k = len(kernel_rectangle)
    sig_PSTH = np.zeros((psth_learning.shape[0]))
    sigg_ = np.convolve(psth_learning, kernel_rectangle)
    sig_PSTH = sigg_(int(L_k/2):-1*int(L_k/2))
    
    # Gaussian kernel PSTH
    FG_K_PSTH = np.zeros((psth_learning.shape[0]))
    bin_width = TW
    res = dt
    t = np.arange(-5.0*bin_width, 5.0*bin_width, res)
    gauss_kernel = 1/sqrt(2.0*pi*(bin_width)^2)*exp(np.power(-t, 2)/(2.0*(bin_width)^2))
    