import numpy as np
from scipy.signal import find_peaks

def find_synced_spikes(psth_learning, percentage_sync, window_sync, TW, trial_num, dt, t_stop):
    min_num_spikes = percentage_sync*trial_num;
    rect_kernel = np.concatenate((
        np.zeros((int(1.0/dt), 1)),
        np.ones((int(window_sync/dt), 1)),
        np.zeros((int(1.0/dt), 1))
    )).flatten()
    L_k = len(rect_kernel)
    
    # Rectangular kernel PSTH: total # of spikes in rectangular window
    # rect_psth = np.zeros((psth_learning.shape[0]))
    rect_psth = np.convolve(psth_learning, rect_kernel)
    rect_psth = rect_psth[int(L_k/2):-1*int(L_k/2)]
    
    # Gaussian kernel PSTH: average # of spikes in Gaussian window
    #     - used because Gaussian kernel yields smoothed graph with 1 local maxima unlike rect kernel
    gaussian_psth = np.zeros((psth_learning.shape[0]))
    bin_width = TW
    res = dt
    t = np.arange(-5.0*bin_width, 5.0*bin_width, res)

    gauss_kernel = 1.0/(bin_width*np.sqrt(2.0*np.pi))*np.exp(-1.0*np.power(t, 2.0)/(2.0*np.power(bin_width, 2.0)))
    gauss_kernel /= np.linalg.norm(gauss_kernel, ord=1)*res # Want area under curve to be 1

    L_hkernel = gauss_kernel.shape[0]

    gauss_convolved = np.convolve(psth_learning, gauss_kernel)[int(L_hkernel/2):-int(L_hkernel/2)]
    gaussian_psth = gauss_convolved[:rect_psth.shape[0]]
    
    # Correcting shift between Gaussian PSTH and Rect PSTH
    count = int(200/dt)
    rect_psth_shifted = np.zeros(psth_learning.shape[0])
    xcorr = np.correlate(
        rect_psth / np.std(rect_psth),
        gaussian_psth / np.std(gaussian_psth),
        'same'
    )[int(count/2)-1:-int(count/2)] # ??????? NEED TO REVIEW
    # lag_for_max = np.argmax(xcorr)
    # L2 = rect_psth.shape[0] - np.abs(lag_for_max-count) + 1;
    # rect_psth_shifted[:L2] = rect_psth[int(np.abs(lag_for_max-count))-1:]
    rect_psth_shifted = rect_psth
    
    # Rescale Gaussian PSTH to represent total # of spikes
    gaussian_psth_rescaled = np.divide(gaussian_psth, np.max(gaussian_psth)) * max(rect_psth_shifted)
    
    # Find peaks
    pk_inds, _ = find_peaks(gaussian_psth_rescaled)
    sync_event = pk_inds[gaussian_psth_rescaled[pk_inds]>=min_num_spikes]

    # Get sync spike indices, async spike indices
    MM_S = np.zeros(rect_psth_shifted.shape[0])
    MM_S[sync_event] = 1.0
    pf = np.convolve(MM_S, rect_kernel)
    M_S = pf[int(L_k/2)-1:-int(L_k/2)+1]
    ind_sync = np.where(psth_learning*M_S == 1.0)
    ind_async = np.where((psth_learning - psth_learning*M_S) == 1.0)

    return ind_async, ind_sync, sync_event, M_S
