import numpy as np
from scipy.signal import find_peaks

from src.lif import lif_compute, spike_binary, id_synaptic_waveform
from src.ou_process import ouprocess_gaussian
from src.spike_sync import find_synced_spikes

class Layer:
    
    def __init__(self, num_neurons):
        self.NUM_NEURONS = num_neurons
        self.tau_V = 10
        self.R = 1 # MOhm
        self.EL = -70.0
        self.V_th = -40.0
        self.std_noise = 25.0
        self.W = np.zeros((self.NUM_NEURONS, 1))

        self.v_E = 0.0
        self.v_ave = -67.0

        self.tau_rise = 0.5
        self.tau_fall = 5.0
        self.syn_kernel_len = 50.0
    
    def output(self, i_inj, dt, t_stop):
        tt = np.arange(0.0, t_stop, dt)
        V = np.zeros((tt.shape[0], self.NUM_NEURONS)) # Membrane potential per neuron
        # Additive noise to individual neurons
        ETA, _ = ouprocess_gaussian(5.0, dt, t_stop, self.NUM_NEURONS)

        F_binary = np.zeros((tt.shape[0], self.NUM_NEURONS))
        # avg_firing_rate = np.zeros(self.NUM_NEURONS)

        I_total = self.std_noise*ETA + i_inj

        for k in range(0, self.NUM_NEURONS):
            V[:,k] = lif_compute(I_total[:, k], self.R, self.tau_V, self.V_th, dt)
        F_binary = spike_binary(V)

        syn_waveform = id_synaptic_waveform(dt, self.syn_kernel_len, self.tau_rise, self.tau_fall)
        syn_wave_len = syn_waveform.shape[0]
        t_steps = F_binary.shape[0]

        F_synaptic = np.zeros(F_binary.shape)
        for neuron in range(0, self.NUM_NEURONS):
            fr_fast = np.convolve(F_binary[:,neuron], syn_waveform)
            F_synaptic[:, neuron] = fr_fast[:-syn_wave_len+1]
        
        ind_neur = np.arange(0, self.NUM_NEURONS)
        Phi = F_synaptic[:t_steps, ind_neur]
        X2 = -1.0*self.v_ave*np.ones((t_steps,ind_neur.shape[0])) + self.v_E

        A = np.multiply(Phi, X2)
        out = np.dot(A, self.W)

        return out, V, F_binary, F_synaptic
    
    def train(self, i_inj, exp_output, dt, t_stop):
        _, _, _, F_synaptic = self.output(i_inj, dt, t_stop)

        t_steps = exp_output.shape[0]

        ind_neur = np.arange(0, self.NUM_NEURONS)
        Phi = F_synaptic[:t_steps, ind_neur]
        X2 = -1.0*self.v_ave*np.ones((t_steps,ind_neur.shape[0])) + self.v_E

        A = np.multiply(Phi, X2)
        self.W, residuals, rank, s = np.linalg.lstsq(A, exp_output)
