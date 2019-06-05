import json
import numpy as np
import os
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
        self.train_input = None
        self.train_exp_output = None

        self.v_E = 0.0
        self.v_ave = -67.0

        self.tau_rise = 0.5
        self.tau_fall = 5.0
        self.syn_kernel_len = 50.0

        self._ETA = None

    def output(self, i_inj, dt, t_stop, int_noise_regen=True):
        tt = np.arange(0.0, t_stop, dt)
        V = np.zeros((tt.shape[0], self.NUM_NEURONS)) # Membrane potential per neuron

        # Additive noise to individual neurons
        if self._ETA is None \
                or int_noise_regen is True\
                or self._ETA.shape != V.shape:
            self._ETA, _ = ouprocess_gaussian(5.0, dt, t_stop, self.NUM_NEURONS)

        F_binary = np.zeros((tt.shape[0], self.NUM_NEURONS))
        # avg_firing_rate = np.zeros(self.NUM_NEURONS)

        I_total = self.std_noise*self._ETA + i_inj

        V = lif_compute(I_total, self.R, self.tau_V, self.V_th, dt)
        F_binary = spike_binary(V)

        syn_waveform = id_synaptic_waveform(
            dt,
            self.syn_kernel_len,
            self.tau_rise,
            self.tau_fall)
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
        self.train_input = i_inj
        self.train_exp_output = exp_output
    
    def as_dict(self):
        props_dict = {}
        props_dict['NUM_NEURONS'] = self.NUM_NEURONS
        props_dict['tau_V'] = self.tau_V
        props_dict['R'] = self.R
        props_dict['EL'] = self.EL
        props_dict['V_th'] = self.V_th
        props_dict['std_noise'] = self.std_noise

        props_dict['v_E'] = self.v_E
        props_dict['v_ave'] = self.v_ave

        props_dict['tau_rise'] = self.tau_rise
        props_dict['tau_fall'] = self.tau_fall
        props_dict['syn_kernel_len'] = self.syn_kernel_len

        return props_dict

    @classmethod
    def from_dict(cls, in_dict: dict) -> 'Layer':
        NUM_NEURONS = in_dict['NUM_NEURONS']
        tau_V = in_dict['tau_V']
        R = in_dict['R']
        EL = in_dict['EL']
        V_th = in_dict['V_th']
        std_noise = in_dict['std_noise']

        v_E = in_dict['v_E']
        v_ave = in_dict['v_ave']

        tau_rise = in_dict['tau_rise']
        tau_fall = in_dict['tau_fall']
        syn_kernel_len = in_dict['syn_kernel_len']

        layer = cls(NUM_NEURONS)
        layer.tau_V = tau_V
        layer.R = R
        layer.EL = EL
        layer.V_th = V_th
        layer.std_noise = std_noise

        layer.v_E = v_E
        layer.v_ave = v_ave

        layer.tau_rise = tau_rise
        layer.tau_fall = tau_fall
        layer.syn_kernel_len = syn_kernel_len

        return layer

    def save(self, path, layer_name):
        LAYER_ATTRS_JSON = layer_name + "_attrs.json"
        LAYER_WEIGHTS_NPZ = layer_name + "_weights.npz"

        with open(os.path.join(path, LAYER_ATTRS_JSON), 'w') as outfile:
            json.dump(self.as_dict(), outfile)

        np.savez(open(os.path.join(path, LAYER_WEIGHTS_NPZ), 'wb'),
            W=self.W)

    @classmethod
    def load(cls, path: str, layer_name: str) -> 'Layer':
        in_dict = {}
        LAYER_ATTRS_JSON = layer_name + "_attrs.json"
        LAYER_WEIGHTS_NPZ = layer_name + "_weights.npz"

        with open(os.path.join(path, LAYER_ATTRS_JSON), 'r') as infile:
            in_dict = json.load(infile)
        layer = cls.from_dict(in_dict)

        data = np.load(open(os.path.join(path, LAYER_WEIGHTS_NPZ), 'rb'))
        layer.W = data['W']

        return layer
