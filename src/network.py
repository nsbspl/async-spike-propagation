import json
import numpy as np
import os
from scipy.signal import find_peaks


from src.lif import lif_compute, spike_binary, id_synaptic_waveform
from src.ou_process import ouprocess_gaussian
from src.spike_sync import find_synced_spikes


class Layer:
    
    def __init__(self, num_neurons, std_noise=25.0):
        self.NUM_NEURONS = num_neurons
        self.tau_V = 10
        self.R = 1 # MOhm
        self.EL = -70.0
        self.V_th = -40.0
        self.std_noise = std_noise
        
        self.W = np.zeros((self.NUM_NEURONS, 1))
        self.train_input = None
        self.train_exp_output = None

        self.v_E = 0.0
        self.v_ave = -67.0

        self.tau_rise = 0.5
        self.tau_fall = 5.0
        self.syn_kernel_len = 50.0

        self._ETA = None

    def spike(self, i_inj, dt, t_stop, int_noise_regen=True):
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

        F_synaptic = np.zeros(F_binary.shape)
        # TODO: VECTORIZE
        for neuron in range(0, self.NUM_NEURONS):
            fr_fast = np.convolve(F_binary[:,neuron], syn_waveform)
            F_synaptic[:, neuron] = fr_fast[:-syn_wave_len+1]
        
        return V, F_binary, F_synaptic

    def output(self, i_inj, dt, t_stop, int_noise_regen=True):
        V, F_binary, F_synaptic = self.spike(i_inj, dt, t_stop, int_noise_regen=True)
        t_steps = F_binary.shape[0]

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

        layer = cls(NUM_NEURONS, std_noise)
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
            W=self.W,
            train_input=self.train_input,
            train_exp_output=self.train_exp_output)

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
        layer.train_input = data['train_input']
        layer.train_exp_output = data['train_exp_output']

        return layer

class PropogationNetwork(Layer):

    def __init__(self, depth, num_neurons, std_noise=25.0):
        super().__init__(num_neurons, std_noise)

        self.depth = depth

    def output(self, i_inj, dt, t_stop, int_noise_regen=True):
        out = i_inj
        V = None
        F_binary = None
        F_synaptic = None

        for layer_num in range(self.depth):
            out, V, F_binary, F_synaptic =\
                super().output(out, dt, t_stop, int_noise_regen=True)
        
        return out, V, F_binary, F_synaptic
    
    def as_dict(self):
        props_dict = super().as_dict()
        props_dict['depth'] = self.depth

        return props_dict

    @classmethod
    def from_layer(cls, layer: Layer, depth: int) -> 'PropogationNetwork':
        prop_ntwrk = PropogationNetwork(depth, layer.NUM_NEURONS, layer.std_noise)

        prop_ntwrk.tau_V = layer.tau_V
        prop_ntwrk.R = layer.R
        prop_ntwrk.EL = layer.EL
        prop_ntwrk.V_th = layer.V_th
        
        prop_ntwrk.W = layer.W
        prop_ntwrk.train_input = layer.train_input
        prop_ntwrk.train_exp_output = layer.train_exp_output

        prop_ntwrk.v_E = layer.v_E
        prop_ntwrk.v_ave = layer.v_ave

        prop_ntwrk.tau_rise = layer.tau_rise
        prop_ntwrk.tau_fall = layer.tau_fall
        prop_ntwrk.syn_kernel_len = layer.syn_kernel_len
        
        prop_ntwrk._ETA = layer._ETA

        return prop_ntwrk

    @classmethod
    def from_dict(cls, in_dict: dict) -> 'PropogationNetwork':
        NUM_NEURONS = in_dict['NUM_NEURONS']
        depth = in_dict['depth']
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

        network = cls(depth, NUM_NEURONS, std_noise)
        network.tau_V = tau_V
        network.R = R
        network.EL = EL
        network.V_th = V_th
        network.std_noise = std_noise

        network.v_E = v_E
        network.v_ave = v_ave

        network.tau_rise = tau_rise
        network.tau_fall = tau_fall
        network.syn_kernel_len = syn_kernel_len

        return network

    @classmethod
    def load(cls, path: str, layer_name: str) -> 'PropgationNetwork':
        super().load(path, layer_name)

class _FullyConnectedLayer(Layer):

    def __init__(self, num_neurons, std_noise=25.0):
        super().__init__(num_neurons, std_noise=std_noise)
        self.W = np.zeros((self.NUM_NEURONS, self.NUM_NEURONS))

class FullyConnectedLayerApprox(_FullyConnectedLayer):

    @classmethod
    def from_layer(cls, layer: Layer) -> 'FullyConnectedLayerApprox':
        prop_ntwrk = FullyConnectedLayerApprox(layer.NUM_NEURONS, layer.std_noise)

        prop_ntwrk.tau_V = layer.tau_V
        prop_ntwrk.R = layer.R
        prop_ntwrk.EL = layer.EL
        prop_ntwrk.V_th = layer.V_th
        
        prop_ntwrk.W = np.random.normal(np.mean(layer.W), np.std(layer.W), prop_ntwrk.W.shape)
        prop_ntwrk.train_input = layer.train_input
        prop_ntwrk.train_exp_output = layer.train_exp_output

        prop_ntwrk.v_E = layer.v_E
        prop_ntwrk.v_ave = layer.v_ave

        prop_ntwrk.tau_rise = layer.tau_rise
        prop_ntwrk.tau_fall = layer.tau_fall
        prop_ntwrk.syn_kernel_len = layer.syn_kernel_len
        
        prop_ntwrk._ETA = layer._ETA

        return prop_ntwrk

class FullyConnectedLayer(_FullyConnectedLayer):

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