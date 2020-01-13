import json
import math
import os
from scipy import optimize

import numpy as np
import torch

from src.lif import lif_compute, spike_binary, id_synaptic_waveform, \
    gaussian_kernel
from src.ou_process import ouprocess_gaussian


class Layer:

    def __init__(self, num_neurons,
                std_noise=25.0,
                tau_fall=5.0):
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
        self.tau_fall = tau_fall
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
        
        return V, F_binary
    
    def synapse(self, F_binary, dt, t_stop):
        tt = np.arange(0.0, t_stop, dt)

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

        return F_synaptic

    def synaptic_weight(self, F_synaptic, t_steps):
        ind_neur = np.arange(0, self.NUM_NEURONS)
        Phi = F_synaptic[:t_steps, ind_neur]
        X2 = -1.0*self.v_ave*np.ones((t_steps,ind_neur.shape[0])) + self.v_E

        A = np.multiply(Phi, X2)
        out = np.dot(A, self.W)

        return out

    def output(self, i_inj, dt, t_stop, int_noise_regen=True):
        V, F_binary = self.spike(i_inj, dt, t_stop, int_noise_regen=True)
        F_synaptic = self.synapse(F_binary, dt, t_stop)
        
        t_steps = F_binary.shape[0]
        out = self.synaptic_weight(F_synaptic, t_steps)

        return out, V, F_binary, F_synaptic

    def firing_rate(self, F_binary, dt, t_stop, grad=False):
        torch.set_grad_enabled(grad)

        tt = np.arange(0.0, t_stop, dt)

        gauss_kernel = gaussian_kernel(dt, 25.0)
        gauss_kernel_len = gauss_kernel.shape[0]

        # CONVOLUTION EXPLAINED
        # kernel:
        #     out_channels = NUM_NEURONS
        #     in_channels / groups = 1
        #     kernel_size = gauss_kernel_len
        # input:
        #     minibatch = 1
        #     in_channels = NUM_NEURONS
        #     input_size = len(tt)
        # => groups = NUM_NEURONS (do SAME convolution SEPARATELY over each neuron spike train)

        gauss_kernel_tensor = torch.as_tensor(gauss_kernel).repeat(
            self.NUM_NEURONS, 1, 1)
        pad = math.ceil(gauss_kernel_len / 2.0)

        convolved_spikes = torch.nn.functional.conv1d(
            torch.as_tensor(F_binary).t()[None, :, :],
            gauss_kernel_tensor,
            groups=self.NUM_NEURONS,
            padding=pad)

        inst_firing_rate = convolved_spikes[0, :, :tt.shape[0]].t()

        # RESCALE FIRING RATE TO MATCH UNITS
        # mean firing rate should equal to mean of instantaneous firing rates
        _, spike_trial = np.where(F_binary > 0)
        mean_fr = spike_trial.shape[0] / self.NUM_NEURONS / (t_stop / 1.0e3)
        inst_fr_mean = inst_firing_rate.mean(0).mean(0).numpy()

        scaling_factor = mean_fr / inst_fr_mean

        return inst_firing_rate * scaling_factor

    def train(self, i_inj, exp_output, dt, t_stop):
        _, _, _, F_synaptic = self.output(i_inj, dt, t_stop)

        t_steps = exp_output.shape[0]
    
        ind_neur = np.arange(0, self.NUM_NEURONS)
        Phi = F_synaptic[:t_steps, ind_neur]
        X2 = -1.0*self.v_ave*np.ones((t_steps,ind_neur.shape[0])) + self.v_E

        A = np.multiply(Phi, X2)
        self.W, residuals, rank, s = np.linalg.lstsq(A, exp_output)
        # print(self.W.shape)
        # self.W, residuals = optimize.nnls(A, exp_output.flatten())
        # self.W = self.W[:, None]
        self.train_input = i_inj
        self.train_exp_output = exp_output

    def deepcopy() -> 'Layer':
        new_layer = Layer(self.num_neurons, self.std_noise, self.tau_fall)
        
        new_layer.tau_V = layer.tau_V
        new_layer.R = layer.R
        new_layer.EL = layer.EL
        new_layer.V_th = layer.V_th

        new_layer.W = layer.W
        new_layer.train_input = layer.train_input
        new_layer.train_exp_output = layer.train_exp_output

        new_layer.v_E = layer.v_E
        new_layer.v_ave = layer.v_ave

        new_layer.tau_rise = layer.tau_rise
        new_layer.tau_fall = layer.tau_fall
        new_layer.syn_kernel_len = layer.syn_kernel_len

        new_layer._ETA = layer._ETA

        return new_layer
    
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


class LayerTorched(Layer):
    def __init__(self, num_neurons, std_noise=25.0, tau_fall=5.0, device='cpu'):

        super().__init__(num_neurons, std_noise=std_noise, tau_fall=tau_fall)

        self._device = torch.device("cuda" if (torch.cuda.is_available() and device=="cuda") else "cpu")
        
        self.W = torch.zeros(self.NUM_NEURONS, 1, dtype=torch.double, device=self._device)
    
    def spike(self, i_inj, dt, t_stop, int_noise_regen=True, grad=False):
        torch.set_grad_enabled(grad)

        tt = np.arange(0.0, t_stop, dt)
        V = torch.zeros(tt.shape[0], self.NUM_NEURONS, requires_grad=grad, device=self._device) # Membrane potential per neuron

        # Additive noise to individual neurons
        if self._ETA is None \
                or int_noise_regen is True\
                or self._ETA.shape != V.shape:
            self._ETA, _ = ouprocess_gaussian(5.0, dt, t_stop, self.NUM_NEURONS)

        # F_binary = torch.zeros(tt.shape[0], self.NUM_NEURONS, requires_grad=grad, device=self._device)
        int_noise = torch.as_tensor(self.std_noise*self._ETA,
                                    device=self._device).requires_grad_(grad)
        # avg_firing_rate = np.zeros(self.NUM_NEURONS)

        I_total = int_noise + i_inj

        V = lif_compute(I_total, self.R, self.tau_V, self.V_th, dt, torched=True,
                        grad=grad, device=self._device)
        F_binary = spike_binary(V, torched=True, grad=grad, device=self._device)

        return V, F_binary

    def synapse(self, F_binary, dt, t_stop, grad=False):
        torch.set_grad_enabled(grad)

        tt = np.arange(0.0, t_stop, dt)

        syn_waveform = id_synaptic_waveform(
            dt,
            self.syn_kernel_len,
            self.tau_rise,
            self.tau_fall)
        syn_wave_len = syn_waveform.shape[0]

        # CONVOLUTION EXPLAINED
        # kernel:
        #     out_channels = NUM_NEURONS
        #     in_channels / groups = 1
        #     kernel_size = syn_wave_len
        # input:
        #     minibatch = 1
        #     in_channels = NUM_NEURONS
        #     input_size = len(tt)
        # => groups = NUM_NEURONS (do SAME convolution SEPARATELY over each neuron spike train)

        # F_synaptic = torch.zeros(F_binary.shape, device=self._device)
        syn_waveform_kernel = torch.as_tensor(syn_waveform).repeat(self.NUM_NEURONS, 1, 1).to(self._device)
        pad = math.ceil(syn_wave_len/2.0)

        fr_fast = torch.nn.functional.conv1d(
                F_binary.t()[None, :, :],
                syn_waveform_kernel,
                groups=self.NUM_NEURONS,
                padding=pad)

        F_synaptic = fr_fast[0, :, :tt.shape[0]].t()

        return F_synaptic

    def synaptic_weight(self, F_synaptic, t_steps, grad=False):
        torch.set_grad_enabled(grad)

        ind_neur = np.arange(0, self.NUM_NEURONS)
        Phi = F_synaptic[:t_steps, ind_neur]
        X2 = (-1.0*self.v_ave*torch.ones(t_steps,ind_neur.shape[0], device=self._device) + self.v_E).double()

        A = torch.mul(Phi, X2)
        out = torch.mm(A, self.W)

        return out

    def output(self, i_inj, dt, t_stop, int_noise_regen=True, grad=False):
        torch.set_grad_enabled(grad)
        i_inj_tensor = torch.as_tensor(i_inj, device=self._device)

        V, F_binary = self.spike(i_inj_tensor, dt, t_stop, int_noise_regen=True, grad=grad)
        F_synaptic = self.synapse(F_binary, dt, t_stop, grad=grad)

        t_steps = F_binary.shape[0]
        out = self.synaptic_weight(F_synaptic, t_steps, grad=grad)

        return out, V, F_binary, F_synaptic

    def firing_rate(self, F_binary, dt, t_stop, grad=False):
        torch.set_grad_enabled(grad)

        tt = np.arange(0.0, t_stop, dt)

        gauss_kernel = gaussian_kernel(dt, 25.0)
        gauss_kernel_len = gauss_kernel.shape[0]

        # CONVOLUTION EXPLAINED
        # kernel:
        #     out_channels = NUM_NEURONS
        #     in_channels / groups = 1
        #     kernel_size = gauss_kernel_len
        # input:
        #     minibatch = 1
        #     in_channels = NUM_NEURONS
        #     input_size = len(tt)
        # => groups = NUM_NEURONS (do SAME convolution SEPARATELY over each neuron spike train)

        # inst_firing_rate = torch.zeros(F_binary.shape, device=self._device)
        gauss_kernel_tensor = torch.as_tensor(gauss_kernel).repeat(self.NUM_NEURONS, 1, 1).to(self._device)
        pad = math.ceil(gauss_kernel_len/2.0)

        convolved_spikes = torch.nn.functional.conv1d(
                F_binary.t()[None, :, :],
                gauss_kernel_tensor,
                groups=self.NUM_NEURONS,
                padding=pad)

        inst_firing_rate = convolved_spikes[0, :, :tt.shape[0]].t()

        return inst_firing_rate

    def train(self, i_inj, exp_output, dt, t_stop):
        torch.set_grad_enabled(False)

        i_inj_tensor = torch.as_tensor(i_inj, device=self._device)
        V, F_binary = self.spike(i_inj_tensor, dt, t_stop, int_noise_regen=True, grad=False)
        F_synaptic = self.synapse(F_binary, dt, t_stop, grad=False)

        t_steps = F_binary.shape[0]

        ind_neur = np.arange(0, self.NUM_NEURONS)
        Phi = F_synaptic[:t_steps, ind_neur]
        X2 = (torch.ones(t_steps,ind_neur.shape[0], device=self._device)*-1.0*self.v_ave + self.v_E).double()

        A = torch.mul(Phi, X2)
        W_np, residuals, rank, s = np.linalg.lstsq(A.cpu(), exp_output)
        self.W = torch.as_tensor(W_np, dtype=torch.double, device=self._device)

        # _, _, _, F_synaptic = self.output(i_inj, dt, t_stop)

        # t_steps = exp_output.shape[0]
    
        # ind_neur = np.arange(0, self.NUM_NEURONS)
        # Phi = F_synaptic[:t_steps, ind_neur]
        # X2 = -1.0*self.v_ave*np.ones((t_steps,ind_neur.shape[0])) + self.v_E

        # A = np.multiply(Phi, X2)
        # self.W, residuals, rank, s = np.linalg.lstsq(A, exp_output)
        self.train_input = i_inj
        self.train_exp_output = exp_output

    def save(self, path, layer_name):
        LAYER_ATTRS_JSON = layer_name + "_attrs.json"
        LAYER_WEIGHTS_NPZ = layer_name + "_weights.npz"

        with open(os.path.join(path, LAYER_ATTRS_JSON), 'w') as outfile:
            json.dump(self.as_dict(), outfile)

        w_new = torch.ones(self.W.shape)
        w_new = w_new.copy_(self.W)
        np.savez(open(os.path.join(path, LAYER_WEIGHTS_NPZ), 'wb'),
            W=w_new.detach().numpy(),
            train_input=self.train_input,
            train_exp_output=self.train_exp_output)

    @classmethod
    def load(cls, path: str, layer_name: str, device="cpu") -> 'Layer':
        in_dict = {}
        LAYER_ATTRS_JSON = layer_name + "_attrs.json"
        LAYER_WEIGHTS_NPZ = layer_name + "_weights.npz"

        with open(os.path.join(path, LAYER_ATTRS_JSON), 'r') as infile:
            in_dict = json.load(infile)
        layer = cls.from_dict(in_dict)
        layer._device = torch.device("cuda" if (torch.cuda.is_available() and device=="cuda") else "cpu")

        data = np.load(open(os.path.join(path, LAYER_WEIGHTS_NPZ), 'rb'))
        layer.W = torch.as_tensor(data['W'], device=layer._device)
        layer.train_input = data['train_input']
        layer.train_exp_output = data['train_exp_output']

        return layer


class BasePropagationNetwork:

    def __init__(self, layer_cls, depth, num_neurons, std_noise=25.0):
        self.layer = layer_cls(num_neurons, std_noise)

        self.depth = depth

    @classmethod
    def from_layer(cls, layer: Layer, depth: int) -> 'BasePropagationNetwork':
        prop_ntwrk = PropagationNetwork(type(layer), depth, layer.NUM_NEURONS, layer.std_noise)
        prop_ntwrk.layer = layer.deepcopy()

        return prop_ntwrk

    def as_dict(self):
        props_dict = self.layer.as_dict()
        props_dict['depth'] = self.depth

        return props_dict

    @classmethod
    def from_dict(cls, layer_cls, in_dict: dict) -> 'BasePropagationNetwork':
        NUM_NEURONS = in_dict['NUM_NEURONS']
        depth = in_dict['depth']
        std_noise = in_dict['std_noise']

        network = cls(layer_cls, depth, NUM_NEURONS, std_noise)
        layer = layer_cls.from_dict(in_dict)
        network.layer = layer

        return network

    # TODO: load, save


class PropagationNetwork(BasePropagationNetwork):

    def __init__(self, depth, num_neurons, std_noise=25.0):
        super().__init__(Layer, depth, num_neurons, std_noise)

    def output(self, i_inj, dt, t_stop, int_noise_regen=True):
        out = i_inj
        V = None
        F_binary = None
        F_synaptic = None

        list_outs = []
        list_V = []
        list_F_binary = []
        list_F_synaptic = []

        for layer_num in range(self.depth):
            out, V, F_binary, F_synaptic =\
                self.layer.output(out, dt, t_stop, int_noise_regen=True)

            list_outs.append(out)
            list_V.append(V)
            list_F_binary.append(F_binary)
            list_F_synaptic.append(F_synaptic)

            self.layer.W = np.random.normal(np.mean(self.layer.W), np.std(self.layer.W), self.layer.W.shape)

        # So that we have resulting firing rate of last layer's synaptic summation
        _, spike_out = self.layer.spike(out, dt, t_stop, int_noise_regen=True)
        list_F_binary.append(spike_out)

        return list_outs, list_V, list_F_binary, list_F_synaptic


class PropagationNetworkTorched(BasePropagationNetwork):

    def __init__(self, depth, num_neurons, std_noise=25.0):
        super().__init__(LayerTorched, depth, num_neurons, std_noise)

    def output(self, i_inj, dt, t_stop, int_noise_regen=True, grad=False):
        out = i_inj
        V = None
        F_binary = None
        F_synaptic = None

        list_outs = []
        list_V = []
        list_F_binary = []
        list_F_synaptic = []

        for layer_num in range(self.depth):
            out, V, F_binary, F_synaptic =\
                super().output(out, dt, t_stop, int_noise_regen=True, grad=grad)
            
            list_outs.append(out)
            list_V.append(V)
            list_F_binary.append(F_binary)
            list_F_synaptic.append(F_synaptic)
        
        # So that we have resulting firing rate of last layer's synaptic summation
        _, spike_out, _ = self.layer.spike(out, dt, t_stop, int_noise_regen=True)
        list_F_binary.append(spike_out)

        return list_outs, list_V, list_F_binary, list_F_synaptic

    
class FullyConnectedLayerApprox(Layer):

    def __init__(self, num_neurons, std_noise=25.0):
        super().__init__(num_neurons, std_noise=std_noise)
        self.W = np.zeros((self.NUM_NEURONS, self.NUM_NEURONS))

    @classmethod
    def from_layer(cls, layer: Layer) -> 'FullyConnectedLayerApprox':
        fcl = FullyConnectedLayerApprox(layer.NUM_NEURONS, layer.std_noise)

        fcl.tau_V = layer.tau_V
        fcl.R = layer.R
        fcl.EL = layer.EL
        fcl.V_th = layer.V_th
        
        fcl.W = np.random.normal(np.mean(layer.W), np.std(layer.W), fcl.W.shape)
        fcl.train_input = layer.train_input
        fcl.train_exp_output = layer.train_exp_output

        fcl.v_E = layer.v_E
        fcl.v_ave = layer.v_ave

        fcl.tau_rise = layer.tau_rise
        fcl.tau_fall = layer.tau_fall
        fcl.syn_kernel_len = layer.syn_kernel_len
        
        fcl._ETA = layer._ETA

        return fcl


class FullyConnectedLayerTorched(LayerTorched):

    def __init__(self, num_neurons, std_noise=25.0, device="cpu"):
        super().__init__(num_neurons, std_noise=std_noise, device=device)
        self.W = torch.as_tensor(
            np.zeros((self.NUM_NEURONS, self.NUM_NEURONS)), device=self._device
        ).requires_grad_(True)

    @classmethod
    def load(cls, path: str, layer_name: str, device="cpu") -> 'Layer':
        fcl = super().load(path, layer_name, device)
        fcl.W = fcl.W.double()
        return fcl

    def train(self, i_inj, exp_output, dt, t_stop, num_iters=15):
        torch.set_grad_enabled(True)
        losses = []

        self.train_input = i_inj
        self.train_exp_output = exp_output

        optimizer = torch.optim.Adam([self.W])

        start_time = time.time()

        for t in range(num_iters): #500
            loop_start_time = time.time()
            curr_losses = []

            # Forward pass: compute predicted y by passing x to the model.
            batch_tensor = torch.as_tensor(i_inj, device=self._device)

            out, _, _, _ = self.output(batch_tensor, dt, t_stop,
                                       int_noise_regen=True, grad=True)

            # Compute and print loss.
            loss = self.loss(torch.tensor(exp_output, dtype=torch.double, device=self._device),
                             torch.mean(out, dim=1, keepdim=True))

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            curr_losses.append(loss.item())

            avg_loss = np.mean(curr_losses)
            losses.append(avg_loss)

            print("ITER ", t, ":")
            print("Avg loss: ", avg_loss)
            print("Loop time: ", time.time()-loop_start_time)
            print("Total time: ", time.time()-start_time)
        
        return losses

    def loss(self, expected_output, actual_output):
        return torch.nn.functional.mse_loss(actual_output, expected_output) # nll-loss


class PropagationNetworkFCTorched:

    def __init__(self, depth, num_neurons, std_noise=25.0, device="cpu"):
        self.layers = []
        self.depth = depth
        self._device = torch.device("cuda" if (torch.cuda.is_available() and device=="cuda") else "cpu")
        for d in range(depth):
            self.layers.append(FullyConnectedLayerTorched(num_neurons, std_noise, device=self._device))
    
    def output(self, i_inj, dt, t_stop, int_noise_regen=True, grad=False):
        torch.set_grad_enabled(grad)

        out = i_inj
        V = None
        F_binary = None
        F_synaptic = None

        for i in range(self.depth):
            out, V, F_binary, F_synaptic =\
                self.layers[i].output(out, dt, t_stop, int_noise_regen=True, grad=grad)

        return out, V, F_binary, F_synaptic

    def train(self, i_inj, exp_output, dt, t_stop, num_iters=15):
        torch.set_grad_enabled(True)
        losses = []

        self.train_input = i_inj
        self.train_exp_output = exp_output

        optimizer = torch.optim.Adam([layer.W for layer in self.layers])

        start_time = time.time()

        for t in range(num_iters): #500
            # Forward pass: compute predicted y by passing x to the model.
            # _, F_binary, F_synaptic = self.spike(i_inj, dt, t_stop, int_noise_regen=True, grad=True)
            # t_steps = F_binary.shape[0]

            # ind_neur = np.arange(0, self.NUM_NEURONS)
            # Phi = F_synaptic[:t_steps, ind_neur]
            # X2 = (-1.0*self.v_ave*torch.ones(t_steps,ind_neur.shape[0]) + self.v_E).double()

            # A = torch.mul(Phi, X2)
            loop_start_time = time.time()
            out, _, _, _ = self.output(i_inj, dt, t_stop, int_noise_regen=True, grad=True)

            # Compute and print loss.
            loss = self.loss(torch.as_tensor(exp_output, dtype=torch.double,
                                           device=self._device),
                             torch.mean(out, dim=1, keepdim=True).to(self._device))
            
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            losses.append(loss.item())

            print("ITER ", t, ":")
            print("Loss: ", loss.item())
            print("Loop time: ", time.time()-loop_start_time)
            print("Total time: ", time.time()-start_time)
        
        return losses

    def loss(self, expected_output, actual_output):
        return torch.nn.functional.mse_loss(actual_output, expected_output) # nll-loss
