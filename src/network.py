import json
import math
import numpy as np
import os
import torch
import time

from scipy.signal import find_peaks
from src.lif import lif_compute, spike_binary, id_synaptic_waveform, gaussian_kernel
from src.ou_process import ouprocess_gaussian
from src.spike_sync import find_synced_spikes


class Layer:

    def __init__(self, num_neurons, std_noise=25.0, device="cpu"):
        self._device = torch.device("cuda" if (torch.cuda.is_available() and device=="cuda") else "cpu")

        self.NUM_NEURONS = num_neurons
        self.tau_V = 10
        self.R = 1.0 # MOhm
        self.EL = -70.0
        self.V_th = -40.0
        self.std_noise = std_noise
        
        self.W = torch.zeros(self.NUM_NEURONS, 1, dtype=torch.double, device=self._device)
        self.train_input = None
        self.train_exp_output = None

        self.v_E = 0.0
        self.v_ave = -67.0

        self.tau_rise = 0.5
        self.tau_fall = 5.0
        self.syn_kernel_len = 50.0

        self._ETA = None

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

        V = lif_compute(I_total, self.R, self.tau_V, self.V_th, dt,
                        grad=grad, device=self._device)
        F_binary = spike_binary(V, grad=grad, device=self._device)

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
            W=self.W.numpy(),
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

class PropogationNetwork(Layer):

    def __init__(self, depth, num_neurons, std_noise=25.0, device="cpu"):
        super().__init__(num_neurons, std_noise, device="device")

        self.depth = depth

    def output(self, i_inj, dt, t_stop, int_noise_regen=True, grad=False):
        out = i_inj
        V = None
        F_binary = None
        F_synaptic = None

        for layer_num in range(self.depth):
            out, V, F_binary, F_synaptic =\
                super().output(out, dt, t_stop, int_noise_regen=True, grad=grad)
        
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

    def __init__(self, num_neurons, std_noise=25.0, device="cpu"):
        super().__init__(num_neurons, std_noise=std_noise, device=device)
        self.W = torch.zeros(self.NUM_NEURONS, self.NUM_NEURONS, dtype=torch.double, device=self._device)


class FullyConnectedLayerApprox(_FullyConnectedLayer):

    @classmethod
    def from_layer(cls, layer: Layer) -> 'FullyConnectedLayerApprox':
        fcla = FullyConnectedLayerApprox(layer.NUM_NEURONS, layer.std_noise)

        fcla.tau_V = layer.tau_V
        fcla.R = layer.R
        fcla.EL = layer.EL
        fcla.V_th = layer.V_th
        
        fcla.W = np.random.normal(np.mean(layer.W), np.std(layer.W), fcla.W.shape)
        fcla.train_input = layer.train_input
        fcla.train_exp_output = layer.train_exp_output

        fcla.v_E = layer.v_E
        fcla.v_ave = layer.v_ave

        fcla.tau_rise = layer.tau_rise
        fcla.tau_fall = layer.tau_fall
        fcla.syn_kernel_len = layer.syn_kernel_len

        fcla._ETA = layer._ETA

        return fcla


class FullyConnectedLayer(_FullyConnectedLayer):

    def __init__(self, num_neurons, std_noise=25.0, device="cpu"):
        super().__init__(num_neurons, std_noise=std_noise, device=device)
        self.W = torch.as_tensor(
            self.W, device=self._device
        ).requires_grad_(True)

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


class PropagationNetworkFC:

    def __init__(self, depth, num_neurons, std_noise=25.0, device="cpu"):
        self.layers = []
        self.depth = depth
        self._device = torch.device("cuda" if (torch.cuda.is_available() and device=="cuda") else "cpu")
        for d in range(depth):
            self.layers.append(FullyConnectedLayer(num_neurons, std_noise, device=self._device))
    
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
