import numpy as np
import time
import os
import json

from src.network import Layer


class Experiment:

    __DESCR_FILE = 'experiment_descr.json'
    __LAYER_FILE = 'experiment_layer'
    __NPZ = 'experiment_results.npz'

    def __init__(self, inputs, layer, num_trials, dt, t_stop):
        self.layer = layer

        self.num_trials = num_trials
        self.dt = dt
        self.t_stop = t_stop
        self.num_t = np.arange(0.0, t_stop, dt).shape[0]

        self.inputs = inputs
        self.outputs = np.empty(inputs.shape)
        self.spike_times = []
        self.spike_neurons = []

        self.runtime = None

    def one_trial(self, i, int_noise_regen=True):
        out, _, f_binary, _ = self.layer.output(self.inputs[:, i, None],
                                                self.dt,
                                                self.t_stop,
                                                int_noise_regen)
        times, neurons = np.where(f_binary != 0)
        self.spike_times.append([times])
        self.spike_neurons.append([neurons])
        self.outputs[:, i] = out.flatten()

    def run(self, status_freq=10):
        start_time = time.time()
        loop_time = start_time

        for i in range(self.num_trials):
            self.one_trial(i)

            if i % status_freq == 0:
                print("Trial ", i)
                print("10 Iter time: ", time.time() - loop_time)
                print("Total time: ", time.time() - start_time)
                print("\n")
                loop_time = time.time()

        self.runtime = time.time() - start_time
        print("EXPERIMENT FINISHED: ", self.runtime)

    def save(self, path, experiment_name):
        folder_name = experiment_name
        folder_path = os.path.join(path, folder_name)

        while os.path.exists(folder_path):
            folder_name += "_c"
            folder_path = os.path.join(path, folder_name)
        os.makedirs(folder_path)

        with open(os.path.join(folder_path, Experiment.__DESCR_FILE), 'w')\
                as outfile:
            json.dump(self._as_dict(), outfile)

        self.layer.save(folder_path, Experiment.__LAYER_FILE)

        np.savez(open(os.path.join(folder_path, Experiment.__NPZ), 'wb'),
            inputs=self.inputs,
            outputs=self.outputs,
            spike_times=self.spike_times,
            spike_neurons=self.spike_neurons)

    @classmethod
    def load(cls, path: str, experiment_name: str) -> 'Experiment':
        folder_path = os.path.join(path, experiment_name)

        if not os.path.exists(folder_path):
            raise FileNotFoundError()
        
        in_dict = {}
        infile = open(os.path.join(folder_path, Experiment.__DESCR_FILE), 'r')
        in_dict = json.load(infile)
        num_trials = in_dict['num_trials']
        dt = in_dict['dt']
        t_stop = in_dict['t_stop']
        num_t = in_dict['num_t']
        infile.close()

        layer = Layer.load(folder_path, Experiment.__LAYER_FILE)

        in_npys = np.load(open(os.path.join(folder_path, Experiment.__NPZ),
                               'rb'), allow_pickle=True)
        inputs = in_npys['inputs']
        outputs = in_npys['outputs']
        spike_times = in_npys['spike_times']
        spike_neurons = in_npys['spike_neurons']

        experiment = cls(inputs, layer, num_trials, dt, t_stop)
        experiment.num_t = num_t
        experiment.outputs = outputs
        experiment.spike_times = spike_times
        experiment.spike_neurons = spike_neurons

        return experiment

    def _as_dict(self):
        return {
            'num_trials': self.num_trials,
            'dt': self.dt,
            't_stop': self.t_stop,
            'num_t': self.num_t
        }

    def spike_graph(self):
        spikes = np.zeros((self.num_t, self.layer.NUM_NEURONS, self.num_trials))

        for i in range(self.num_trials):
            spikes[self.spike_times[i], self.spike_neurons[i], i] = 1.0

        return spikes

    # class MultiVariableExperiment:
    #
    #     def __init__(self, variables: dict, num_trials, dt, t_stop):
    #         """
    #
    #         Args:
    #             variables (dict): Dictionary of form
    #                 {'var1': {'start': , 'stop': , 'step': }, ...}
    #             inputs:
    #             layers:
    #             num_trials:
    #             dt:
    #             t_stop:
    #         """
    #         self.vars = variables
    #         # Ind of list = dim when saving
    #         self.var_dim = list(self.vars.keys())
    #
    #         self.experiments = []
    #
    #         self.dt = dt
    #         self.t_stop = t_stop
    #
    #     def one_trial(self, ):
