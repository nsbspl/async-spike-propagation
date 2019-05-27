import numpy as np
import time

class Experiment:
    def __init__(self, inputs, layer, num_trials, dt, t_stop):
        self.layer = layer
        
        self.num_trials = num_trials
        self.dt = dt
        self.t_stop = t_stop
        
        self.inputs = inputs
        self.outputs = np.empty(inputs.shape)
        self.spike_times = []
        self.spike_neurons = []
        
        self.runtime = None

    def run(self, status_freq=10):
        start_time = time.time()
        loop_time = start_time

        for i in range(self.num_trials):
            out, _, f_binary, _ = self.layer.output(self.inputs[:, i, None], self.dt, self.t_stop)
            times, neurons = np.where(f_binary != 0)
            self.spike_times.append([times])
            self.spike_neurons.append([neurons])

            self.outputs[:, i] = out.flatten()

            if i % status_freq == 0:
                print("Trial ", i)
                print("10 Iter time: ", time.time() - loop_time)
                print("Total time: ", time.time() - start_time)
                print("\n")
                loop_time = time.time()
        
        self.runtime = time.time() - start_time
        print("EXPERIMENT FINISHED: ", self.runtime)
