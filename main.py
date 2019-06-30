import time

import seaborn as sns

from src.network import FullyConnectedLayer, Layer
from src.ou_process import ouprocess_gaussian

sns.set()

if __name__ == '__main__':
    NUM_NEURONS = 200
    tau_V = 10
    R = 1  # MOhm
    EL = -70.0
    V_th = -40.0
    dt = 0.1  # msec
    t_stop = 3.0e3  # 30.0e3

    RESULTS_DIR = "./results"
    GRAPHS_DIR = "./graphs"

    # Slow Signal: INPUT
    start_time = time.time()
    input_slow, _ = ouprocess_gaussian(50.0, dt, t_stop, 1)
    i_inj = 16.0 + 6.0 * input_slow
    print(time.time() - start_time)

    fcl = FullyConnectedLayer(NUM_NEURONS, device="cpu")
    losses = fcl.train(i_inj=i_inj, exp_output=i_inj, dt=dt, t_stop=t_stop,
                       num_iters=15)

    fcl.save("./results", "fcl_neurons=200")
    np.save(open("./results/losses.npy", 'wb'), losses)
    # layer = Layer(NUM_NEURONS, device="cpu")
    # layer.train(i_inj=i_inj, exp_output=i_inj, dt=dt, t_stop=t_stop)
    #
    # out, V, F_binary, F_synaptic = layer.output(i_inj, dt, t_stop)
    # _, spike_out = layer.spike(out, dt, t_stop, int_noise_regen=True)
