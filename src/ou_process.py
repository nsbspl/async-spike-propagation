import numpy as np

def ouprocess_gaussian(tau, dt, t_stop, num_trials):
    ETA = np.zeros((int(t_stop/dt), num_trials))
    inp_sig = ETA.copy()
    N_tau = np.sqrt(2.0/tau)

    k = 0
    for t in np.arange(dt, t_stop, dt):
        inp_sig[k+1] = np.random.normal(size=num_trials)
        Einf = tau*N_tau*inp_sig[k]/np.sqrt(dt)
        ETA[k+1] = Einf + (ETA[k] - Einf)*np.exp(-dt/tau)
        k += 1
    return ETA, inp_sig

# import numpy as np
# import torch

# @torch.no_grad
# def ouprocess_gaussian(tau, dt, t_stop, num_trials):
#     ETA = torch.zeros(int(t_stop/dt), num_trials)
#     # ETA = np.zeros((int(t_stop/dt), num_trials))
#     inp_sig = ETA.clone()
#     N_tau = np.sqrt(2.0/tau)

#     k = 0
#     for t in np.arange(dt, t_stop, dt):
#         inp_sig[k+1] = torch.randn(num_trials) # np.random.normal(size=num_trials)
#         Einf = tau*N_tau*inp_sig[k]/np.sqrt(dt)
#         ETA[k+1] = Einf + (ETA[k] - Einf)*np.exp(-dt/tau)
#         k += 1
#     print("torchified")
#     return ETA, inp_sig
