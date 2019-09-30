import numpy as np

Trial_num = 1
tau_V = 10
R = 1  #Mohm
EL = -70
V_th = -30
dt = 0.1  # msec
t_stop = 10e3  # msec
tt = 0:dt:t_stop
L = length(tt)
param = NeuronType(dt, t_stop/1e3, 'Integ')
tw = 100  # msec as the kernel window
V = zeros(length(tt),Trial_num)
tau_rise = 0.5 # msec
tau_fall = 3 # msec
dt = param.dt
tEnd = 0.5 # sec
Gz = exp2syn_new(dt, tEnd, tau_rise, tau_fall)
EndTime = t_stop*1e-3 # sec

def tsodyks_synapse(self, V_, tau_Dep=50.0, tau_Fac=50.0, tau_syn=20.0, U=0.65):
    A = 1
    R = []
    u = []
    R_disc = [1]
    u_disc = [0]
    R_cont = [1, 1]
    u_cont = [0, 0]
    I_syn = np.zeros(1,L)
    tspk = [] # zeros(p.Net_size,1)
    Delta_syn = []
    Th = -40
    k=3  # start from t_indx = 3
    i=0
    for t in range(dt, t_stop-dt, dt):
        indx_active = np.where(V_[k-1] >= Th and V[k-2] < Th) # spike time for active neurons

        if len(indx_active) == 0:
            u_cont[k] = u_cont[k-1]*np.exp(-1.0*dt/tau_Fac) 
            R_cont[k] = 1 + (R_cont[k-1] - 1) * np.exp(-1.0*dt/tau_Dep) 
            I_syn(k) = I_syn(k-1)*exp(-dt/tau_syn)
        else
            i=i+1
            if i==1:
                tspk[i] = t
                Delta_syn[1] =  A * R_disc[1] * u_disc[1] 
            else
                tspk[i] = t

                u_disc[i] = u_disc[i-1]*np.exp(-1.0*(tspk[i]-tspk[i-1]) / tau_Fac ) + U*(1 - u_disc[i-1]*exp(-1.0*(tspk[i]-tspk[i-1]) / tau_Fac ))

                R_disc[i] = R_disc[i-1] * (1-u_disc[i]) * exp(-1.0*(tspk[i]-tspk[i-1]) / tau_Dep ) + 1 - exp(- (tspk[i]-tspk[i-1]) / tau_Dep )
                
                Delta_syn[i] =  A * R_disc[i] * u_disc[i] 
            end
                u_cont[k] = u_disc[i]
                R_cont[k] = R_disc[i]
                I_syn[k] = I_syn[k-1] + Delta_syn[i]
        end
        k=k+1
    
    end