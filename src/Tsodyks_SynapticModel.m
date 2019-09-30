clear all
addpath('C:\Users\User\OneDrive - UHN\General_Codes\Other')
addpath('C:\Users\User\OneDrive - UHN\General_Codes\IEEE_CIM')
addpath('C:\Users\User\OneDrive - UHN\General_Codes\Info_Transmission')
addpath('C:\Users\User\OneDrive - UHN\General_Codes\Information Theory and Neuroscience')
%--------------------------------------------------------------------------
%% Pre-settings
Trial_num = 1;
tau_V = 10;
R = 1; %Mohm
EL = -70;
V_th = -30;
dt = 0.1; % msec
t_stop = 10e3; % msec
tt = 0:dt:t_stop;
L = length(tt);
param = NeuronType(dt, t_stop/1e3, 'Integ');
tw = 100; % msec as the kernel window
V = zeros(length(tt),Trial_num);
tau_rise = 0.5;% msec
tau_fall = 3;% msec
dt = param.dt;
tEnd = 0.5;% sec
Gz = exp2syn_new(dt, tEnd, tau_rise, tau_fall);
EndTime = t_stop*1e-3;% sec
%--------------------------------------------------------------------------
%% Make the DBS and Other related currents
%--- DBS pulses
IDBS_nA = zeros(L,1);
td_dbs = 0;
F_DBS = 15; % Hz
t_samp = 1:round(1e3/F_DBS/param.dt):param.tStop/param.dt;
width_pulse = 0.2; % msec --> the pulse width of the DBS
dt = param.dt;
pulse_kernel = zeros(1/dt,1);
pulse_kernel(td_dbs/dt+1:(td_dbs+width_pulse)/dt) = 1;
sig = zeros(L,1); sig(t_samp) = 1;
sig_con1 = conv(sig,pulse_kernel); % convolution with the square pulse
% Gz = exp2syn(dt,0.5,2);        % convolution with the synaptic waveform  
% sig_con2 = conv(sig_con1,Gz);
sig_filter = sig_con1(1:L);% sig_con2(1:L);
sig_filter(1:500) = 0;% not initiating from time = 0
figure; plot(sig_filter)
% I_DBS = 0.1* sig_filter;
IDBS_nA = sig_filter;
%--------------------------------------------------------------------------
%% Make pre-synaptic spikes (in response to the DBS-like input)
V_syn = 0; % as the cortical neurons are Exc
V_rest = -70;
V_th = -40;
%I_tot = g_tot.* (V_syn - V_rest);
V_ = LIF_Simple(IDBS_nA*1e4, R, tau_V, V_th, dt);
figure; plot(tt,V_)
ylabel('Voltage (mV)')
xlabel('Time (msec)')
%--------------------------------------------------------------------------
%% Investigate Synaptic plasticity (EPSP-IPSP-Isyn-Isyn by condunctances)
A = 1;
tau_Dep = 50; % msec
tau_Fac = 50;
tau_syn = 20;
U = 0.65;
R = [];
u = [];
R_disc(1) = 1;
u_disc(1) = 0;
R_cont(1:2) = 1;
u_cont(1:2) = 0;
I_syn = zeros(L,1);
tspk = [];% zeros(p.Net_size,1);
Delta_syn = [];
Th = -40;
k=3; % start from t_indx = 3;
i=0;
for t = dt:dt:t_stop-dt
    
    
    indx_active = find( V_(k-1)>=Th & V_(k-2)<Th ); % spike time for active neurons
    %indx_active = find( F_(k-1)>0 );
    
    if isempty(indx_active)==1
        
        u_cont(k) = u_cont(k-1)* exp(- dt/ tau_Fac ) ;
        
        R_cont(k) = 1 + (R_cont(k-1) - 1) * exp(- dt/ tau_Dep ) ;
        
        I_syn(k) = I_syn(k-1)*exp(-dt/tau_syn);
    else
        i=i+1;
        if i==1
           tspk(i) = t;
           Delta_syn(1) =  A * R_disc(1) * u_disc(1) ;
        else
           tspk(i) = t; 
           
           u_disc(i) = u_disc(i-1)*exp(- (tspk(i)-tspk(i-1)) / tau_Fac ) + U*(1 - u_disc(i-1)*exp(- (tspk(i)-tspk(i-1)) / tau_Fac ));
           
           R_disc(i) = R_disc(i-1) * (1-u_disc(i)) * exp(- (tspk(i)-tspk(i-1)) / tau_Dep ) + 1 - exp(- (tspk(i)-tspk(i-1)) / tau_Dep );
           
           Delta_syn(i) =  A * R_disc(i) * u_disc(i) ;
        end
        u_cont(k) = u_disc(i);
        R_cont(k) = R_disc(i);
        I_syn(k) = I_syn(k-1) + Delta_syn(i);
    end

 k=k+1;
 
end
figure; plot(I_syn)
figure; hold on,
plot(R_cont)
plot(u_cont,'k')
%% Calculate the firing rate in response to the total input
% V_syn = 0; % as the cortical neurons are Exc
% V_rest = -70;
% V_th = -40;
% I_tot = g_tot.* (V_syn - V_rest);
% V_ = LIF_Simple(I_tot, R, tau_V, V_th, dt);
% figure; plot(tt,V_)
% ylabel('Voltage (mV)')
% xlabel('Time (msec)')
% %% Make the Connectivity Matrix
% % --- Note that, we use a trick, i.e., make the transpose of connectivity
% % matrix (rows --> receiver & cols --> sender) and then transpose it to the original version where rows --> sender & cols --> receiver
% Pool_CTX = 1000;% all cortical neurons
% % --- For randomly selecting the indecies --> [1 + floor(N*rand)] generate numbers (uniform dist) between [1 to N]
% N_CTX = 400;%1000; 
% N_STN = 100;%20;
% N_GPe = 200;%40;
% N_Total = N_CTX + N_STN + N_GPe;
% Con_Matrix = zeros(N_Total,N_Total); % rows --> sender & cols --> receiver
% % Con_Matrix = [Ctx(1000,1000),       Ctx2Stn(1000,10),        Ctx2Gpe(1000,20);
% %               Stn2Ctx(10,1000);     Stn(10,10),              Stn2Gpe(10,20);
% %               Gpe2Ctx(20,1000);     Gpe2Stn(20,10)           Gpe(20,20)]; 
% Conn_tr_Matrix = Con_Matrix;
% % N_preSynInp_CTX2STN = 20; N_preSynInp_CTX2GPe = 20;
% % N_preSynInp_STN2STN = 5; N_preSynInp_STN2GPe = 10;
% % N_preSynInp_GPe2STN = 10; N_preSynInp_GPe2GPe = 10;
% 
% N_preSynInp_CTX2STN = 0.05*N_CTX; N_preSynInp_CTX2GPe = 0.025*N_CTX;
% N_preSynInp_STN2STN = 0.0*N_STN; N_preSynInp_STN2GPe = 0.06*N_STN;
% N_preSynInp_GPe2STN = 0.15*N_GPe; N_preSynInp_GPe2GPe = 0.00*N_GPe;
% 
% %--- for CTX Neurons
% for i=1:N_CTX
%     %--- Received from CTX
%     % all zeros as there is no interaction between CTX neurons
%     %--- Received from STN
%     % all zero
%     %--- Received from GPe
%     % all zero
% end
% %--- for STN Neurons
% for i=N_CTX+1:N_CTX+N_STN
%     %--- Received from CTX
%     indx_rand_CTX2STN = floor(N_CTX * rand(N_preSynInp_CTX2STN,1)); 
%     indx_rand_CTX2STN = 1 + indx_rand_CTX2STN;
%     Conn_tr_Matrix(i,indx_rand_CTX2STN) = 1;
%     %--- Received from STN
%     indx_rand_STN2STN = floor(N_STN * rand(N_preSynInp_STN2STN,1)); 
%     indx_rand_STN2STN = 1 + N_CTX + indx_rand_STN2STN;
%     Conn_tr_Matrix(i,indx_rand_STN2STN) = 1;
%     %--- Received from GPe
%     indx_rand_GPe2STN = floor(N_GPe * rand(N_preSynInp_GPe2STN,1));
%     indx_rand_GPe2STN = 1 + N_CTX + N_STN + indx_rand_GPe2STN;
%     Conn_tr_Matrix(i,indx_rand_GPe2STN) = 1;
% end
% %--- for Gpe Neurons
% for i=N_CTX+N_STN+1:N_CTX+N_STN+N_GPe
%     %--- Received from CTX
%     indx_rand_CTX2GPe = floor(N_CTX * rand(N_preSynInp_CTX2GPe,1)); 
%     indx_rand_CTX2GPe = 1 + indx_rand_CTX2GPe;
%     Conn_tr_Matrix(i,indx_rand_CTX2GPe) = 1;
%     %--- Received from STN
%     indx_rand_STN2GPe = floor(N_STN * rand(N_preSynInp_STN2GPe,1));  
%     indx_rand_STN2GPe = 1 + N_CTX + indx_rand_STN2GPe;
%     Conn_tr_Matrix(i,indx_rand_STN2GPe) = 1;
%     %--- Received from GPe
%     indx_rand_GPe2GPe = floor(N_GPe * rand(N_preSynInp_GPe2GPe,1));  
%     indx_rand_GPe2GPe = 1 + N_CTX + N_STN + indx_rand_GPe2GPe;
%     Conn_tr_Matrix(i,indx_rand_GPe2GPe) = 1;
% end
% %%%% Consider here that it should be off diagonal
% 
% Con_Matrix = (Conn_tr_Matrix)';
% % --- Note that we should activate the CTX because they send signal to STN and GPe
% % Con_Matrix(1:N_CTX,N_CTX+1:N_Total) = 1;
% %% Reversal Potential, Synaptic Strength, time delays and time constants of the syanpses
% %--- Reversal Ppotential (mV)
% E_syn = zeros(size(Con_Matrix)); 
% E_syn(1:N_CTX,:) = 0; % All CTX are exc
% E_syn(N_CTX+1:N_CTX + N_STN,:) = 0; % All STN are exc
% E_syn(N_CTX+N_STN+1:N_Total,:) = -80; % All GPe are inh
% E_syn = E_syn.*Con_Matrix;
% %--- Synaptic Strength (nS)
% G_syn = zeros(size(Con_Matrix));
% G_syn(1:N_CTX,N_CTX+1:N_CTX+N_STN) = 7.5*0.05; % The strength of CTX2STN (note that we finally multiply by Con_Matrix)
% G_syn(1:N_CTX,N_CTX+N_STN+1:end) = 2.5*0.05; % The strength of CTX2GPe
% G_syn(N_CTX+1:N_CTX + N_STN,N_CTX+1:N_CTX + N_STN) = 0*0.05; % STN2STN
% G_syn(N_CTX+1:N_CTX + N_STN,N_CTX+N_STN+1:N_Total) = 5*0.05; % STN2GPe
% G_syn(N_CTX+N_STN+1:N_Total,N_CTX+1:N_CTX + N_STN) = 5*0.05; % GPe2STN
% G_syn(N_CTX+N_STN+1:N_Total,N_CTX+N_STN+1:N_Total) = 0*0.05; % GPe2GPe
% G_syn = G_syn.*Con_Matrix;
% %--- Delay (msec)
% t_delay = zeros(size(Con_Matrix));
% t_delay(1:N_CTX,:) = 3; % All CTX are exc
% t_delay(N_CTX+1:N_CTX + N_STN,:) = 3; % All STN are exc
% t_delay(N_CTX+N_STN+1:N_Total,:) = 5; % All GPe are inh
% t_delay = t_delay.*Con_Matrix;
% %--- time constant (msec)
% tau_cons = 1e3*ones(size(Con_Matrix)); %Time constant matrix (msec) --> we use 1e3 for no connections (avoide zeros in den)
% tau_cons(1:N_CTX,:) = 3; % All CTX are exc
% tau_cons(N_CTX+1:N_CTX + N_STN,:) = 3; % All STN are exc
% tau_cons(N_CTX+N_STN+1:N_Total,:) = 10; % All GPe are inh
% %% Making the Table_Exponential Function
% Type_syn = 2; % we have two types Exc (3 msec) and Inh (10 msec)
% Type_tau = [3 10];%
% Type_delay = [3 5];
% L_syn = 1000/param.dt; % 1000 msec
% Table_Exponential = zeros(L_syn,Type_syn); % 3D matrix
% t=param.dt:param.dt:L_syn*param.dt;
% for i=1:Type_syn % row (send)
%     alpha = t/Type_tau(i) .* exp(-t ./Type_tau(i));
%     alpha = alpha'/max(alpha);
%     L_delay = Type_delay(i)/param.dt; 
%     Table_Exponential(:,i) = [zeros(L_delay,1); alpha(1:L_syn-L_delay)];
% end
% 
% figure; plot(t,Table_Exponential)
% %% Make the paremters ready
% param.Con_Matrix = Con_Matrix;
% param.E_syn = E_syn; param.G_syn = G_syn;
% param.t_delay = t_delay; param.tau_cons = tau_cons;
% param.L_syn = L_syn; param.Net_size = size(G_syn,2);
% param.Table_Exponential = Table_Exponential;
% % param.spike_thresh = -40; % mV
% param.tau_V = tau_V;
% param.N_active = N_STN + N_GPe;
% param.N_STN = N_STN;
% Th_STN = -50; std_Th_STN = 2;% mV
% Th_GPe = -50; std_Th_GPe = 2;% mV
% param.spike_thresh = [(Th_STN + std_Th_STN*randn(N_STN,1));(Th_GPe + std_Th_GPe*randn(N_GPe,1))];
% %% Make the DBS and Other related currents
% %--- DBS pulses
% IDBS_nA = zeros(L,1);
% td_dbs = 0;
% F_DBS = 2; % Hz
% t_samp = 1:round(1e3/F_DBS/param.dt):param.tStop/param.dt;
% width_pulse = 0.2; % msec --> the pulse width of the DBS
% dt = param.dt;
% pulse_kernel = zeros(1/dt,1);
% pulse_kernel(td_dbs/dt+1:(td_dbs+width_pulse)/dt) = 1;
% sig = zeros(L,1); sig(t_samp) = 1;
% sig_con1 = conv(sig,pulse_kernel); % convolution with the square pulse
% % Gz = exp2syn(dt,0.5,2);        % convolution with the synaptic waveform  
% % sig_con2 = conv(sig_con1,Gz);
% sig_filter = sig_con1(1:L);% sig_con2(1:L);
% sig_filter(1:500) = 0;% not initiating from time = 0
% figure; plot(sig_filter)
% % I_DBS = 0.1* sig_filter;
% IDBS_nA = sig_filter;
% %--- DBS induced acivation currents (activate the STN pre-synaptic terminals)
% IDBS_Activation = zeros(size(IDBS_nA));
% Delay_Activation = 2/dt; % equivalent to 2 msec
% Width_Activation = 2/dt; % duration of activation
% indx_Activation_start = 1 + Delay_Activation + find(diff(IDBS_nA) == 1); % indices associated with the onset of each DBS pulse
% indx_Activation_end = indx_Activation_start + Width_Activation;
% IDBS_Activation(indx_Activation_start:indx_Activation_end) = 1; % we use this signal to activate STN synapses (GPe2STN and CTX2STN)
% figure; 
% plot(IDBS_Activation,'k')
% title('Activation Effect onto STN-terminals by DBS')
% %--- DBS induced Fast-Input (A model for generated spikes in the Axon of STNs --> received by GPe neurons)
% IDBS_Fast = zeros(L,N_STN + N_GPe);
% IDBS_Fast_GPe = zeros(L,N_GPe);
% Delay_Fast = 25/dt; % equivalent to 25 msec
% std_Jitter = 2/dt;
% indx_ones = find(diff(IDBS_nA) == 1);
% for k=1:N_GPe
%     indx_Fast_GPe = 1 + floor(Delay_Fast + std_Jitter*randn(length(indx_ones),1) + indx_ones);
%     IDBS_Fast_GPe(indx_Fast_GPe,k) = 1;
%     g_synaptic = 10 * 0.050;
%     g_tot = conv(IDBS_Fast_GPe(:,k),g_synaptic * Gz); g_tot = g_tot(1:L);
%     V_syn = 0; % for Exc synapses STN2GPe
%     V_rest = -70;
%     IDBS_Fast_GPe(:,k) = N_preSynInp_STN2GPe *g_tot.* (V_syn - V_rest);
% end
% IDBS_Fast(:,N_STN+1:N_STN + N_GPe) = IDBS_Fast_GPe;
% figure; 
% plot(IDBS_Fast(:,1:N_STN),'k')
% title('Fast event onto STN Neurons')
% figure; 
% plot(IDBS_Fast(:,1+N_STN:N_STN+N_GPe),'k')
% title('Fast event onto GPe Neurons')
% %% Run the network simulation
% %--- F_binary is the spikes generated by other neurons (inactive) like CTX
% % yy = snglComp_StnGpe_Network_LIF(IDBS_nA, F_binary, param);
% yy = snglComp_StnGpe_Network_LIF_Hypothesis_I(IDBS_nA, IDBS_Activation, IDBS_Fast, F_binary, param);
% 
% %% Plots
% 
% figure; hold on,
% ax(1) = subplot(3,1,1); plot(tt,yy(:,2),'b');title('CTX')
% ax(2) = subplot(3,1,2); plot(tt,yy(:,6),'r'); title('GPe')
% ax(3) = subplot(3,1,3); plot(tt,yy(:,5),'k');title('STN')
% xlabel('Time (msec)')
% linkaxes(ax,'x')
% 
% figure; hold on,
% ax(1) = subplot(3,1,1); plot(tt,yy(:,1),'b');title('CTX')
% ax(2) = subplot(3,1,2); plot(tt,yy(:,5),'r'); title('STN')
% ax(3) = subplot(3,1,3); plot(tt,yy(:,6),'k');title('GPe')
% xlabel('Time (msec)')
% linkaxes(ax,'x')
% 
% 
% 
% 
% 
