function gg = NeuronType(SamplingTime, EndTime, type_)

%disp('Neuron Characteristics');
p.A = 4 * 1 * 5^2; % surface area of neuron (um^2)
p.tStop = EndTime*1e3;  % mSec
p.dt = SamplingTime;     % msec
p.cm_M = 2;    % uF/cm^2
p.Vrest = -70; 

%tt = 0:p.dt:p.tStop;

    if strcmp(type_,'CD')==1
    p.EL_M = -70;
    p.rL_M = 500;%500; %ohm cm^2 eq to gL = 2mS/cm^2
% K
    p.EK_M = -100;     % mV
    p.gKdrBar_M = 20   ;  % mS / cm^2
%Na
    p.ENa_M = 50;     % mV
    p.gNaBar_M = 20;  % mS / cm^2 %%% 35
%
    p.V1 = -10;%-1.2 ; %mV
    p.V2 = 18;
    p.V3 = -27;%-21;
    p.V4 = 10;
    p.Phi = 0.15;
    p.exc_avg = 1.6;%0;%0.8;
    p.inh_avg = 1.2;%0;%0.6;
    p.scale_noise = 10;
    elseif strcmp(type_,'Integ')==1
        p.EL_M = -70;
    p.rL_M = 500; %ohm cm^2 eq to gL = 2mS/cm^2
% K
    p.EK_M = -100;     % mV
    p.gKdrBar_M = 20   ;  % mS / cm^2
%Na
    p.ENa_M = 50;     % mV
    p.gNaBar_M = 20;  % mS / cm^2
%
    p.V1 = -10 ; %mV
    p.V2 = 18;
    p.V3 = -10;
    p.V4 = 10;
    p.Phi = 0.15;
    p.exc_avg = 0.918;
    p.inh_avg = 1.882;
    p.scale_noise = 10;%10;
    else
  end

gg = p;
        


