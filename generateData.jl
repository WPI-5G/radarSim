using Plots
using DSP
using FFTW
using Random
using JLD2

include("radarsim.jl")
using .radarsim;

print(Threads.nthreads());
print("\n")

dataPoints = 500;



available_env = ["Rectangular", "HalfSin"];
available_pulse = ["Increasing", "Decreasing"];
prf = 100

samp_rate = 5e6  #hz
fc = 900_000_000; #Hz

Gt = 40; #dB
Gr = 45; #dB
system_loss = 6; #dB

repetitions = 1;
duty_factor = 0.15; #rand((0.0:1e-3:0.5))

inputs = zeros(Float64, dataPoints, round(Int, samp_rate/prf));
in = ReentrantLock();
actuals = zeros(Float64, dataPoints);
act = ReentrantLock();
lkPrnt = ReentrantLock();

Threads.@threads for i in 1:dataPoints
    lock(lkPrnt) do 
        print(Threads.threadid())
        print("\n")
    end
    β = rand((0.0:100:samp_rate/2)); #Pulse Bandwidth

    targets = [Target(rand((200e3:100:800e3)), 0)];
    snr = rand((-64:0));

    env_type= rand(available_env);
    pulse_type= rand(available_pulse);

    τ = duty_factor * (1/prf);
    λ = (1/fc) * c;

    t, pulse_train = gen_pulse_train(repetitions, samp_rate, prf, τ, β, envelope_type=env_type, pulse_type=pulse_type);

    rx_pulse_train = sim_return(samp_rate, pulse_train, targets, fc=fc, system_loss=system_loss, Gt=Gt, Gr=Gr, SNR=snr);

    null, p = single_pulse(samp_rate, prf, τ, β, envelope_type=env_type, pulse_type=pulse_type);
    pulse_compression = conv(rx_pulse_train, reverse(p[begin:round(Int, τ*samp_rate)]));
    pulse_compression = pulse_compression[end - length(rx_pulse_train) + 1: end];

    #actual = ((targets[1].range*2)/c) * samp_rate
    lock(act) do
        actuals[i] = targets[1].range;
    end
    lock(in) do
        inputs[i,:] = abs.(pulse_compression);
    end
end

jldsave("Training.jld2"; inputs)
jldsave("Targets.jld2" ; actuals)
