

module radarsim

    export Target;
    export POW;
    export gen_pulse_train, sim_return, single_pulse;

    c = 299_792_458;  # the speed of light
    export c;

    struct Target
        range::Int          #Meters
        attenuation::Int    #dB
    end

    FSPL(d,f) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)  #Free Space Path Loss 
    POW(sig) = 10*log10(sum(abs.(sig).^2)/length(sig));     #Power of complex signal

    function single_pulse(samp_rate, prf, τ, β; target_pow = 0, envelope_type="Rectangular", pulse_type="Increasing", coefficients=[1])
        prt = 1/prf;
        t = 1/samp_rate:1/samp_rate:prt;

        if envelope_type == "Rectangular"
            a = ones(length(t));
            if(length(a) > τ * samp_rate)
                a[round(Int, τ*samp_rate):end] .= 0;
            end
        elseif envelope_type == "Gaussian"
            a = exp.((-t.^2)/(τ^2)) #Gaussian
        elseif envelope_type == "HalfSin"
            a = zeros(length(t));
            a[1:round(Int, τ * samp_rate )] = sin.(1/(τ/pi).*t[1:round(Int, τ * samp_rate )]);
        end

        if pulse_type == "Increasing"
            pulse = a.*exp.(im*π*(β/τ)*t.^2)
        elseif pulse_type == "Decreasing"
            pulse = a.*exp.(-im*π*β/τ*(t.^2 - 2*τ.*t))
        elseif pulse_type == "NOrderPoly"
            # coefficients = [3, 62, 10];
            coefficients = coefficients.*(β/τ);
            # pushfirst!(coefficients, 0);
            pwr = Array((1:length(coefficients)))';
            f = (t.^pwr).*coefficients';
            f = vec(sum(f, dims=2));
            pulse = a.*exp.(im*π.*f.*t);
        end

        pulse = pulse * 10^((target_pow - POW(pulse))/20)

        return t, pulse;
    end

    function gen_pulse_train(repetitions, samp_rate, prf, τ, β; envelope_type="Rectangular", pulse_type="Increasing", coefficients=[1])
        pulse = Array{ComplexF64}(undef,round(Int, 1/prf * repetitions * samp_rate));
        t, p = single_pulse(samp_rate, prf, τ, β, envelope_type=envelope_type, pulse_type=pulse_type, coefficients=coefficients);

        for i in 0:repetitions-1
            idx = i*length(t)
            pulse[idx+1:idx+length(t)] = p;
        end

        t = 1/samp_rate:1/samp_rate:1/prf*repetitions;
        return t, pulse;
    end

    function sim_return(samp_rate, pulse_train, targets::Array{Target}; fc = Int(900e6), system_loss=6, Gt=20, Gr=20, SNR = 100)
        rx_pulse_train = zeros(ComplexF64, length(pulse_train))
        num_targets = length(targets);
        for i in 1:num_targets
            range = targets[i].range

            pathloss = FSPL(range*2,fc);
            total_attenuation = pathloss - Gt - Gr + system_loss + targets[i].attenuation;
            rx_pow = POW(pulse_train) - total_attenuation;
            target_pulse_train = pulse_train * 10^((rx_pow - POW(pulse_train))/20)    
            
            
            t_delay = (range * 2) / c;
            ϕ = ((t_delay % (1/samp_rate)) / (1/samp_rate)) * 2π;
            offset = Int(t_delay ÷ (1/samp_rate));
            target_pulse_train = abs.(target_pulse_train) .* exp.(im*(angle.(target_pulse_train) .+ ϕ))
            
            rx_pulse_train[offset+1:end] = rx_pulse_train[offset+1:end] .+ target_pulse_train[begin:end-offset] #TODO: Why +1        
        end
        
        noise = rand(ComplexF64, length(rx_pulse_train));        
        noise_pwr = POW(rx_pulse_train) - SNR;
        noise = noise * 10^((noise_pwr - POW(noise)/20))
        
        rx_pulse_train = rx_pulse_train .+ noise;

        return rx_pulse_train, noise
    end
end