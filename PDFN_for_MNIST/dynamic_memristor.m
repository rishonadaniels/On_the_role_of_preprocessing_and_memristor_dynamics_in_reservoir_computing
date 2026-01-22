function [w_out, I_out] = dynamic_memristor(w_in, V, t_pulse, tau, variability)
    % Constants
    w_max = 1;
    w_min = 0.1;
    alpha = 1e-8;
    beta = 0.5;
    gamma = 1e-5;
    delta = 4;
    lambda = 2000;
    eta = 8;
    tau_1 = tau;

    % Mask for update vs leak
    is_update = V > 0.6;
    
    % Common calculations
    R = 1 - (exp(w_in * 3) / exp(w_max * 3));
    dw_up = R * t_pulse * lambda * sinh(eta * V);
    dw_up = dw_up * (1 + variability*randn);
    w_update = w_in + dw_up;

    dw_leak = (w_in - w_min) * (1 - exp(-t_pulse / tau_1));
    dw_leak = dw_leak * (1 + variability*randn);
    w_leak = w_in - dw_leak;

    % Select final weight
    w_out = is_update * w_update + (~is_update) * w_leak;
    w_out = max(min(w_out, w_max), w_min);

    % Compute read current from final weight
    I_out = (1 - w_out) * alpha * (1 - exp(-beta * V)) + ...
             w_out * gamma * sinh(delta * V);
end
