function [XTrain_temp, XTrain_temp_w] = run_reservoir_cpu(V_t, V_read, t_read, t_write, tau, variability)
    

    sz_in = size(V_t);
    num_images = sz_in(3);
    n_rows = sz_in(1);
    n_cols = sz_in(2);


    XTrain_temp = zeros(n_rows, num_images);
    XTrain_temp_w = zeros(n_rows, n_cols, num_images);

    tau_d2d = tau .* (1 + variability .* randn(n_rows, 1));

    % Parallel loop over images
    parfor f = 1:num_images
        %w = repmat([0.1; 0; 1], 1, n_rows);  % 3 x n_rows
        %w = zeros(n_rows, 1) + 0.1;
        w = 0.1 + variability * rand(n_rows, 1);
        temp_I_write = zeros(n_rows, n_cols);  % Temporary 2D slice for this image
        
        for t = 1:n_cols
            V_pulse = V_t(:, t, f);
            %w_new = zeros(3, n_rows);
            w_new = zeros(n_rows, 1);
            I_write = zeros(n_rows, 1);

            for r = 1:n_rows
                %[w_new(:, r), I_write(r)] = diffusive_memristor(w(:, r),V_pulse(r), t_write); % Diffusive Memristor
                [w_new(r), I_write(r)] = dynamic_memristor(w(r),V_pulse(r), t_write, tau_d2d(r), variability); % Dynamic Memristor with variability
                %[w_new(r), I_write(r)] = dynamic_memristor_concurrent(w(r), V_pulse(r), t_write, tau); % Dynamic Memristor (No variability)
            end
            w = w_new; 
            temp_I_write(:, t) = I_write;

        end

        % Compute read and write currents
        I_read = zeros(n_rows, 1);
        for r = 1:n_rows
            %[~, I_read(r)] = diffusive_memristor(w(:, r), V_read, t_read);
            [~, I_read(r)] = dynamic_memristor(w(r), V_read, t_read, tau, variability);
        end

        XTrain_temp(:, f) = I_read;
        XTrain_temp_w(:, :, f) = temp_I_write;

    end
end
