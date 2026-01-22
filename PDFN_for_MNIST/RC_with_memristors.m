
function [reservoir_size, test_accuracy, avg_read_energy_per_image, avg_write_energy_per_image] = RC_with_memristors(dimension, parity_flag, section_count, quantization, tau_1, dataset_q, sq_section_size, stride, variability)

    tic %Start stopwatch timer
    
    %% Function Definations
    logistic = @(v) 1./(1+exp(-v));

    %% Add the path to the datasets here

    % MNIST
    train_img_path = "MNIST/train-images.idx3-ubyte";
    train_label_path = "MNIST/train-labels.idx1-ubyte";
    test_img_path = "MNIST/t10k-images.idx3-ubyte";
    test_label_path = "MNIST/t10k-labels.idx1-ubyte";

    %% Reading from MNIST Dataset
    no_train_imgs = getNumberOfImages(train_img_path);
    no_test_imgs = getNumberOfImages(test_img_path);
    
    train_offset = 0;
    test_offset = 0;
    
    [train_img, train_label] = readMNIST_28(train_img_path, train_label_path, no_train_imgs, train_offset);
    [test_img, test_label] = readMNIST_28(test_img_path, test_label_path, no_test_imgs, test_offset);
    % Move data to GPU
    if canUseGPU
        train_img = gpuArray(train_img);
        test_img = gpuArray(test_img);
    else
        train_img = train_img;  % No conversion needed
        test_img = test_img;
    end
    
    % Display basic information about the loaded data
    %disp(['Training images size: ', num2str(size(train_img))]);
    %disp(['Training labels size: ', num2str(size(train_label))]);
    %disp(['Testing images size: ', num2str(size(test_img))]);
    %disp(['Testing labels size: ', num2str(size(test_label))]);
    
    %% Shuffling the datasets 
    [train_img, train_label] = shuffle_3D_matrices_gpu(train_img, train_label);
    [test_img, test_label] = shuffle_3D_matrices_gpu(test_img, test_label);
    
    %% Binarizing the MNIST Dataset 
    n_train = numel(train_img);
    n_test = numel(test_img);
    train_img = single(train_img > 0.1); 
    test_img = single(test_img > 0.1);

    %display_image(train_img, train_label, 1, 'training');
    %display_image(train_img, train_label, 2, 'training');
    %display_image(train_img, train_label, 3, 'training');
    %display_image(test_img, test_label, 1, 'testing');

    %% Check Images 
    %display_image(train_img, train_label, 1, 'training');
    %display_image(train_img, train_label, 2, 'training');
    %display_image(train_img, train_label, 3, 'training');
    %display_image(test_img, test_label, 1, 'testing');
    
    %% Check Original Sizes 
    og_sz_train = size(train_img);
    og_sz_test = size(test_img);
    
    %% Preprocessing
    % 1D or 2D 
    train_img = process_img_dimension(train_img, dimension);
    test_img = process_img_dimension(test_img, dimension);
    %display_image(train_img, train_label, 1, 'training');
        
    % Parity
    [parity_train_img, parity_train_sz] = parity(train_img, parity_flag);
    [parity_test_img, parity_test_sz] = parity(test_img, parity_flag);
    %display_image(parity_train_img, train_label, 1, 'training');
    
    clear train_img
    clear test_img
    
    % Image to pulse train conversion
    a = 0; % Zero-padding parameter
    b = 1; % Expansion parameter
    [pulsed_train_img_no_sections, sz_train] = img_2_pulse(parity_train_img, a, b);
    [pulsed_test_img_no_sections, sz_test] = img_2_pulse(parity_test_img, a, b);
    %display_image(pulsed_train_img_no_sections, train_label, 1, 'training');
    
    clear parity_train_img
    clear parity_test_img
    
    % No sectioning Test
    %pulsed_train_img = pulsed_train_img_no_sections;
    %pulsed_test_img = pulsed_test_img_no_sections;

    % Sectioning
    s = section_count;
    [pulsed_train_img_sectioned, sz_train_sectioned] = inputSectioning(pulsed_train_img_no_sections, s);
    [pulsed_test_img_sectioned, sz_test_sectioned] = inputSectioning(pulsed_test_img_no_sections, s);
    %display_image(pulsed_train_img_sectioned, train_label, 1, 'training');
    
    clear pulsed_train_img_no_sections
    clear pulsed_test_img_no_sections

    pulsed_train_img = pulsed_train_img_sectioned;
    pulsed_test_img = pulsed_test_img_sectioned;

    clear pulsed_train_img_sectioned
    clear pulsed_test_img_no_sectioned

    sz_train = size(pulsed_train_img);
    sz_test = size(pulsed_test_img);
    
    toc
    
    %% Defining constant
    % Defining Memristor Input Constants
    V_write = 1.5;
    V_read = 0.6;
    t_write = 10^(-9); % Write pulse duration = 1 ns
    tread = 10^(-9); % Read pulse duration =  1 ns
    t_pulse = t_write;
    w_max = 1;
    w_min = 0.1;

    
    %% Reservoir layer 
    V_t = V_write * pulsed_train_img;  % Vector of voltages
    V_t = gather(V_t);
    [XTrain_temp, XTrain_temp_w] = run_reservoir_cpu(V_t, V_read, t_pulse, t_pulse, tau_1, variability);
    
    XTrain_temp = gather(XTrain_temp);
    XTrain_temp_w = gather(XTrain_temp_w);
    
    disp("Space")
    
    toc;
    %% Write Power and Write Energy - Per Memristor and Average
    WritePowerPerMemristorPerImage = sum(XTrain_temp_w, 2) * V_write;
    WriteEnergyPerMemristorPerImage = WritePowerPerMemristorPerImage * t_write;
    
    %disp('Write Average Power and Energy')
    AverageWritePowerPerMemristorFullSet = mean(WritePowerPerMemristorPerImage, 3);
    AverageWriteEnergyPerMemristorFullSet = mean(WriteEnergyPerMemristorPerImage, 3);
    %disp(AverageWritePowerPerMemristorFullSet(:));
    %disp(AverageWriteEnergyPerMemristorFullSet(:));
    
    %disp('Total Write Power');
    TotalWritePower = sum(WritePowerPerMemristorPerImage, 'all');
    TotalWriteEnergy = sum(WriteEnergyPerMemristorPerImage, 'all');
    %disp(['Total Write Power = ', num2str(TotalWritePower)]);
    %disp(['Total Write Energy = ', num2str(TotalWriteEnergy)])
        
    %Read Power and Read Energy 
    %disp('Read Power and Energy Per Memristor')
    ReadPowerPerMemristorPerImage = XTrain_temp * V_write;
    ReadEnergyPerMemristorPerImage = ReadPowerPerMemristorPerImage * tread;
    %disp(ReadPowerPerMemristorPerImage(:, 1));
    %disp(ReadEnergyPerMemristorPerImage(:, 1));
    
    %disp('Read Average Power and Energy')
    AverageReadPowerPerMemristorFullSet = mean(ReadPowerPerMemristorPerImage, 3);
    AverageReadEnergyPerMemristorFullSet = mean(ReadEnergyPerMemristorPerImage, 3);
    %disp(AverageReadPowerPerMemristorFullSet(:));
    %disp(AverageReadEnergyPerMemristorFullSet(:));
    
    %% Quantization 
    
    % === Parameters ===
    L = quantization;          % number of levels or bins
    %mode = 'levels';           % 'levels' = L equal levels, 'bins' = L equal bins
    
    % === Range ===
    I_read_max = (1 - w_max)*alpha*(1 - exp(-beta*V_read)) + w_max*gamma*sinh(delta*V_read);     
    I_read_min = (1 - w_min)*alpha*(1 - exp(-beta*V_read)) + w_min*gamma*sinh(delta*V_read);
    
    % === Quantization Setup ===
    delta_q = (I_read_max - I_read_min) / L;
    levels  = I_read_min + (0.5:1:L-0.5) * delta_q;  % bin centers

    % Clip and map
    I_clipped = min(max(XTrain_temp, I_read_min), I_read_max);
    index = floor((I_clipped - I_read_min) / delta_q);
    index = min(max(index,0), L-1);
    I_quantized = levels(index+1);    % +1 for MATLAB indexing

    % === Debugging ===
    unique_vals = unique(I_quantized(:));
    num_unique  = numel(unique_vals);
    %disp(['Number of unique quantized values: ' num2str(num_unique)]);
    %disp('Quantization levels used (in µA):');
    %disp(unique_vals' * 1e6);
    
    % Replace XTrain_temp with quantized version
    XTrain_temp = I_quantized;
    
    %% Rescaling current values - Training
    
    %XTrain_rescaled = rescale(XTrain_temp, 0, 1);
    colmin = I_read_min;

    %% Logistic Regression Readout Training
    reservoir_size = sz_train(1);
    XTrain = XTrain_rescaled(1:sz_train(1), 1:sz_train(3)).';
    %XTrain = XTrain_temp(1:sz_train(1), 1:sz_train(3)).';
    nOut = 10;
    theta = randn(sz_train(1), nOut);
    %writematrix(theta, 'theta_before_training.txt')
    br = 2e-2;
    epochErr = [];
    nEpoch = 500;
    disp_train_acc = zeros(nEpoch, 1);
    trainErr = 0;
    
    for e = 1:nEpoch
        trainErr = 0;
        for f = 1:sz_train(3)
            R1 = XTrain(f, :)';
            %vo = logistic(theta' * [R1]);
            z = theta' * R1;
            vo = logistic(z);
            [voMax,voMaxInd] = max(vo);
            trainErr = trainErr + (train_label(f)+1 ~= voMaxInd);
            vRef = zeros(nOut,1);
            vRef(train_label(f)+1)=1;
            err = vo-vRef;
            dtheta = br*[R1]*err';
            theta = theta - dtheta;
        end
        train_accuracy = 1-trainErr/sz_train(3);
        %disp(['Epoch Number: ', num2str(e),  '|| Accuracy: ', num2str(100 * train_accuracy), '%'])
        epochErr = [epochErr, trainErr];
        disp_train_acc(e) = train_accuracy;
    end
    
    disp("Training Done")
    
    %% Prediction using testing set
    
    % --- GENERATE voltage matrix for test images ---
    V_t_test = V_write * pulsed_test_img;     % same logic as training
    %pulsed_test_img = max(0, min(1, pulsed_test_img));
    %V_t_test = Vmin + (Vmax - Vmin) * pulsed_test_img; 
    %V_t_test(pulsed_test_img == 0) = 0;
    V_t_test = gather(V_t_test);
    [XTest_temp, XTest_temp_w] = run_reservoir_cpu(V_t_test, V_read, t_pulse, t_pulse, tau_1, variability);
    XTest_temp = gather(XTest_temp);
    XTest_temp_w = gather(XTest_temp_w);

    % === Quantization Setup ===

    % L equal-width bins
    delta_q = (I_read_max - I_read_min) / L;
    levels  = I_read_min + (0.5:1:L-0.5) * delta_q;  % bin centers

    % Clip and map
    I_clipped = min(max(XTest_temp, I_read_min), I_read_max);
    index = floor((I_clipped - I_read_min) / delta_q);
    index = min(max(index,0), L-1);
    I_quantized = levels(index+1);    % +1 for MATLAB indexing
    % === Debugging ===
    unique_vals = unique(I_quantized(:));
    num_unique  = numel(unique_vals);
    
    % Replace XTrain_temp with quantized version
    XTest_temp = I_quantized;
    
    XTest_rescaled = rescale(XTest_temp, "InputMin", colmin, "InputMax", colmax);
    
    % --- Inference using softmax classifier ---
    vo = softmax(theta' * XTest_rescaled);  % size: [nOut x num_test_samples]
    [voMax, voMaxInd] = max(vo, [], 1);
    errCount = sum((test_label(:) + 1) ~= voMaxInd(:));
    test_accuracy = 100 * (1 - errCount / size(XTest_rescaled, 2));
    
    disp(['Test Accuracy: ', num2str(test_accuracy), '%']);
    
    %I_read_max = max(I_read_test(:));
    %I_write_max = max(I_write_test(:));
    
    %NoOfWritePulses = sum(pulsed_test_img == 1, 2);
    %disp(NoOfWritePulses(:, 1));
    %AvgNoOfWritePulses = mean(NoOfWritePulses, 1);
    %disp(AvgNoOfWritePulses(1));
    
    
    %disp("Space")
    %disp(['Testing Accuracy:', num2str(100*(1-(errCount/sz_test(3)))), '%'])
    
    toc; 
    
    %Write Power and Write Energy - Per Memristor and Average
    %disp('Write Power and Energy Per Memristor')
    
    WritePowerPerImageTest = squeeze(sum(sum(XTest_temp_w, 1), 2)) * V_write;
    WriteEnergyPerImageTest = WritePowerPerImageTest * t_write;
    %disp(WritePowerPerMemristorPerImageTest(:, 1));
    %disp(['Write Energy Per Image (1) = ', num2str(WriteEnergyPerImageTest(1, 1))]);
    
    %disp('Write Average Power and Energy')
    %AverageWritePowerPerMemristorFullSetTest = mean(WritePowerPerMemristorPerImageTest, 'all');
    AverageWriteEnergyPerImage = mean(WriteEnergyPerImageTest, 'all');
    %disp(AverageWritePowerPerMemristorFullSetTest(:));
    %disp(['Average Write Energy Per Image = ', num2str(AverageWriteEnergyPerImage(:))]);
    avg_write_energy_per_image = AverageWriteEnergyPerImage;
    
    %disp('Total Write Power Testing');
    %TotalWritePowerTest = sum(WritePowerPerMemristorPerImageTest, 'all');
    TotalWriteEnergyTest = sum(WriteEnergyPerImageTest, 'all');
    %disp(['Total Write Power Testing = ', num2str(TotalWritePowerTest)]);
    %disp(['Total Write Energy Testing = ', num2str(TotalWriteEnergyTest)])
    
    %Read Power and Read Energy 
    %disp('Read Power and Energy Per Memristor')
    ReadPowerPerImageTest = squeeze(sum(XTest_temp, 1)) * V_write;
    ReadEnergyPerImageTest = ReadPowerPerImageTest * tread;
    %disp(ReadPowerPerMemristorPerImageTest(:, 1));
    %disp(['Read Energy Per Image (1) = ', num2str(ReadEnergyPerImageTest(1, 1))]);
    
    %disp('Read Average Power and Energy')
    %AverageReadPowerPerMemristorFullSetTest = mean(ReadPowerPerMemristorPerImageTest, 3);
    
    AverageReadEnergyTest = mean(ReadEnergyPerImageTest, "all");
    %disp(AverageReadPowerPerMemristorFullSetTest(:));
    %disp(['Average Read Energy Per Image = ', num2str(AverageReadEnergyTest)]);
    avg_read_energy_per_image = AverageReadEnergyTest;
    
    %disp('Total Read Power');
    %TotalReadPowerTest = sum(ReadPowerPerMemristorPerImageTest, 'all');
    TotalReadEnergyTest = sum(ReadEnergyPerImageTest, 'all');
    %disp(['Total Read Power = ', num2str(TotalReadPowerTest)]);
    %disp(['Total Read Energy = ', num2str(TotalReadEnergyTest)])
    
    %Total Power 
    %TotalTrainPowerTest = TotalReadPowerTest + TotalWritePowerTest;
    %TotalTrainEnergyTest = TotatReadEnergyTest + TotalWriteEnergyTest;
    %disp(['Total Testing Power = ', num2str(TotalTrainPowerTest)]);
    %disp(['Total Testing Energy = ', num2str(TotalTrainEnergyTest)]);
    
    %Total Energy
    TotalAvgEnergy = AverageReadEnergyTest + AverageWriteEnergyPerImage;
    %disp(['Average Total Energy=', num2str(TotalAvgEnergy)])
    TotalEnergy = TotalReadEnergyTest + TotalWriteEnergyTest;
    %disp(['Total Energy Test=', num2str(TotalEnergy)])
    
end

