
clear; 
close all; 
clc

%dimensions = {'1D', '2D'};
dimensions = {'2D'};
%parities = {'no_parity', 'parity'};
parities = {'parity'};
sections = [1, 2, 4, 6, 7, 8];
%sections = [6];
quantization = [2, 4, 8, 16, 32, 64, 128];
%quantization = [256];
%tau_1 = [1e-9, 2e-9, 3e-9, 5e-9, 6e-9, 10e-9, 15e-9, 20e-9];
tau_1 = [6e-9, 10e-9, 15e-9, 20e-9];
%tau_1 = [6e-9];
%dataset_q = [2, 4, 8, 16, 32, 64];
dataset_q = [2];
%sq_section_size = [3, 4, 5];
sq_section_size = [0];
%sq_stride = [2, 3, 4, 5];
sq_stride = [0];
variability = 0.05;

% Preallocate result table
results = [];

% Loop over all combinations
    for dq = 1: length(dataset_q)
        for q = 1:length(quantization)
            for tau = 1:length(tau_1)
                for d = 1:length(dimensions)
                    for p = 1:length(parities)
                        for sq = 1:length(sq_section_size)
                            for stride = 1:length(sq_stride)
                                if stride > sq
                                    continue;
                                end
                                for s = 1:length(sections)
                                    disp('--------------------------------------');
                                    disp(['Running: ', dimensions{d}, ' | ', parities{p}, ' | Sections: ', num2str(sections(s)) ...
                                        ' | Quantization: ', num2str(quantization(q)) ' | Tau: ',num2str(tau_1(tau)), ...
                                        ' | Dataset Quantization: ', num2str(dataset_q(dq)), ' | Square Section Size: ', num2str(sq_section_size(sq)), ' | Stride: ', num2str(sq_stride(stride))]);
                        
                                    % Run the main RC pipeline with the selected configuration
                                    [reservoir_size, accuracy, E_read, E_write] = RC_with_memristors(dimensions{d}, parities{p}, sections(s), quantization(q), tau_1(tau), dataset_q(dq), sq_section_size(sq), sq_stride(stride), variability);
                                    
                                    % Append to results
                                    results = [results; ...
                                       {dataset_q(dq), dimensions{d}, parities{p}, sections(s), sq_section_size(sq), sq_stride(stride), quantization(q), tau_1(tau), reservoir_size, accuracy, E_read, E_write}];
                              
                                
                                end
                            end
                        end
                    end
                end
            end
        end
    end
% Convert to table 
resultsTable = cell2table(results, ... 
    'VariableNames', {'Dataset Quantization', 'Dimension', 'Parity', 'Sections', 'Square Section Size', 'Stride', 'Quantization', 'Tau', 'Reservoir Size', 'Accuracy (%)', 'Read Energy (J/Image)', 'Write Energy (J/Image)'});
% Save to file
writetable(resultsTable, 'results_all_configs.csv');
disp('All configurations complete. Results saved to results_all_configs.csv');

disp(resultsTable);

