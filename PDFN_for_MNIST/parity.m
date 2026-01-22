function [parity_matrix, sz_parity_matrix] = parity(input_matrix, mode)
    [dim1, dim2, dim3] = size(input_matrix);
    
    parity_matrix = input_matrix;
    switch mode 
        case 'parity'
             % Expand size for parity extension
            parity_matrix = zeros(dim1 + dim2 - 1, dim2, dim3, 'like', input_matrix);

            % Copy original
            parity_matrix(1:dim1, :, :) = input_matrix;

            % Apply parity by summing adjacent rows
            for i = 1:(dim2 - 1)
                parity_matrix(dim1 + i, :, :) = ...
                    input_matrix(i, :, :) + input_matrix(i + 1, :, :);
            end

        case 'only_parity'
            parity_matrix = zeros(dim2 - 1, dim2, dim3, 'like', input_matrix);
            for i = 1:(dim2 - 1)
                parity_matrix(i, :, :) = ...
                    input_matrix(i, :, :) + input_matrix(i + 1, :, :);
            end

        case 'no_parity'
            % Do nothing
            sz_parity_matrix = size(input_matrix);
            return;
        otherwise
            error('Invalid mode.');
    end

    % Vectorized parity logic: set any value == 2 to 0
    parity_matrix(parity_matrix == 2) = 0;

    % Return size
    sz_parity_matrix = size(parity_matrix);
end
