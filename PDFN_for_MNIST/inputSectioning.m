function [output_matrix, final_size] = inputSectioning(input_matrix, s)
% INPUTSECTIONING Splits a 3D matrix and concatenates sections along the 1st dimension.
%   [output_matrix, final_size] = inputSectioning(input_matrix, s)
%   If the 2nd dimension of the input matrix is not divisible by 's',
%   the matrix is padded with zeros along the 2nd dimension.

    % Get the size of the input matrix
    [dim1, dim2, dim3] = size(input_matrix);

    % Calculate the size of each section along the 2nd dimension
    section_size = ceil(dim2 / s);

    % Calculate the new padded size for the 2nd dimension
    padded_dim2 = section_size * s;

    % Pad the input matrix with zeros along the 2nd dimension if necessary
    if dim2 < padded_dim2
        pad_amount = padded_dim2 - dim2;
        input_matrix = padarray(input_matrix, [0, pad_amount, 0], 0, 'post');
    end

    % Preallocate output: dim1 * s × section_size × dim3
    output_matrix = zeros(dim1 * s, section_size, dim3, 'like', input_matrix);


    % Fill each section into preallocated output
    for i = 1:s
        start_idx = (i - 1) * section_size + 1;
        end_idx   = i * section_size;

        section = input_matrix(:, start_idx:end_idx, :);  % slice section
        output_matrix((i-1)*dim1+1 : i*dim1, :, :) = section;  % stack along dim1
    end

    output_matrix = single(output_matrix);  % Ensure consistent datatype
    final_size = size(output_matrix);
end
