function [pulsed_train_img_h, sz_train_h] = img_2_pulse(train_img, a, b)
% IMG_2_PULSE Processes a 3D matrix by calling insertZeros and expandMatrix.
%   [pulsed_train_img_h, sz_train_h] = img_2_pulse(train_img, a, b)
%   takes the 3D matrix train_img, applies zero-padding with insertZeros using
%   the parameter 'a', and then expands the matrix using expandMatrix with 'b'.
%   The resulting 3D matrix is returned as pulsed_train_img_h along with its size.

    % Call insertZeros function
    zero_padded_train_img_h = insertZeros(train_img, a);

    % Call expandMatrix function
    pulsed_train_img_h = expandMatrix(zero_padded_train_img_h, b);

    % Get the size of the processed matrix
    sz_train_h = size(pulsed_train_img_h);

end
function C = insertZeros(A, b)
    [n, m, a] = size(A);
    new_m = m + m * b + b; 

    C = zeros(n, new_m, a, 'like', A);
    C(:, (b+1):(b+1):(b+1)*m, :) = A;
    
end

function output_matrix = expandMatrix(input_matrix, b)
    output_matrix = repelem(input_matrix, 1, b, 1);
end
