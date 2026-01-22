function output_matrix = process_img_dimension(img, mode)
% PROCESS_TRAIN_IMG Processes a 3D matrix based on the selected mode.
%   output_matrix = process_train_img(train_img, mode)
%   If mode is '1D', the matrix remains unchanged.
%   If mode is '2D', the 3D matrix is modified so that the 1st dimension 
%   is doubled by appending the transpose of each image along the 1st dimension.

    % Get the size of the input matrix
    [dim1, dim2, dim3] = size(img);

    switch mode
        case '1D'
            % In 1D mode, return the matrix as is
            output_matrix = img;
        case '2D'
            % In 2D mode, create a modified matrix
            output_matrix = zeros(dim1 * 2, dim2, dim3, 'like', img);
            for i = 1:dim3
                output_matrix(1:dim1, :, i) = img(:, :, i);
                output_matrix(dim1+1:end, :, i) = img(:, :, i).';
            end
        otherwise
            error('Invalid mode. Choose either "1D" or "2D".');
    end
end
