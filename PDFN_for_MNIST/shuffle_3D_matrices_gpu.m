function [shuffled_matrix1, shuffled_matrix2] = shuffle_3D_matrices_gpu(matrix1, matrix2)
    % matrix1: H x W x N, matrix2: N x 1

    % Determine number of elements
    N = size(matrix1, 3);

    % Generate permutation on the correct device
    if canUseGPU
        indices = gpuArray.randperm(N);
    else
        indices = randperm(N);
    end

    % Shuffle using those indices
    shuffled_matrix1 = matrix1(:, :, indices);
    shuffled_matrix2 = matrix2(indices);
end
