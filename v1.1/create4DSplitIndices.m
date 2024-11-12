function [idx_train, idx_validation] = create4DSplitIndices(data, train_percentage)
    % Input validation
    if ~isnumeric(train_percentage) || train_percentage <= 0 || train_percentage > 100
        error('Percentage must be a number between 0 and 100');
    end
    
    % Initialize output matrices with zeros
    idx_train = zeros(size(data));
    idx_validation = zeros(size(data));
    
    % Find valid (non-NaN) elements
    valid_mask = ~isnan(data);
    valid_indices = find(valid_mask);
    total_valid_elements = length(valid_indices);
    
    if total_valid_elements == 0
        error('Input data contains no valid (non-NaN) elements');
    end
    
    % Calculate number of elements for training
    num_train = round(total_valid_elements * (train_percentage/100));
    
    % Randomly shuffle valid indices
    shuffled_indices = valid_indices(randperm(total_valid_elements));
    
    % Separate indices for training and validation
    train_indices = shuffled_indices(1:num_train);
    validation_indices = shuffled_indices(num_train+1:end);
    
    % Create binary masks
    idx_train(train_indices) = 1;
    idx_validation(validation_indices) = 1;
    
    % Verify the split
    assert(sum(idx_train(:) & idx_validation(:)) == 0, 'Overlap detected in train/validation split');
    assert(sum(idx_train(:)) + sum(idx_validation(:)) == sum(valid_mask(:)), 'Total elements mismatch');

    idx_train(idx_train<0.5) = nan;
    idx_validation(idx_validation<0.5) = nan;
end
