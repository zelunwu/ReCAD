function [R2, RMSE] = calculateR2RMSE(x, y)
    % Input validation
    if ~isvector(x) || ~isvector(y)
        error('Inputs must be vectors');
    end
    if length(x) ~= length(y)
        error('Inputs must have same length');
    end
    
    % Remove NaN pairs
    valid_idx = ~isnan(x) & ~isnan(y);
    x = x(valid_idx);
    y = y(valid_idx);
    
    if isempty(x) || isempty(y)
        error('No valid paired data points after removing NaNs');
    end
    
    % Calculate R-squared
    correlation = corrcoef(x, y);
    R2 = correlation(1,2)^2;
    
    % Calculate RMSE
    RMSE = sqrt(mean((x - y).^2));
    
    % Optional: add number of valid points used
    n_valid = sum(valid_idx);
    fprintf('Number of valid data pairs used: %d\n', n_valid);
end