function I = fun_GetI(indx_end,start_indexes, stop_indexes)
    global pm

    % Check if the number of start and stop indexes match
    if length(start_indexes) ~= length(stop_indexes)
        error('Start and stop indexes must have the same length.');
    end
    
    % Make a copy of the input zero array to modify it
    I = zeros(indx_end,1);
    
    stim_idx_Offset = round(pm.stim.duration/pm.st) + 1;
    
    % Loop through each pair of start and stop indexes
    for i = 1:length(start_indexes)
        % Validate start and stop indexes
        if start_indexes(i) < 1 || stop_indexes(i) > length(I)
            error('Start or stop indexes are out of bounds.');
        end
        if start_indexes(i) > stop_indexes(i)
            error('Start index cannot be greater than stop index.');
        end
        
        % Fill the section with the value x
        I(start_indexes(i):stop_indexes(i)-stim_idx_Offset) = pm.stim.amplitude;
    end
end

