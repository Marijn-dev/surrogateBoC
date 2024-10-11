function [stim_delta] = fun_getStimDelta(t,t_stim)
global pm

len_t = length(t);
n_full_blocks = floor(len_t / 4);  % Number of full [1; 2; 3; 4] blocks
n_rem = rem(len_t, 4);             % Remaining elements

% Construct the full blocks
full_blocks = repmat((1:4)', n_full_blocks, 1);

% Construct the remaining elements
remaining_elements = (1:n_rem)';

% Combine the full blocks and remaining elements
delta = [full_blocks; remaining_elements];

idx_t_stim = find(t==t_stim);
stim_delta = delta(idx_t_stim);

end