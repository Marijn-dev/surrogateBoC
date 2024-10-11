function plot_index = func_label2index(Channed_ID)
global pm

% Extract letter and number from each cell   
% Regular expression to match letter and number
match = regexp(Channed_ID{1}, '([A-Z]+)(\d+)', 'tokens', 'once');

if ~isempty(match)
    % Extracted letter and number
    letter = match{1};
    number = str2double(match{2});

    % Display the results
%     fprintf('Cell: %s, Letter: %s, Number: %d\n', Channed_ID{1}, letter, number);
else
%     fprintf('Invalid format for cell: %s\n', Channed_ID{1});
end

[column_i,~] = find(pm.standardsettings.column_strings==letter);
row_i = number;
plot_index = (row_i-1)*pm.standardsettings.sizeOfMatrix(1)+column_i;
end

