function [StimID, amp, dur] = fun_getStimProp(MEA, filename)
    % Data from the CSV file
    if MEA=='A'
        prop_matrix = {
            'rec1.h5',   NaN, NaN, NaN, NaN;
            'rec2.h5',   'reservoir A', 'volt', '1000', 200;
            'rec3.h5',   'reservoir B', 'volt', '1000', 200;
            'rec4.h5',   'Tunnels', 'volt', '1000', 200;
            'rec5.h5',   'J1', 'curr', '20', 200;
            'rec6.h5',   'K2', 'curr', '20', 200;
            'rec7.h5',   'M6', 'curr', '20', 200;
            'rec8.h5',   'J5', 'volt', '1000', 200;
            'rec9.h5',   'J5', 'curr', '16', 200;
            'rec10.h5',  'J6', 'curr', '10', 200;
            'rec11.h5',  'J5', 'curr', '5', 200;
            'rec12.h5',  'J5', 'curr', '2', 200;
            'rec13.h5',  'J5', 'curr', '5', 200;
        };
    elseif MEA=='C'   
        prop_matrix = {
        'rec1.h5',   NaN, NaN, NaN, NaN;
        'rec2.h5',   'reservoir A', 'volt', '1000', 200;
        'rec3.h5',   'reservoir B', 'volt', '1000', 200;
        'rec4.h5',   'Tunnels', 'volt', '1000', 200;
        'rec5.h5',   'J1', 'cur', '5', 200;
        'rec6.h5',   'G6', 'cur', '5', 200;
        'rec7.h5',   'G6', 'cur', '5', 300;
        'rec8.h5',   'G6', 'cur', '5', 200;
        'rec9.h5',   'J1', 'curr', '5', 200;
        'rec10.h5',  'J9', 'cur', '5', 200;
        'rec11.h5',  'K11', 'cur', '5', 200;
        'rec12.h5',  'G6', 'cur', '5', 200;
        'rec13.h5',  'J1', 'cur', '5', 200;
        };
    end

    % Extract row based on filename
    idx = find(strcmp(prop_matrix(:, 1), filename), 1);
    
    if isempty(idx)
        error('Filename not found');
    end
    
    % Extracting the properties
    StimID = prop_matrix{idx, 2};
    amp = str2double(prop_matrix{idx, 4});
    
    if prop_matrix{idx, 3}=='volt'
        amp = amp*10^3; %[muV]
    end

    dur = prop_matrix{idx, 5}*10^-6; %[s]
end
