function [] = fun_plotAll(t,data,t_stim,chanID,th,plot_TH,xlimits,ylimits)
global pm
plot_settings
% Get the screen size
screenSize = get(0, 'ScreenSize');

% Set the figure size to be the maximum screen size
figure('Position', [10, 50, 1.2*screenSize(3), 10*screenSize(4)]);
tiledlayout(2,2, 'Padding', 'compact', 'TileSpacing', 'compact'); 

clr = lines(2);

% Loop through each column of the matrix and create subplots
for i = 1:size(data, 2)
    idCH = chanID(i);
    plot_index = func_label2index(idCH);
    
    subplot(pm.standardsettings.sizeOfMatrix(1),pm.standardsettings.sizeOfMatrix(1),plot_index);
    plot(t,data(:, i));
    hold on

    if plot_TH
        yline(th{i}, 'r')
        yline(-th{i}, 'r', 'HandleVisibility', 'off')
    end

    % if strcmp(pm.stim.channelID,idCH{1}) && ~isempty(I_stim)
    %     hold on
    %     plot(t,I_stim,'y')
    % end

    if ~isempty(t_stim)
        xline(t_stim,'color', clr(2,:)) 
    end

    if ~isempty(ylimits)
        ylim(ylimits)
    else
        ylim([-th{i}*2.5 th{i}*2.5])
    end
    if ~isempty(xlimits)
        xlim(xlimits)
    end 
end

end

