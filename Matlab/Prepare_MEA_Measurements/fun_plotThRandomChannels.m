function [] = fun_plotThRandomChannels(Data, t, th, ID, x)
    % Get the screen size and create a figure
    screenSize = get(0, 'ScreenSize');
    fig = figure('Position', [1, 1, 0.9*screenSize(3), 0.9*screenSize(4)]);
    
    % Randomly select x channels out of the total number of channels
    totalChannels = size(Data, 2);
    selectedChannels = randperm(totalChannels, x); % Randomly select x indices
    
    % Create a tiled layout for subplots
    tiledlayout(ceil(x/2), 2, 'Padding', 'none', 'TileSpacing', 'compact'); % 2 columns, adjust rows dynamically
    
    % Loop through the selected channels and plot
    for i = 1:x
        channelIdx = selectedChannels(i);
        nexttile % Automatically place plots in the tiled layout
        plot(t, Data(:, channelIdx));
        hold on
        yline(th{channelIdx}, 'k')
        yline(-th{channelIdx}, 'k', 'HandleVisibility', 'off')
        xlabel('time (s)')
        ylabel('Voltage ($\mu V$)')
        grid on
        title(['Thresholding: Ch. ' ID{channelIdx}], 'FontSize', 14);
        xlim([t(1) t(end)])
        ylim([-th{channelIdx}*3 th{channelIdx}*3])
    end
    
    % Add legend to the first subplot
    legend('Data', 'Threshold', 'Location', 'Best')
end