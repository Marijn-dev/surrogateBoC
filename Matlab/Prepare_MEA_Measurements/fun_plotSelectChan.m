function [] = fun_plotSelectChan(t,data,ID,chanIDselect,xlimits,ylimits,t_stim)
global pm

% selectID = [];
% for i = 1:length(ID)
%     ID_now = ID(i);
% 
%     %Check if the channel is in the selected channels to be plotted
%     j = find(strcmp(chanIDselect, ID_now{1}));    
%     if ~isempty(j)
%         selectID = [selectID i];
%     end  
% end

selectID = [];
for i = 1:length(chanIDselect)
    ID_now = chanIDselect(i);
    
    %Check if the channel is in the selected channels to be plotted
    j = find(strcmp(ID, ID_now{1}));    
    if ~isempty(j)
        selectID = [selectID j];
    end  
end

%% Figure with subplots

screenSize = get(0, 'ScreenSize');
fig = figure('Position', [10, 10, 0.6*screenSize(3), 0.9*screenSize(4)]);
tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact'); 

colour_plot = lines(length(selectID)+1);
colour_plot(end+1,:) = colour_plot(2,:);
colour_plot(2,:) = [];

for i=1:length(selectID)
    %Plot the data
    % subplot(length(selectID),1,i);
    subplot(ceil(length(selectID)/2),2,i);
    plot(t,data(:, selectID(i)));
    hold on
    if ~isempty(t_stim)
        xline(t_stim,'Color',colour_plot(end,:),'LineWidth',1.5) %,'HandleVisibility','off' 
    end
    xlabel('Time (s)')
    % ylabel(['Data ' chanIDselect{i}])
    ylabel(chanIDselect{i})
    grid on
    
    if ~isempty(ylimits)
        ylim(ylimits)
    end
    if ~isempty(xlimits)
        xlim(xlimits)
    else
        xlim([t(1) t(end)])
    end   
end

legend({'Data';['Stim ' pm.stim.channelID ]},'Location','best')

%% Figure with 1 plot
% figure('Position', [1, 1, screenSize(3), 0.6*screenSize(4)]);
figure('Position', [10, 50, 0.6*screenSize(3), 0.6*screenSize(4)]);
tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact'); 

for i=1:length(selectID)
    plot(t,data(:, selectID(i)),'Color',colour_plot(i,:),'DisplayName',chanIDselect{i});
    hold on
    xlabel('Time (s)')
    ylabel('Potential (V)')
    % ylabel('Firing rate (Hz)')
    grid on
end
if ~isempty(ylimits)
        ylim(ylimits)
end
if ~isempty(xlimits)
    xlim(xlimits)
else
    xlim([t(1) t(end)])
end   
if ~isempty(t_stim)
    xline(t_stim,'Color',colour_plot(end,:),'LineWidth',1.5)
end

legend_text = chanIDselect;
legend_text{length(chanIDselect)+1} = ['Stim ' pm.stim.channelID ];
legend(legend_text,'Location','bestoutside')


end
