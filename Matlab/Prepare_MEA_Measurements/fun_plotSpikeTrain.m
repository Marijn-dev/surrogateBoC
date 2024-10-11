function [] = fun_plotSpikeTrain(sp_struct,ID,chanIDselect,xlimits,ylimits,t_stim,plotTot)
global pm

selectID = [];
for i = 1:length(chanIDselect)
    ID_now = chanIDselect(i);
    
    %Check if the channel is in the selected channels to be plotted
    j = find(strcmp(ID, ID_now{1}));    
    if ~isempty(j)
        selectID = [selectID j];
    end  
end

spk_times = [];
spk_chan = [];
for i=1:length(selectID)
    spk_times = [spk_times sp_struct{selectID(i),1}];
    spk_chan = [spk_chan i*ones(size(sp_struct{selectID(i),1}))];    
end

%% Figure
screenSize = get(0, 'ScreenSize');

if plotTot
    % figure('Position', [10, 20, 0.8*screenSize(3), 0.8*screenSize(4)]);
    fig = figure();
    defaultPos = get(fig, 'Position');  % [left, bottom, width, height]
    newWidth = 2 * defaultPos(3);       % new width is twice the original
    newPos = [defaultPos(1), defaultPos(2), newWidth, defaultPos(4)];  % [left, bottom, newWidth, height]
    set(fig, 'Position', newPos)

    scatter(spk_times, spk_chan, 6, 'filled');
    hold on
    if ~isempty(t_stim)
        xline(t_stim,'color', [0 .5 0],'LineWidth',1.5) 
    end
    xlabel('Time (s)');
    ylabel('Electrode number');
    % ylim([0.5, length(selectID)+0.5]); 
    % yticks(1:length(selectID));
    % yticklabels(chanIDselect);
    legend('Detected spike','Stimulation timing') %,'Location','bestoutside'
    if ~isempty(ylimits)
        ylim(ylimits)
    end
    if ~isempty(xlimits)
        xlim(xlimits)
    else
        % xlim([t(1) t(end)])
    end  
else
    if ~isempty(t_stim)
        N = min(6,length(t_stim));
        % figure('Position', [10, 50, 0.6*screenSize(3), 0.6*screenSize(4)]);
        figure();
        for i=1:N
            subplot(ceil(N/2),2,i);
            scatter(spk_times, spk_chan, 6, 'filled');
            hold on
            xline(t_stim,'color', [0 .5 0],'LineWidth',1.5) 
            xlim([t_stim(i)-1 t_stim(i)+3])
            xlabel('Time (s)');
            ylabel('Electrode IDs');
            ylim([0.5, length(selectID)+0.5]); 
            yticks(1:length(selectID));
            yticklabels(chanIDselect);
        end
        legend('Detected spike','Stimulation timing') %,'Location','bestoutside'
    end
end

