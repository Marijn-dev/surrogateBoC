function [r,t_eval] = fun_calcFiringRate(chanIDselect,chanID,sp_struct,t_eval,temp_res,num_closest,window,I_stim)
global pm

%% Extract channel ID to calculate
selectID = [];
for i = 1:length(chanIDselect)
    ID_now = chanIDselect(i);
    
    %Check if the channel is in the selected channels to be plotted
    j = find(strcmp(chanID, ID_now{1}));    
    if ~isempty(j)
        selectID = [selectID j];
    end  
end

%% Calculate actual firing rate 
% determine time of evaluation
% dt_eval = ((t(end)-t(1))/nt);
% t_eval = t(1):dt_eval:t(end);

% r = zeros(length(t_eval),size(y,2));
r = zeros(length(t_eval),length(selectID));

count = 1;
for j = selectID %1:size(y,2)
    for k = 1:length(sp_struct{j,1})
        
        if num_closest~=-1
            % Find the index of the closest time point to the current spike time
            [~, idx] = min(abs(t_eval - sp_struct{j,1}(k)));
            
            % Determine the range of indices around the spike time to consider
            start_idx = max(1, idx - floor(num_closest/2));
            end_idx = min(length(t_eval), idx + floor(num_closest/2));
        else
            start_idx = 1;
            end_idx = length(t_eval);

        time_diff = t_eval(start_idx:end_idx) - sp_struct{j,1}(k);
        
        if strcmp(window,'Gaussian')
            w = (1/(sqrt(2 * pi) * temp_res)) * exp(-time_diff.^2 / (2 * temp_res^2));
        elseif strcmp(window,'Causal')
            alpha = 1/temp_res;
            F = alpha^2*time_diff.*exp(-alpha*time_diff);
            w = max(zeros(size(F)),F);
        end
        % Add these contributions to the firing rate
        r(start_idx:end_idx,count) = r(start_idx:end_idx,count) + w';
        end
    end
    count = count+1;
end

disp('Firing rates calculated')

% %% plot
% if length(chanIDselect)<6 %plot it if there are not too many firing rates to include
%     % Plot the firing rates
%     legend_text = chanIDselect;
%     legend_text{length(chanIDselect)+1} = ['Stim ' pm.stim.channelID ];
%     colour_plot = lines(length(chanIDselect)+1);
%     colour_plot(end+1,:) = colour_plot(2,:);
%     colour_plot(2,:) = [];
% 
%     figure()
%     yyaxis left
%     for i=1:length(chanIDselect)
%         plot(t_eval,r(:,i),'-','Color',colour_plot(i,:))
%         hold on
%     end
%     ylabel('Firing rate (Hz)')
%     yyaxis right
%     plot(t,I_stim,'Color',colour_plot(end,:))
%     xlabel('Time (s)')
%     ylabel('Current stimulation ($\mu A$)')
%     % xlim([0 10])
%     legend(legend_text )
% end

end