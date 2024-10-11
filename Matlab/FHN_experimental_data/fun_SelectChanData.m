function [data_Chan] = fun_SelectChanData(data,chanID,chanIDSelect)

selectID = [];
for i = 1:length(chanIDSelect)
    ID_now = chanIDSelect(i);
    
    %Check if the channel is in the selected channels
    j = find(strcmp(chanID, ID_now{1}));    
    if ~isempty(j)
        selectID = [selectID j];
    end  
end

data_Chan = data(:,selectID);


%% old code
% %% Extract data
% for i = 1:size(data, 2)
%     ID = chanID(i);
% 
%     %Check if the channel is in the selected channels to be plotted
%     j = find(strcmp(chanIDSelect, ID{1}));    
%     if ~isempty(j)
%         data_Chan(:,j) = data(:, i);
%     end  
% 
% end
end

