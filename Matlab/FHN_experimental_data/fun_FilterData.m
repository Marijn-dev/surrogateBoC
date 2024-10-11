function [DataF] = fun_FilterData(DataR,disp_filter)
global pm

Wn = pm.filtersettings.cut_off_frequency/(pm.filtersettings.sampling_frequency/2);
[b,a] = butter(pm.filtersettings.filter_order,Wn,'high');

% Display frequency response of the filter
if disp_filter
    figure()
    freqz(b, a, 512, pm.filtersettings.sampling_frequency);
    title('Frequency Response of the High-pass Butterworth Filter','Interpreter','LaTeX','FontSize',14);
    xlabel('Frequency (Hertz)','Interpreter','LaTeX','FontSize',12);
    ylabel('Magnitude (dB)','Interpreter','LaTeX','FontSize',12);
    % exportgraphics(gcf,'Figures/filter.emf','ContentType','vector','BackgroundColor','none')
end

DataF = zeros(size(DataR));

for i = 1:size(DataR,2)
    DataF(:,i) = filtfilt(b,a,double(DataR(:,i)));
end

% DataF(DataF>1000)=nan;
disp('Raw data successfully filtered using butterworth')

end

