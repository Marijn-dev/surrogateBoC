function [ToT,ToT2] = fun_getSpikes_toolbox(t,y,th)
global pm

%% Positive Peaks
pospk  = cell(size(y,2),2);
for i=1:size(y,2)
    [pk,tk] = findpeaks(y(:,i),t,'MinPeakHeight',th{i},'MinPeakDistance',pm.spike.minpeakdistance);
    pospk{i,1} = pk;
    pospk{i,2} = tk;
end

pospeaksT=[];  %amount of positive spikes (Count)
for i=1:size(y,2)
    pospeaksS=length(pospk{i,2});
    pospeaksT=[pospeaksT; pospeaksS];
end

clear pospeaksS

%% Negative Peaks
negpk  = cell(size(y,2),2);
for i=1:size(y,2)
    [pk,tk] = findpeaks(-y(:,i),t,'MinPeakHeight',th{i},'MinPeakDistance',pm.spike.minpeakdistance);
    negpk{i,1} = -pk;
    negpk{i,2} = tk;
end

negpeaksT=[];  %amount of positive spikes (Count)
for i=1:size(y,2)
    negpeaksS=length(negpk{i,2});
    negpeaksT=[negpeaksT; negpeaksS];
end

clear negpeaksS

%% Combine the pos and neg peak vlaues into a matrix
valo=cellfun('length',pospk(:,1));
[maxiu1(1),Imax1(1)]=max(valo);   %find biggest number to make them equal length to
valo1=cellfun('length',negpk(:,1));
[maxiu1(2),Imax1(2)]=max(valo1);

clear Imax1 valo1 valo

testcell11=[];
testcell12=[];
for i=1:size(y,2)
    Padlll=[pospk{i,1}; zeros((maxiu1(1))-length(pospk{i,1}),1)]';  %make them equal length
    testcell11(:,i)=Padlll;
    paddllll=[negpk{i,1}; zeros((maxiu1(2))-length(negpk{i,1}),1)]';
    testcell12(:,i)=paddllll;
end
testcell11=testcell11';
testcell12=testcell12';
testcell11(testcell11==0)=nan; %change zeros to nan values
testcell12(testcell12==0)=nan;

ToT2 = [testcell12,testcell11];  %contains all the y values of all the hits in microvolt (pos and neg)

testcell13=[];
testcell14=[];
for i=1:size(y,2)
    Padlll=[pospk{i,2}, zeros(1,(maxiu1(1))-length(pospk{i,2}))];  %make them equal length
    testcell13(:,i)=Padlll;
    paddllll=[negpk{i,2}, zeros(1,(maxiu1(2))-length(negpk{i,2}))];
    testcell14(:,i)=paddllll;
end
testcell13=testcell13';
testcell14=testcell14';
testcell13(testcell13==0)=nan; %change zeros to nan values
testcell14(testcell14==0)=nan;


ToT = [testcell14,testcell13]; %Contains all the time values of all the hits in seconds(Pos and Neg)

end