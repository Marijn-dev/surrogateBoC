function [data_t,data_fr,data_I] = fun_getTraj(t,r,I,seq_t,onlyStim)
global pm

N_t_seq = seq_t/(1/pm.filtersettings.sampling_frequency);
N = floor(t(end)/seq_t);

count = 1;
for i=1:N 
    %include only sequences with stimulation(s)
    I_seq = I((i-1)*N_t_seq+1:i*N_t_seq,:);
    t_seq = t(1:N_t_seq);

    if onlyStim
        if sum(I_seq,'all')~=0
            data_fr{count} = r((i-1)*N_t_seq+1:i*N_t_seq,:);
            data_t{count} = t_seq; %t((i-1)*N_t_seq+1:i*N_t_seq);
            data_I{count} = I_seq;

            % data_fr(count, :, :) = r((i-1)*N_t_seq+1:i*N_t_seq,:);
            % data_t(count, :) = t_seq; %t((i-1)*N_t_seq+1:i*N_t_seq);
            % data_I(count, :, :) = I_seq;
            count = count+1;
        end
    end
end
end