function [path_info_output,ind,max_N_value] = Det_peaks(samples,signal_space,X,Y)

path_info_output = zeros(signal_space,2);
ind = zeros(signal_space,2);
max_N_value = zeros(1,signal_space);

for m = 1:length(X)
    for n = 1:length(Y)
        step = [1 0;0 1;-1 0;0 -1];
        scope = [length(X),length(Y)];
        mark = 1;

        for k = 1:size(step,1)
            temp_x = m + step(k,1);
            if temp_x < 1 || temp_x > scope(1)
                temp_x = m;
            end
            temp_y = n + step(k,2);
            if temp_y < 1 ||temp_y > scope(2)
                temp_y = n;
            end
            if samples(m,n) < samples(temp_x,temp_y)
                mark = 0;
                break;
            end
        end
       
        if mark == 1
            min_index = minI(max_N_value);
            if max_N_value(min_index) < samples(m,n)  
                max_N_value(min_index) =  samples(m,n);
                path_info_output(min_index,:) = [X(m) Y(n)];
                ind(min_index,:) = [m,n]; 
            end
        end
    end
end
path_info_output(sum(path_info_output,2)==0,:) = [];
ind(sum(ind,2)==0,:) = [];

for jj = 1:size(ind,1)
    path_info_output(jj,3) = samples(ind(jj,1),ind(jj,2));
end

end

%% 求得输入数组中最小元素的下标
% input = max_N_value;
function index = minI(input)
    index  = 1;
    for k = 2:length(input)
        if input(k) < input(index)
            index = k;
        end
    end
end