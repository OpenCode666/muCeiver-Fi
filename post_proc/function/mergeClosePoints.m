function mergedData = mergeClosePoints(data, threshold,num)
    N = size(data, 1);        
    used = false(N, 1);      
    mergedData = [];   

    for i = 1:N
        if used(i)         
            continue;
        end
        distances = vecnorm(data(:,1:3) - data(i, 1:3), 2, 2);  
        closeIdx = find(distances < threshold);      
        avgPoint = mean(data(closeIdx, :), 1);  
        used(closeIdx) = true;   
        mergedData = [mergedData; avgPoint];
    end
    non_ind = [];
    for i = 1:size(mergedData,1)
        if mergedData(i,1) < 0.05 || mergedData(i,1) > 1 || mergedData(i,3) < 0.1
            non_ind(i,:) = 1;
        end
    end
    mergedData(non_ind==1,:) = [];
    if size(mergedData,1)>num
        mergedData = mergedData(1:num,:);
    elseif size(mergedData,1)<num
        padding = num-size(mergedData,1);
        mergedData = [mergedData;zeros(padding,3)];
    end
end
