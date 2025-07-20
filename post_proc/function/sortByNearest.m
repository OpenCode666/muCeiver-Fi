function [matrix_sorted,coef] = sortByNearest(A, B)
    N = size(B, 2);                              % sample number
    matrix_sorted = zeros(size(B));              % sorted B
    
    used = false(1, N);                          % used points in B
    for i = 1:N 
        distances = vecnorm(B - A(:, i), 2, 1);  % dist between unused points in B and A
        distances(used) = inf;                   
        [~, minIdx] = min(distances);            
        matrix_sorted(:, i) = B(:, minIdx); 
        used(minIdx) = true;
    end
    coef = norm(A - matrix_sorted, 'fro'); 
end