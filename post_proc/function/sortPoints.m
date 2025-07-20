function [out_sort,dis] = sortPoints(out_point)

currentPath = fileparts(mfilename('fullpath'));
ref_path = [currentPath '\\cod_res_3d.mat'];
load(ref_path)
dict = {'open';'sit';'right';'stand';'up';'walk'};
coef_pose = +inf;
out_sort = out_point;
for i = 1:6
    eval(['cod_res_3d = cod_res_3d_' dict{i,1} ';'])
    [out,coef] = sortByNearest(cod_res_3d, out_point);
    if coef <= coef_pose
        coef_pose = coef;
        out_sort = out;
        dis = min(mean((sum((out_sort-cod_res_3d).^2,1)).^0.5)/0.7*180,10);
    end
end
end