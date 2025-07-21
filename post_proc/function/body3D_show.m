function body3D_show(point,flag,lgd)
%% line
connections = [1,2; 
    2,3;
    3,4;
    4,5;
    2,6;
    6,7;
    7,8;
    2,9;
    9,10;
    10,11;
    11,12;
    9,13;
    13,14;
    14,15;
    ];
%% color
color_lib = [1, 0, 0;
    0.6350, 0.0780, 0.1840;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.6350, 0.0780, 0.1840;
    0.8500, 0.3250, 0.0980;
    0.9290, 0.6940, 0.1250;
    0.19,0.68,0.40;
    0.35,0.75,0.88;
    0, 0.4470, 0.7410;
    0.09,0.31,0.38;
    0.35,0.75,0.88;
    0, 0.4470, 0.7410;
    0.09,0.31,0.38;
    0.40,0.38,0.06;
    0.30,0.75,0.93;
    0.00,0.45,0.74];
%% plot
set(0,'defaultfigurecolor','w'); % White background
figure();
fig = gcf;
fig.Position = [200 200 1000 600];
scatter3(point(1,:), point(2,:), point(3,:), Marker='o', SizeData= 230,MarkerFaceColor=  color_lib(17,:),...
    MarkerEdgeColor='k');
hold on
if flag.line ==1
    for i = 1:size(connections, 1)
        p1 = connections(i, 1);       % begin index
        p2 = connections(i, 2);       % end index
        line([point(1,p1), point(1,p2)], [point(2,p1), point(2,p2)], [point(3,p1), point(3,p2)], ...
             Color=color_lib(16,:), LineWidth= 5);
    end 
end
hold on
%% plot setting
% xlabel('Z');
% ylabel('X');
% zlabel('Y');


xlim([0 1])    
xticks([0:0.2:1])	
xticklabels({'0','0.2','0.4','0.6','0.8','1'})
ylim([0 1.1])    
yticks([0:0.2:1])	
yticklabels({'0','0.2','0.4','0.6','0.8','1'})
zlim([0 1.1])   
zticks([0:0.2:1])	
zticklabels({'0','0.2','0.4','0.6','0.8','1'})


% clim([0.993, 1]); 
set(gca,'xticklabel',[],'yticklabel',[],'zticklabel',[])
% set(gca,'xminortick','on','yminortick','on','zminortick','on');
set(gca,'YDir','normal')
set(gcf,'Color','w');
% set(gca,'linewidth',2)
set(gca, 'LooseInset', [0,0,0.01,0.01]);
set(gca, 'FontName', 'Times')
grid on;
box on;
ax = gca; 
ax.GridColor = [0.5, 0.5, 0.5];
ax.GridAlpha = 1;
ax.GridLineStyle = ':';
ax.GridLineWidth = 1.7;
% ax.MinorGridLineStyle = '-';
ax.TickDir = 'in';
ax.FontSize = 40;
ax.LineWidth = 3;
view(54, 11)
legend(lgd,Location='best')
hold off
end