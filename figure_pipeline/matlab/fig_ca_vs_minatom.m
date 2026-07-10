% Cα–Cα vs minimum heavy-atom distance features: held-out R^2 for z0 and z1.
% Shows the choice of distance metric barely changes downstream accuracy.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'ca_vs_minatom','ca_vs_minatom_model_comparison.csv'));
fs = string(T.feature_set); tg = string(T.target);
val = @(f,t) T.r2_test(fs==f & tg==t);
M = [val("ca","z0") val("min","z0"); val("ca","z1") val("min","z1")];  % rows=z0,z1 cols=ca,min
navy=[0.192 0.373 0.557]; teal=[0.20 0.60 0.55];

fig = figure('Color','w','Units','inches','Position',[1 1 11 9]);
ax = axes(fig); hold(ax,'on');
hb = bar(ax, M, 0.75, 'EdgeColor','none');
hb(1).FaceColor=navy; hb(2).FaceColor=teal;
hb(1).DisplayName='C\alpha–C\alpha'; hb(2).DisplayName='min heavy-atom';
set(ax,'XTick',1:2,'XTickLabel',{'z0','z1'});
ylabel(ax,'held-out R^2'); ylim(ax,[0 1]);
title(ax,'Distance metric: C\alpha vs min heavy-atom');
pubstyle(ax);
% value labels above bars
xo=[-0.15 0.15];
for i=1:2, for j=1:2
    text(ax, i+xo(j), M(i,j)+0.02, sprintf('%.3f',M(i,j)), ...
         'HorizontalAlignment','center','FontName','Arial','FontSize',22);
end, end
titlegap(ax);
hL = legend(ax,'Location','south'); set(hL,'Box','off','FontSize',22);

out = fullfile(here,'figures_matlab','ca_vs_minatom');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
