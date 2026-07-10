% Drug off-target latent separation vs Davis 2011 S(3uM) selectivity.
% Tests whether latent on->off separation predicts measured selectivity.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'off_target_vs_literature','drug_offtarget_vs_literature.csv'));
ok = ~isnan(T.davis2011_S_3uM) & ~isnan(T.centroid_separation);
T = T(ok,:);
x = T.davis2011_S_3uM; y = T.centroid_separation;
bbw = strcmpi(strtrim(string(T.off_target_BBW)),"True");
red = [0.752 0 0]; blue = [0.098 0.463 0.824];

fig = figure('Color','w','Units','inches','Position',[1 1 12 9]);
ax = axes(fig); hold(ax,'on');
scatter(ax, x(~bbw), y(~bbw), 150, blue, 'filled', 'MarkerEdgeColor','k','LineWidth',0.6);
scatter(ax, x(bbw),  y(bbw),  150, red,  'filled', 'MarkerEdgeColor','k','LineWidth',0.6);
set(ax,'XScale','log');
xlabel(ax,'Davis 2011 S(3 \muM)  (lower = more selective)');
ylabel(ax,'latent on\rightarrowoff centroid separation');
title(ax,'Latent separation vs measured selectivity');
pubstyle(ax); titlegap(ax);
% Spearman for the caption
rho = corr(x,y,'Type','Spearman');
text(ax,0.04,0.96,sprintf('Spearman \\rho = %+.2f (n=%d)',rho,height(T)), ...
     'Units','normalized','FontName','Arial','FontSize',22,'VerticalAlignment','top');
% legend (frameless)
hL = legend(ax,{'no boxed warning','FDA boxed warning'},'Location','northeast');
set(hL,'Box','off','FontSize',22);
drawnow;
placelabels(ax, x, y, string(T.drug_ref), 22);

out = fullfile(here,'figures_matlab','drug_offtarget_vs_literature');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (n=%d, rho=%.2f)\n', out, height(T), rho);
