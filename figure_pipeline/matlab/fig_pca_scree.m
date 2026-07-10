% PCA vs autoencoder dimensionality: scree (variance explained per PC) +
% cumulative variance. Shows the 2-D non-linear AE captures the loop manifold
% far more compactly than linear PCA (~11 PCs to reach ~94%).
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'pca_variance_explained.csv'));
pc = T.PC; ve = T.var_explained*100; cum = T.cum_var*100;
navy=[0.192 0.373 0.557]; orange=[0.85 0.42 0.10];

fig = figure('Color','w','Units','inches','Position',[1 1 12 9]);
ax = axes(fig); hold(ax,'on');
bar(ax, pc, ve, 0.6, 'FaceColor',navy, 'EdgeColor','none', 'DisplayName','per-PC variance');
plot(ax, pc, cum, '-o', 'Color',orange, 'LineWidth',2.4, 'MarkerFaceColor',orange, ...
     'MarkerSize',8, 'DisplayName','cumulative');
text(ax, pc(end), cum(end)+2.5, sprintf('%.1f%% at PC%d', cum(end), pc(end)), ...
     'FontName','Arial','FontSize',22,'HorizontalAlignment','right','Color',orange);
xlabel(ax,'principal component'); ylabel(ax,'variance explained (%)');
title(ax,'Linear PCA dimensionality of the loop');
pubstyle(ax); titlegap(ax);
xlim(ax,[0.4 max(pc)+0.6]); ylim(ax,[0 100]); set(ax,'XTick',pc);
hL = legend(ax,'Location','east'); set(hL,'Box','off','FontSize',22);
% annotate the AE comparison (PCA needs ~11 PCs for ~94%; the 2-D AE matches it)
text(ax, 2.2, 32, sprintf(['2-D non-linear AE\n\\approx 11 linear PCs\n(~94%%) for comparable\ncoverage']), ...
     'FontName','Arial','FontSize',22,'Color',navy);

out = fullfile(here,'figures_matlab','pca_scree');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (PC1=%.1f%%, PC1-2=%.1f%%, PC1-%d=%.1f%%)\n', out, ve(1), cum(2), pc(end), cum(end));
