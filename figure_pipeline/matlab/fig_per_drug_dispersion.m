% Per-drug latent compactness (dispersion), horizontal bars, lower = tighter.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'per_drug_analysis','per_drug_summary.csv'));
T = T(T.n_chains >= 3, :);
[~,o] = sort(T.dispersion,'ascend');      % tightest at bottom of barh
T = T(o,:);
n = height(T);
cmap = parula(n);

fig = figure('Color','w','Units','inches','Position',[1 1 12 max(8,0.55*n+3)]);
ax = axes(fig); hold(ax,'on');
for i = 1:n
    barh(ax, i, T.dispersion(i), 0.7, 'FaceColor',cmap(i,:), 'EdgeColor','none');
end
set(ax,'YTick',1:n,'YTickLabel',string(T.name));
ylim(ax,[0.4 n+0.6]);
xmax = max(T.dispersion);
for i = 1:n
    text(ax, T.dispersion(i)+0.01*xmax, i, ...
         sprintf('  n=%d, %d kin', T.n_chains(i), T.n_kinases(i)), ...
         'FontName','Arial','FontSize',22,'VerticalAlignment','middle','Color',[0.3 0.3 0.3]);
end
xlim(ax,[0 xmax*1.28]);
xlabel(ax,'latent dispersion (mean dist. to centroid)');
title(ax,'Per-drug compactness in the latent');
pubstyle(ax); titlegap(ax);

out = fullfile(here,'figures_matlab','per_drug_dispersion_bars');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (n=%d drugs)\n', out, n);
