% EGFR resistance/oncogenic mutations in the latent, relative to the WT
% centroid (origin). Arrows = centroid displacement; small points = the actual
% per-chain data (WT cloud + the individual mutant chains) so the spread /
% statistics are visible. T790M (gatekeeper resistance) and V948R.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'extended','egfr_escape_routes.csv'));
T = T(~isnan(T.delta_z0) & ~isnan(T.delta_z1), :);
[~,o] = sort(T.delta_latent,'descend'); T = T(o,:);
n = height(T); cmap = lines(n);
P = readtable(fullfile(base,'extended','egfr_escape_points.csv'));
grp = string(P.group);

fig = figure('Color','w','Units','inches','Position',[1 1 12 10]);
ax = axes(fig); hold(ax,'on');
xline(ax,0,'-','Color',[0.6 0.6 0.6],'LineWidth',0.8,'HandleVisibility','off');
yline(ax,0,'-','Color',[0.6 0.6 0.6],'LineWidth',0.8,'HandleVisibility','off');

wt = grp=="WT";
scatter(ax, P.dz0(wt), P.dz1(wt), 34, [0.62 0.62 0.62], 'filled', ...
        'MarkerFaceAlpha',0.55, 'DisplayName',sprintf('WT / other EGFR (n=%d)',nnz(wt)));

for i = 1:n
    c = cmap(i,:); mut = string(T.mutation(i));
    m = grp==mut;
    scatter(ax, P.dz0(m), P.dz1(m), 70, c, 'filled', 'MarkerEdgeColor','k', ...
            'LineWidth',0.3, 'MarkerFaceAlpha',0.85, ...
            'DisplayName',sprintf('%s (n=%d, \\sigma=%.1f)',mut,nnz(m),T.mahalanobis_sigma(i)));
end

scatter(ax,0,0,260,[0.20 0.20 0.20],'filled','Marker','x','LineWidth',3, ...
        'DisplayName','WT centroid');
xlabel(ax,'\Deltaz0 from WT centroid');
ylabel(ax,'\Deltaz1 from WT centroid');
title(ax,'EGFR mutation populations in the latent');
pubstyle(ax); titlegap(ax); axis(ax,'equal');
hL = legend(ax,'Location','northwest'); set(hL,'Box','off','FontSize',19);

out = fullfile(here,'figures_matlab','egfr_escape_routes');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (%d mutations; WT n=%d)\n', out, n, nnz(wt));
