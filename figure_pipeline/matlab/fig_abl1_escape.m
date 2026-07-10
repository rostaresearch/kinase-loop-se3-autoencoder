% ABL1 resistance-mutation escape vectors in the latent, relative to the WT
% centroid (origin). Arrows = centroid displacement; small points = the actual
% per-chain data (WT cloud + the individual mutant chains) so the spread /
% statistics are visible.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'extended','abl1_escape_routes.csv'));
T = T(~isnan(T.delta_z0) & ~isnan(T.delta_z1), :);
[~,o] = sort(T.delta_latent,'descend'); T = T(o,:);
n = height(T); cmap = lines(n);
P = readtable(fullfile(base,'extended','abl1_escape_points.csv'));   % per-chain dz0,dz1,group
grp = string(P.group);

fig = figure('Color','w','Units','inches','Position',[1 1 12 10]);
ax = axes(fig); hold(ax,'on');
xline(ax,0,'-','Color',[0.6 0.6 0.6],'LineWidth',0.8,'HandleVisibility','off');
yline(ax,0,'-','Color',[0.6 0.6 0.6],'LineWidth',0.8,'HandleVisibility','off');

% --- WT per-chain cloud (background) ---
wt = grp=="WT";
scatter(ax, P.dz0(wt), P.dz1(wt), 34, [0.62 0.62 0.62], 'filled', ...
        'MarkerFaceAlpha',0.55, 'DisplayName',sprintf('WT chains (n=%d)',nnz(wt)));

% --- arrows + per-chain mutant points, coloured per mutation ---
for i = 1:n
    c = cmap(i,:); mut = string(T.mutation(i));
    quiver(ax, 0,0, T.delta_z0(i),T.delta_z1(i), 0, 'Color',c, ...
           'LineWidth',2.0,'MaxHeadSize',0.25,'HandleVisibility','off');
    m = grp==mut;
    scatter(ax, P.dz0(m), P.dz1(m), 90, c, 'filled', 'MarkerEdgeColor','k', ...
            'LineWidth',0.5, 'DisplayName',sprintf('%s (n=%d)',mut,nnz(m)));
    % centroid marker (hollow, on top)
    scatter(ax, T.delta_z0(i),T.delta_z1(i), 150, c, 'Marker','o', ...
            'LineWidth',2.2,'MarkerEdgeColor',c,'HandleVisibility','off');
end

scatter(ax,0,0,260,[0.30 0.45 0.69],'filled','Marker','x','LineWidth',3,'HandleVisibility','off');
text(ax,0,0,'  WT centroid','FontName','Arial','FontSize',22,'VerticalAlignment','top');
xlabel(ax,'\Deltaz0 from WT centroid');
ylabel(ax,'\Deltaz1 from WT centroid');
title(ax,'ABL1 resistance-mutation escape routes');
pubstyle(ax); titlegap(ax); axis(ax,'equal');
hL = legend(ax,'Location','northwest'); set(hL,'Box','off','FontSize',20);
drawnow;
placelabels(ax, T.delta_z0, T.delta_z1, string(T.mutation), 22);

out = fullfile(here,'figures_matlab','abl1_escape_routes');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (%d mutations; WT n=%d)\n', out, n, nnz(wt));
