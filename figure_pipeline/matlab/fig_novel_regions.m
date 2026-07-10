% Novel latent regions: clusters of chains occupying latent regions not cleanly
% labelled by Kincore. Full kinome (grey) + cluster centroids sized by n_chains,
% annotated with their dominant genes.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
L = readtable(fullfile(base,'v91_full_kinome_CORRECT.csv'));
C = readtable(fullfile(base,'extended','novel_regions','novel_region_clusters.csv'));
cmap = lines(height(C));

fig = figure('Color','w','Units','inches','Position',[1 1 12 9.5]);
ax = axes(fig); hold(ax,'on');
scatter(ax, L.z0, L.z1, 6, [0.80 0.80 0.80], 'filled', 'MarkerFaceAlpha',0.18);
labs = strings(height(C),1);
for i = 1:height(C)
    scatter(ax, C.centroid_z0(i), C.centroid_z1(i), max(200, 6*C.n_chains(i)), ...
            cmap(i,:), 'filled', 'MarkerEdgeColor','k','LineWidth',0.8);
    g = strsplit(string(C.top_genes(i)), ";");          % top 3 genes only
    tg = strjoin(g(1:min(3,numel(g))), ", ");
    labs(i) = sprintf('C%d (n=%d): %s', C.cluster(i), C.n_chains(i), tg);
end
xlabel(ax,'z0'); ylabel(ax,'z1');
title(ax,'Novel latent regions (top genes)');
pubstyle(ax); titlegap(ax); axis(ax,'tight');
xl = xlim(ax); xlim(ax, [xl(1)-0.05*range(xl), xl(2)+0.12*range(xl)]);  % room for labels
drawnow;
placelabels(ax, C.centroid_z0, C.centroid_z1, labs, 22);

out = fullfile(here,'figures_matlab','novel_regions_latent');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (%d clusters)\n', out, height(C));
