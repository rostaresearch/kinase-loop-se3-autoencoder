% Drug-design selectivity landscape: per-kinase own latent spread (compactness)
% vs isolation from other kinases. Bottom-right = compact AND isolated = easy target.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
C = readtable(fullfile(base,'per_gene_compactness.csv'));
I = readtable(fullfile(base,'per_gene_selectivity_isolation.csv'));
M = innerjoin(C, I, 'Keys','gene');
x = M.p95_radius_from_centroid;     % own spread
y = M.median;                       % isolation distance
nc = M.n_chains;
navy = [0.192 0.373 0.557];

fig = figure('Color','w','Units','inches','Position',[1 1 12 9.5]);
ax = axes(fig); hold(ax,'on');
scatter(ax, x, y, max(40, nc*3), navy, 'filled', ...
        'MarkerFaceAlpha',0.55, 'MarkerEdgeColor','k','LineWidth',0.4);
xlabel(ax,'own latent spread (p95 radius)');
ylabel(ax,'median isolation from other kinases');
title(ax,'Per-kinase selectivity landscape');
pubstyle(ax); titlegap(ax);
% label easiest targets (high isolation, low spread) + a few hardest
score = y - x;                       % isolated & compact -> high
[~,o] = sort(score,'descend');
sel = [o(1:min(10,numel(o))); o(end-min(4,numel(o)-1):end)];
sel = unique(sel,'stable');
drawnow;
placelabels(ax, x(sel), y(sel), string(M.gene(sel)), 22);

out = fullfile(here,'figures_matlab','compactness_vs_isolation_scatter');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (n=%d genes)\n', out, height(M));
