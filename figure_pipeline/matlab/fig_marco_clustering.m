% Marco follow-up #2: K-means (spherical) vs HDBSCAN (density, no sphericity)
% on the v9.1 latent. HDBSCAN gives cleaner separation + honest noise.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'marco_followup','clustering_labels.csv'));
z0 = T.z0; z1 = T.z1; km = T.kmeans4; hb = T.hdbscan;

fig = figure('Color','w','Units','inches','Position',[1 1 16 7.5]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% Panel A: K-means k=4
axA = nexttile(tl); hold(axA,'on');
ck = lines(max(km)+1);
for c = 0:max(km)
    m = km==c; scatter(axA, z0(m), z1(m), 10, ck(c+1,:), 'filled', 'MarkerFaceAlpha',0.5);
end
xlabel(axA,'z0'); ylabel(axA,'z1');
title(axA,'K-means (k=4, spherical)');
pubstyle(axA); axA.Title.FontSize=24;
text(axA,0.04,0.96,'silhouette = 0.60','Units','normalized','FontName','Arial', ...
     'FontSize',20,'VerticalAlignment','top','Color',[0.4 0.4 0.4]);

% Panel B: HDBSCAN (noise grey)
axB = nexttile(tl); hold(axB,'on');
noise = hb==-1;
scatter(axB, z0(noise), z1(noise), 8, [0.8 0.8 0.8], 'filled', 'MarkerFaceAlpha',0.4);
cl = sort(unique(hb(~noise))); ch = turbo(numel(cl));
for i = 1:numel(cl)
    m = hb==cl(i); scatter(axB, z0(m), z1(m), 10, ch(i,:), 'filled', 'MarkerFaceAlpha',0.7);
end
xlabel(axB,'z0'); ylabel(axB,'z1');
title(axB,sprintf('HDBSCAN (%d clusters, 14%% noise)',numel(cl)));
pubstyle(axB); axB.Title.FontSize=24;
text(axB,0.04,0.96,'silhouette = 0.65','Units','normalized','FontName','Arial', ...
     'FontSize',20,'VerticalAlignment','top','Color',[0.4 0.4 0.4]);

title(tl,'Latent clustering: K-means vs HDBSCAN (Marco Q2)', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');
out = fullfile(here,'figures_matlab','marco_clustering_kmeans_vs_hdbscan');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
