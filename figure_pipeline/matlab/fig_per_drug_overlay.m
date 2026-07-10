% Per-drug latent footprints: full kinome (grey) + per-drug centroids,
% coloured by drug, sized by n_chains, labelled. Ring = latent dispersion.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
L = readtable(fullfile(base,'v91_full_kinome_CORRECT.csv'));
D = readtable(fullfile(base,'per_drug_analysis','per_drug_summary.csv'));
D = D(D.n_chains >= 1, :);
n = height(D); cmap = lines(n);

fig = figure('Color','w','Units','inches','Position',[1 1 12.5 11]);
ax = axes(fig); hold(ax,'on');
scatter(ax, L.z0, L.z1, 6, [0.80 0.80 0.80], 'filled', 'MarkerFaceAlpha',0.18);
% centroids only (star marker, size ~ sqrt(n_chains)); rings removed (clutter)
for i = 1:n
    scatter(ax, D.z0_centroid(i), D.z1_centroid(i), max(160,40*sqrt(D.n_chains(i))), ...
            cmap(i,:), 'filled','Marker','p', 'MarkerEdgeColor','k','LineWidth',0.6);
end
xlabel(ax,'z0'); ylabel(ax,'z1');
title(ax,'Per-drug latent footprints');
pubstyle(ax); titlegap(ax); axis(ax,'tight');
% De-pile labels: many ATP-site inhibitors share one spot, so label only
% spatially-distinct centroids (greedy by chain count; skip if within
% mindist of an already-labelled drug). Keeps the figure readable.
[~,ord] = sort(D.n_chains,'descend');
mindist = 0.06*hypot(range(xlim(ax)),range(ylim(ax)));
keepL = false(height(D),1); kept = zeros(0,2);
for k = ord'
    c = [D.z0_centroid(k) D.z1_centroid(k)];
    if isempty(kept) || min(hypot(kept(:,1)-c(1),kept(:,2)-c(2))) > mindist
        keepL(k) = true; kept(end+1,:) = c; %#ok<SAGROW>
    end
end
drawnow;
placelabels(ax, D.z0_centroid(keepL), D.z1_centroid(keepL), string(D.name(keepL)), 22);

out = fullfile(here,'figures_matlab','per_drug_latent_overlay');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (n=%d drugs)\n', out, n);
