% Marco follow-up #3: Mahalanobis (whitened) latent space. Left: original
% anisotropic latent; right: Mahalanobis/whitened space (isotropic). Both
% coloured by the SAME density clusters -> clusters are ~metric-invariant.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'marco_followup','clustering_mahalanobis_labels.csv'));
lab = T.maha_hdbscan; cl = sort(unique(lab(lab>=0))); ch = turbo(numel(cl));
nz = lab==-1;

fig = figure('Color','w','Units','inches','Position',[1 1 16 7.5]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% Panel A: original anisotropic latent
axA = nexttile(tl); hold(axA,'on');
scatter(axA, T.z0(nz), T.z1(nz), 8, [0.8 0.8 0.8], 'filled', 'MarkerFaceAlpha',0.4);
for i = 1:numel(cl)
    m = lab==cl(i); scatter(axA, T.z0(m), T.z1(m), 10, ch(i,:), 'filled', 'MarkerFaceAlpha',0.7);
end
xlabel(axA,'z0'); ylabel(axA,'z1'); title(axA,'Original latent (anisotropic)');
pubstyle(axA); axA.Title.FontSize=24; axis(axA,'tight');

% Panel B: Mahalanobis / whitened space
axB = nexttile(tl); hold(axB,'on');
scatter(axB, T.z0_white(nz), T.z1_white(nz), 8, [0.8 0.8 0.8], 'filled', 'MarkerFaceAlpha',0.4);
for i = 1:numel(cl)
    m = lab==cl(i); scatter(axB, T.z0_white(m), T.z1_white(m), 10, ch(i,:), 'filled', 'MarkerFaceAlpha',0.7);
end
xlabel(axB,'z0'''); ylabel(axB,'z1'''); title(axB,'Mahalanobis / whitened space');
pubstyle(axB); axB.Title.FontSize=24; axis(axB,'equal');

title(tl,'Mahalanobis latent similarity (Marco Q3) — clusters ~metric-invariant (ARI=0.93)', ...
      'FontName','Arial','FontSize',24,'FontWeight','bold');
out = fullfile(here,'figures_matlab','marco_mahalanobis_latent');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
