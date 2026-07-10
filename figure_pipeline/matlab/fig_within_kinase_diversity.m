% Within-kinase conformational diversity: one panel per kinase (target list,
% n>=20, up to 8), coloured by DFG state. Background = all chains in grey.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
L = readtable(fullfile(base,'v91_full_kinome_CORRECT.csv'));
gene = string(L.gene); dfg = string(L.dfg_spatial);
targets = ["ABL1","EGFR","BRAF","CDK2","MAPK14","AURKA","MAPK1","KIT", ...
           "JAK2","FGFR1","MAP2K1","PRKACA"];
keep = strings(0,1);
for g = targets
    if sum(gene==g) >= 20, keep(end+1,1) = g; end %#ok<SAGROW>
    if numel(keep) >= 8, break; end
end
nC = numel(keep); ncol = 4; nrow = ceil(nC/ncol);

fig = figure('Color','w','Units','inches','Position',[1 1 5.5*ncol 5*nrow]);
tl = tiledlayout(fig,nrow,ncol,'TileSpacing','compact','Padding','compact');
ax1 = [];
for i = 1:nC
    ax = nexttile(tl);
    if i==1, ax1 = ax; end
    m = gene==keep(i);
    facet_panel(ax, L.z0, L.z1, L.z0(m), L.z1(m), dfg(m), ...
                sprintf('%s (n=%d)', keep(i), nnz(m)));
    if mod(i-1,ncol)==0, ylabel(ax,'z1'); end
    if i > nC-ncol, xlabel(ax,'z0'); end
end
title(tl,'Within-kinase conformational diversity', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');
shared_dfg_legend(ax1, tl);

out = fullfile(here,'figures_matlab','within_kinase_diversity');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (%d kinases)\n', out, nC);
