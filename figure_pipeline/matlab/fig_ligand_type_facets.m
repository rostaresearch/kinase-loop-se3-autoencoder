% Ligand-type latent footprints, one panel per ligand type (n>=50),
% coloured by Kincore DFG state. Background = all chains in grey.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
L = readtable(fullfile(base,'v91_full_kinome_CORRECT.csv'));
lig = string(L.ligand_type);
lig(strlength(lig)==0) = "No_ligand";
dfg = string(L.dfg_spatial);

cats = unique(lig);
keep = strings(0,1);
for c = cats'
    if c~="No_ligand" && sum(lig==c) >= 50, keep(end+1,1) = c; end %#ok<SAGROW>
end
% order by abundance
cnt = arrayfun(@(c) sum(lig==c), keep);
[~,o] = sort(cnt,'descend'); keep = keep(o);
nC = numel(keep); ncol = 3; nrow = ceil(nC/ncol);

fig = figure('Color','w','Units','inches','Position',[1 1 6*ncol 5*nrow]);
tl = tiledlayout(fig,nrow,ncol,'TileSpacing','compact','Padding','compact');
ax1 = [];
for i = 1:nC
    ax = nexttile(tl);
    if i==1, ax1 = ax; end
    m = lig==keep(i);
    facet_panel(ax, L.z0, L.z1, L.z0(m), L.z1(m), dfg(m), ...
                sprintf('%s (n=%d)', keep(i), nnz(m)));
    if mod(i-1,ncol)==0, ylabel(ax,'z1'); end
    if i > nC-ncol, xlabel(ax,'z0'); end
end
title(tl,'Ligand-type latent footprints (by DFG state)', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');
shared_dfg_legend(ax1, tl);

out = fullfile(here,'figures_matlab','ligand_type_latent_facets');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (%d ligand types)\n', out, nC);
