% Top-15 conserved-distance features by mean|SHAP| for z1 (companion to z0).
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'lgbm_shap_top.csv'));
T = T(strcmp(string(T.target),"z1"), :);
[~,o] = sort(T.mean_abs_shap,'descend');
T = T(o(1:min(15,height(T))), :);
lab = arrayfun(@(i,j) sprintf('%d-%d', i, j), T.resi_i, T.resi_j, 'uni',0);
v = flipud(T.mean_abs_shap); lab = flipud(lab);

fig = figure('Color','w','Units','inches','Position',[1 1 11 9]);
ax = axes(fig);
barh(ax, v, 'FaceColor',[0.753 0.314 0.302], 'EdgeColor','none', 'BarWidth',0.78);
set(ax,'YTick',1:numel(lab),'YTickLabel',lab); ylim(ax,[0.4 numel(lab)+0.6]);
xlabel(ax,'mean |SHAP| (z1)'); ylabel(ax,'conserved residue pair');
title(ax,'Top conserved-distance features for z1');
pubstyle(ax); titlegap(ax);

out = fullfile(here,'figures_matlab','lgbm_shap_summary_z1');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
