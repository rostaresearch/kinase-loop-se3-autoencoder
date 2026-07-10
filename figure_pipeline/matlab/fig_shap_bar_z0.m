% Top-15 conserved-distance features by mean|SHAP| for z0 (horizontal bar).
% MATLAB-native substitute for the SHAP beeswarm (same ranking information).
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'lgbm_shap_top.csv'));
T = T(strcmp(string(T.target),"z0"), :);
[~,o] = sort(T.mean_abs_shap,'descend');
T = T(o(1:min(15,height(T))), :);
% relabel d_482_514 -> "482-514"
lab = arrayfun(@(i,j) sprintf('%d-%d', i, j), T.resi_i, T.resi_j, 'uni',0);
% ascending so largest sits at top of barh
v = flipud(T.mean_abs_shap); lab = flipud(lab);

fig = figure('Color','w','Units','inches','Position',[1 1 11 9]);
ax = axes(fig);
barh(ax, v, 'FaceColor',[0.192 0.373 0.557], 'EdgeColor','none', 'BarWidth',0.78);
set(ax,'YTick',1:numel(lab),'YTickLabel',lab);
ylim(ax,[0.4 numel(lab)+0.6]);
xlabel(ax,'mean |SHAP| (z0)');
ylabel(ax,'conserved residue pair');
title(ax,'Top conserved-distance features for z0');
pubstyle(ax); titlegap(ax);

out = fullfile(here,'figures_matlab','lgbm_shap_summary_z0');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
