% Feature-importance method agreement: Spearman correlation between the four
% importance methods (LightGBM gain, SHAP, permutation, RF), for z0 and z1.
% Shows the residue ranking is stable across methods.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
d = fullfile(base,'fi_methods_agreement');
T0 = readtable(fullfile(d,'fi_method_agreement_z0.csv'));
T1 = readtable(fullfile(d,'fi_method_agreement_z1.csv'));

prettify = @(s) strrep(strrep(strrep(strrep(string(s), ...
    "lgbm_gain","gain"),"lgbm_shap_meanabs","SHAP"),"lgbm_permutation","perm"),"rf_impurity","RF");
pairlab = arrayfun(@(a,b) sprintf('%s\\leftrightarrow%s',a,b), ...
                   prettify(T0.method_a), prettify(T0.method_b), 'uni',0);
n = numel(pairlab);
y = (1:n);

fig = figure('Color','w','Units','inches','Position',[1 1 12 9]);
ax = axes(fig); hold(ax,'on');
hb = barh(ax, [T0.spearman, T1.spearman], 'EdgeColor','none');
hb(1).FaceColor=[0.192 0.373 0.557]; hb(2).FaceColor=[0.753 0.314 0.302];
hb(1).DisplayName='z0'; hb(2).DisplayName='z1';
set(ax,'YTick',1:n,'YTickLabel',pairlab,'TickLabelInterpreter','tex');
xlabel(ax,'Spearman \rho (pair-level importance)');
title(ax,'Feature-importance method agreement');
pubstyle(ax); titlegap(ax);
xlim(ax,[0 1]); ylim(ax,[0.4 n+0.6]);
hL = legend(ax,'Location','southeast'); set(hL,'Box','off','FontSize',22);

out = fullfile(here,'figures_matlab','fi_method_agreement');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (%d method pairs)\n', out, n);
