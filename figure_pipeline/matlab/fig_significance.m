% Mutation significance scatter: effect size (Mahalanobis sigma) vs
% significance (-log10 perm p), v9.1 latent. Publication style.
here = fileparts(mfilename('fullpath'));
base = fileparts(here);                       % v91_lgbm_shap/
T = readtable(fullfile(base,'v91_significance_summary.csv'));
sig = T.mahalanobis_sigma;
p   = T.perm_pvalue;
ok = ~isnan(sig) & ~isnan(p);
sig = sig(ok); p = p(ok); gene = string(T.gene(ok)); mut = string(T.mutation(ok));
y = -log10(max(p, 1e-4));                      % cap for plotting
xs = max(sig, 0.05);

fig = figure('Color','w','Units','inches','Position',[1 1 12 9]);
ax = axes(fig); hold(ax,'on');
% colour by significance (both thresholds passed)
passed = (p < 0.05) & (sig > 3.03);
scatter(ax, xs(~passed), y(~passed), 90, [0.6 0.6 0.6], 'filled', ...
        'MarkerFaceAlpha',0.65, 'DisplayName','n.s.');
scatter(ax, xs(passed), y(passed), 130, [0.16 0.44 0.70], 'filled', ...
        'DisplayName','p<0.05 & \sigma>3.03');
set(ax,'XScale','log');
hL = legend(ax,'Location','southeast'); set(hL,'Box','off','FontSize',22);
% threshold lines
yl = ylim(ax); xl = xlim(ax);
plot(ax, [3.03 3.03], [0 max(y)*1.05], '--', 'Color',[0.75 0.30 0.10], 'LineWidth',2.2, 'HandleVisibility','off');
plot(ax, xl, -log10(0.05)*[1 1], '--', 'Color',[0.75 0.30 0.10], 'LineWidth',2.2, 'HandleVisibility','off');
xlabel(ax,'Mahalanobis \sigma (effect size)');
ylabel(ax,'-log_{10} permutation p');
title(ax,'Mutation latent shift vs significance (v9.1)');
pubstyle(ax); titlegap(ax);
text(ax, 3.2, 0.16, '\sigma = 3.03', 'Color',[0.75 0.30 0.10], ...
     'FontSize',22,'FontName','Arial');
% Label the notable hits: union of the most-significant (lowest p) and the
% largest-effect (highest sigma), so the visually extreme far-right points
% (e.g. MARK2, PRP4K, GRK1) get named alongside the lowest-p ones.
[~,op] = sort(p,'ascend');
[~,os] = sort(sig,'descend');
sel = unique([op(1:min(6,numel(op))); os(1:min(6,numel(os)))], 'stable');
lab = arrayfun(@(k) sprintf('%s %s',gene(k),mut(k)), sel, 'uni', 0);
drawnow;  % so axis limits/positions are final before label placement
placelabels(ax, xs(sel), y(sel), string(lab), 22);
out = fullfile(here,'figures_matlab','significance_scatter');
exportgraphics(fig, [out '.png'], 'Resolution', 600);
exportgraphics(fig, [out '.pdf'], 'ContentType','vector');
fprintf('wrote %s.png/.pdf  (n=%d, %d passed both thresholds)\n', out, numel(sig), sum(passed));
