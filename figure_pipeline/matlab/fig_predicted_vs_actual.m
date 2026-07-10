% LightGBM predicted vs actual latent (test set), z0 + z1 panels.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'lgbm_test_predictions.csv'));
navy = [0.192 0.373 0.557];

fig = figure('Color','w','Units','inches','Position',[1 1 15 7]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
pairs = {'actual_z0','pred_z0','z0'; 'actual_z1','pred_z1','z1'};
for k = 1:2
    a = T.(pairs{k,1}); p = T.(pairs{k,2});
    ax = nexttile(tl); hold(ax,'on');
    mn = min([a;p]); mx = max([a;p]);
    plot(ax,[mn mx],[mn mx],'--','Color',[0.2 0.2 0.2],'LineWidth',1.8);
    scatter(ax, a, p, 42, navy, 'filled', 'MarkerFaceAlpha',0.5);
    r2 = 1 - sum((a-p).^2)/sum((a-mean(a)).^2);
    xlabel(ax,sprintf('Actual %s',pairs{k,3}));
    ylabel(ax,sprintf('Predicted %s',pairs{k,3}));
    title(ax,sprintf('%s:  R^2 = %.3f',pairs{k,3},r2));
    pubstyle(ax); axis(ax,'tight'); axis(ax,'square');
end
title(tl,'LightGBM predicting the latent from conserved distances', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');

out = fullfile(here,'figures_matlab','lgbm_predicted_vs_actual');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
