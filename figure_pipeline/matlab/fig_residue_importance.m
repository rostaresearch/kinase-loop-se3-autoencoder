% Per-residue SHAP importance (z0 top, z1 bottom) vs BRAF residue number.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'lgbm_residue_importance.csv'));
resi = T.braf_resi;

fig = figure('Color','w','Units','inches','Position',[1 1 14 9]);
tl = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');

lo = 594; hi = 623;   % activation loop (target) - excluded from the features

ax1 = nexttile(tl); hold(ax1,'on');
yl1 = [0 max(T.shap_z0)*1.08];
patch(ax1,[lo hi hi lo],[yl1(1) yl1(1) yl1(2) yl1(2)],[0.9 0.9 0.9], ...
      'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');
bar(ax1, resi, T.shap_z0, 1.0, 'FaceColor',[0.192 0.373 0.557], 'EdgeColor','none');
ylim(ax1,yl1);
text(ax1,(lo+hi)/2, yl1(2)*0.9,'loop','HorizontalAlignment','center', ...
     'FontName','Arial','FontSize',15,'Color',[0.4 0.4 0.4]);
ylabel(ax1,'\Sigma |SHAP| (z0)'); pubstyle(ax1);
set(ax1,'XTickLabel',[]);

ax2 = nexttile(tl); hold(ax2,'on');
yl2 = [0 max(T.shap_z1)*1.08];
patch(ax2,[lo hi hi lo],[yl2(1) yl2(1) yl2(2) yl2(2)],[0.9 0.9 0.9], ...
      'EdgeColor','none','FaceAlpha',0.6,'HandleVisibility','off');
bar(ax2, resi, T.shap_z1, 1.0, 'FaceColor',[0.753 0.314 0.302], 'EdgeColor','none');
ylim(ax2,yl2);
text(ax2,(lo+hi)/2, yl2(2)*0.9,'loop','HorizontalAlignment','center', ...
     'FontName','Arial','FontSize',15,'Color',[0.4 0.4 0.4]);
xlabel(ax2,'BRAF residue number'); ylabel(ax2,'\Sigma |SHAP| (z1)'); pubstyle(ax2);

title(tl,'Per-residue importance for predicting the latent (N-lobe + C-lobe)', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');
linkaxes([ax1 ax2],'x'); xlim(ax1,[min(resi)-5 max(resi)+5]);

out = fullfile(here,'figures_matlab','lgbm_residue_importance');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
