% Conservation / feature-selection figure for the FULL non-loop scaffold
% (both lobes). (A) per-residue structural coverage vs BRAF residue number
% (loop window shaded, excluded); (B) per-pair coverage distribution with the
% 0.75 cutoff that selects the LightGBM feature pairs.
% Reads coverage CSVs produced by dump_coverage_fullnonloop.py. Pass the
% coverage dir as the COVDIR variable, or it defaults to ../fi_fullnonloop/coverage.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
if ~exist('COVDIR','var') || isempty(COVDIR)
    COVDIR = fullfile(base,'coverage');   % where per_residue_coverage.csv lives
end
R = readtable(fullfile(COVDIR,'per_residue_coverage.csv'));
P = readtable(fullfile(COVDIR,'pair_coverage.csv'));

fig = figure('Color','w','Units','inches','Position',[1 1 14 9]);
tl = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
lo = 594; hi = 623;   % activation loop (target) - excluded

% (A) per-residue coverage
ax1 = nexttile(tl); hold(ax1,'on');
patch(ax1,[lo hi hi lo],[0 0 1.05 1.05],[0.9 0.9 0.9],'EdgeColor','none', ...
      'FaceAlpha',0.6,'HandleVisibility','off');
isN = strcmp(string(R.lobe),'N-lobe');
bar(ax1, R.braf_resi(isN), R.coverage(isN), 1.0,'FaceColor',[0.192 0.373 0.557],'EdgeColor','none','DisplayName','N-lobe');
bar(ax1, R.braf_resi(~isN), R.coverage(~isN), 1.0,'FaceColor',[0.851 0.373 0.008],'EdgeColor','none','DisplayName','C-lobe');
ylim(ax1,[0 1.05]);
text(ax1,(lo+hi)/2,0.95,'loop','HorizontalAlignment','center','FontName','Arial','FontSize',15,'Color',[0.4 0.4 0.4]);
ylabel(ax1,'per-residue coverage'); xlabel(ax1,'BRAF residue number');
hL = legend(ax1,'Location','southwest'); set(hL,'Box','off','FontSize',18);
pubstyle(ax1); xlim(ax1,[min(R.braf_resi)-5 max(R.braf_resi)+5]);

% (B) per-pair coverage distribution + cutoff
ax2 = nexttile(tl); hold(ax2,'on');
histogram(ax2, P.coverage, 40, 'FaceColor',[0.4 0.45 0.5],'EdgeColor','none');
xline(ax2, 0.75, '--', 'Color',[0.75 0.2 0.2],'LineWidth',2.5, ...
      'Label',sprintf('0.75 cutoff (%d/%d pairs kept)', sum(P.coverage>=0.75), height(P)), ...
      'LabelOrientation','horizontal','FontName','Arial','FontSize',16);
xlabel(ax2,'per-pair coverage (fraction of chains)'); ylabel(ax2,'# candidate pairs');
pubstyle(ax2);

title(tl,'Conserved-scaffold feature selection (full non-loop, both lobes)', ...
      'FontName','Arial','FontSize',24,'FontWeight','bold');
out = fullfile(here,'figures_matlab','conservation_feature_selection');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (%d residues, %d pairs, %d kept)\n', out, height(R), height(P), sum(P.coverage>=0.75));
