% Marco follow-up #1: AE latent vs PCA of the encoder input, and the PCA
% axis-scale reconciliation (centered vs superposed vs standardised).
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'marco_followup','pca_vs_latent_scores.csv'));
navy=[0.192 0.373 0.557]; red=[0.753 0.314 0.302]; teal=[0.20 0.60 0.55];
rr = @(a,b) abs(corr(a,b));

fig = figure('Color','w','Units','inches','Position',[1 1 16 7.5]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% Panel A: z0 vs PC1 (of the standardised coords the AE saw)
axA = nexttile(tl); hold(axA,'on');
scatter(axA, T.PC1_std, T.z0, 26, navy, 'filled', 'MarkerFaceAlpha',0.30);
xlabel(axA,'PC1 of encoder input'); ylabel(axA,'AE latent z0');
title(axA,sprintf('z0 \\approx PC1   (|r| = %.2f)', rr(T.z0,T.PC1_std)));
pubstyle(axA); axA.Title.FontSize=24;
text(axA,0.05,0.95,sprintf('z1 vs PC2: |r| = %.2f',rr(T.z1,T.PC2_std)), ...
     'Units','normalized','FontName','Arial','FontSize',20,'VerticalAlignment','top','Color',[0.4 0.4 0.4]);

% Panel B: PC1/PC2 score span across the three representations (scale question)
axB = nexttile(tl); hold(axB,'on');
span = @(v) max(v)-min(v);
S = [span(T.PC1_centered) span(T.PC2_centered);
     span(T.PC1_superposed) span(T.PC2_superposed);
     span(T.PC1_std) span(T.PC2_std)];
hb = bar(axB, S, 0.8, 'EdgeColor','none');
hb(1).FaceColor=navy; hb(2).FaceColor=teal;
hb(1).DisplayName='PC1 span'; hb(2).DisplayName='PC2 span';
set(axB,'XTick',1:3,'XTickLabel',{'centered','superposed','standardised'});
ylabel(axB,'PC score span'); title(axB,'PCA axis scale by alignment');
pubstyle(axB); axB.Title.FontSize=24;
hL=legend(axB,'Location','north'); set(hL,'Box','off','FontSize',20);

title(tl,'AE latent vs PCA of the encoder input (Marco Q1)', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');

if ~exist(fullfile(here,'figures_matlab'),'dir'), mkdir(fullfile(here,'figures_matlab')); end
out = fullfile(here,'figures_matlab','marco_pca_vs_latent');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
