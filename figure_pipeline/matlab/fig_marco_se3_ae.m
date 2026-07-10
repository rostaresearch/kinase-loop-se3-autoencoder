% Marco follow-up #6: SE(3)-invariant distance-matrix AE vs coordinate AE.
% Left: coordinate FoldingNet latent (z0 ~ loop placement, anisotropic).
% Right: SE(3) distance-matrix latent (isotropic, placement-free, better DFG
% separation). Both coloured by Kincore DFG state.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'marco_followup','q6_se3_vs_coord.csv'));
dfg = string(T.dfg_spatial);
states = ["DFGin","DFGout","DFGinter"];

fig = figure('Color','w','Units','inches','Position',[1 1 16 7.5]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% Panel A: coordinate AE
axA = nexttile(tl); hold(axA,'on');
mo=~ismember(dfg,states); scatter(axA,T.z0(mo),T.z1(mo),6,dfgcolor("other"),'filled','MarkerFaceAlpha',0.25,'HandleVisibility','off');
for s=states, m=dfg==s; scatter(axA,T.z0(m),T.z1(m),10,dfgcolor(s),'filled','MarkerFaceAlpha',0.5,'DisplayName',char(s)); end
xlabel(axA,'z0'); ylabel(axA,'z1'); title(axA,'Coordinate AE (z0 \approx placement)');
pubstyle(axA); axA.Title.FontSize=24;
text(axA,0.04,0.96,{'anisotropy 2.7\times','z0 vs placement |r|=0.97','DFG silhouette +0.08'}, ...
     'Units','normalized','FontName','Arial','FontSize',18,'VerticalAlignment','top','Color',[0.35 0.35 0.35]);

% Panel B: SE(3) distance-matrix AE
axB = nexttile(tl); hold(axB,'on');
scatter(axB,T.s0(mo),T.s1(mo),6,dfgcolor("other"),'filled','MarkerFaceAlpha',0.25,'HandleVisibility','off');
for s=states, m=dfg==s; scatter(axB,T.s0(m),T.s1(m),10,dfgcolor(s),'filled','MarkerFaceAlpha',0.5,'DisplayName',char(s)); end
xlabel(axB,'s0'); ylabel(axB,'s1'); title(axB,'SE(3) distance-matrix AE');
pubstyle(axB); axB.Title.FontSize=24; axis(axB,'equal');
text(axB,0.04,0.96,{'isotropic (1.0\times)','s vs placement |r|=0.11','DFG silhouette +0.09'}, ...
     'Units','normalized','FontName','Arial','FontSize',18,'VerticalAlignment','top','Color',[0.35 0.35 0.35]);
hL=legend(axB,'Location','southeast'); set(hL,'Box','off','FontSize',20);

title(tl,'SE(3)-invariant latent removes the placement confound (Marco Q6)', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');
out = fullfile(here,'figures_matlab','marco_se3_vs_coord_ae');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
