% MAP2K1 phosphorylation worked example. Panel A: global latent with MAP2K1
% chains highlighted. Panel B: WT centroid -> phospho-mutant displacement.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
L = readtable(fullfile(base,'v91_full_kinome_CORRECT.csv'));
ck = upper(string(L.chain_key)); gene = upper(string(L.gene));
dfg = string(L.dfg_spatial);

DEAD = upper(string(split("6NYBB;6PP9B;6Q0JC;6Q0JD;6Q0TC;6V2WB;7M0TB;7M0UB;7M0VB;7M0WB;7M0XB;7M0YB;7M0ZB;8CHFE;8CHFF;8DGSB;8DGTB",";")));
MIM  = upper(["5YT3B","5YT3D"]);
isM  = gene=="MAP2K1";
isDead = ismember(ck,DEAD); isMim = ismember(ck,MIM);
isWT = isM & ~isDead & ~isMim;

grey=[0.30 0.30 0.30]; blue=[0.235 0.471 0.847]; red=[0.80 0 0];
mimMk=140; deadMk=80; wtMk=40;

fig = figure('Color','w','Units','inches','Position',[1 1 16 7.5]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% ---- Panel A: global ----
axA = nexttile(tl); hold(axA,'on');
for s = ["DFGin","DFGinter","DFGout"]
    m = dfg==s & ~isM;
    scatter(axA, L.z0(m), L.z1(m), 5, dfgcolor(s), 'filled', ...
            'MarkerFaceAlpha',0.28,'HandleVisibility','off');
end
scatter(axA, L.z0(isWT), L.z1(isWT), wtMk, grey, 'filled','MarkerEdgeColor','k', ...
        'LineWidth',0.3,'DisplayName',sprintf('MAP2K1 WT (n=%d)',nnz(isWT)));
scatter(axA, L.z0(isDead),L.z1(isDead), deadMk, blue,'filled','MarkerEdgeColor','k', ...
        'LineWidth',0.4,'DisplayName',sprintf('S218A+S222A (n=%d)',nnz(isDead)));
scatter(axA, L.z0(isMim), L.z1(isMim), mimMk, red,'filled','Marker','p','MarkerEdgeColor','k', ...
        'LineWidth',0.5,'DisplayName',sprintf('S218D+S222D (n=%d)',nnz(isMim)));
xlabel(axA,'z0'); ylabel(axA,'z1');
title(axA,'A. global latent, MAP2K1 highlighted');
pubstyle(axA); axA.Title.FontSize=24;
hL=legend(axA,'Location','southwest'); set(hL,'Box','off','FontSize',20);

% ---- Panel B: displacement (zoom on MAP2K1) ----
axB = nexttile(tl); hold(axB,'on');
wt = [mean(L.z0(isWT)) mean(L.z1(isWT))];
cd = [mean(L.z0(isDead)) mean(L.z1(isDead))];
cm = [mean(L.z0(isMim)) mean(L.z1(isMim))];
ddead = norm(cd-wt); dmim = norm(cm-wt);
mz0=L.z0(isM); mz1=L.z1(isM);
rng = max([max(mz0)-min(mz0), max(mz1)-min(mz1)]); off = 0.10*rng;   % scale-aware offsets (SE(3) latent ~ unit scale)
scatter(axB, mz0, mz1, wtMk, grey, 'filled','MarkerFaceAlpha',0.5,'HandleVisibility','off');
scatter(axB, L.z0(isDead),L.z1(isDead), deadMk, blue,'filled','MarkerEdgeColor','k','LineWidth',0.4);
scatter(axB, L.z0(isMim), L.z1(isMim), mimMk, red,'filled','Marker','p','MarkerEdgeColor','k','LineWidth',0.5);
quiver(axB, wt(1),wt(2), cd(1)-wt(1),cd(2)-wt(2), 0,'Color',blue,'LineWidth',2.2,'MaxHeadSize',0.4);
quiver(axB, wt(1),wt(2), cm(1)-wt(1),cm(2)-wt(2), 0,'Color',red,'LineWidth',2.2,'MaxHeadSize',0.4);
scatter(axB, wt(1),wt(2), 220, [0.06 0.06 0.06],'filled','Marker','x','LineWidth',3);
text(axB, wt(1)-0.3*off, wt(2)+off, 'WT centroid','FontName','Arial','FontSize',22, ...
     'HorizontalAlignment','center','VerticalAlignment','bottom');
% annotations as a colour-coded block in the empty upper-left corner (normalized units, no clipping)
text(axB, 0.04, 0.97, sprintf('mimetic:  \\sigma=3.16  (|\\Delta|=%.2f)',dmim), ...
     'Units','normalized','Color',red,'FontName','Arial','FontSize',22, ...
     'HorizontalAlignment','left','VerticalAlignment','top');
text(axB, 0.04, 0.87, sprintf('dead:  \\sigma=0.48 (n.s.)  (|\\Delta|=%.2f)',ddead), ...
     'Units','normalized','Color',blue,'FontName','Arial','FontSize',22, ...
     'HorizontalAlignment','left','VerticalAlignment','top');
xlabel(axB,'z0'); ylabel(axB,'z1');
title(axB,'B. WT \rightarrow mutant displacement');
pubstyle(axB); axB.Title.FontSize=24;
pad=0.18*rng;
xlim(axB,[min([mz0;cm(1);cd(1)])-pad max([mz0;cm(1);cd(1)])+pad]);
ylim(axB,[min([mz1;cm(2);cd(2)])-pad max([mz1;cm(2);cd(2)])+pad]);

out = fullfile(here,'figures_matlab','map2k1_phospho_example');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (WT=%d dead=%d mim=%d; |d|dead=%.1f mim=%.1f)\n', ...
        out, nnz(isWT),nnz(isDead),nnz(isMim),ddead,dmim);
