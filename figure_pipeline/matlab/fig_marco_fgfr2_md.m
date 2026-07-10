% Marco follow-up #4: FGFR2 MD (Mahapatra & Kar 2025) projected on the v9.1
% latent, CORRECTED to the BRAF 6UAN reference frame. 4 systems (WT unphos /
% WT phos / N549K / K659E). MD-frame density (red) over the kinome (grey) with
% FGFR2 experimental chains (orange) + MD centroid.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
L = readtable(fullfile(base,'v91_full_kinome_CORRECT.csv'));
isF = upper(string(L.gene))=="FGFR2";
M = readtable(fullfile(base,'marco_followup','fgfr2','fgfr2_corrected_all_frames.csv'));
sys = string(M.system);
order = ["wt_unphos","wt_phos","n549k","k659e"];
titles = ["FGFR2 WT (unphos)","FGFR2 WT (phos)","FGFR2 N549K","FGFR2 K659E"];

% plotting region: FGFR2 experimental + MD occupancy, padded by ~8% of the data range
occx=[L.z0(isF); M.z0]; occy=[L.z1(isF); M.z1];
px=0.08*(max(occx)-min(occx)); py=0.08*(max(occy)-min(occy));
xl=[min(occx)-px max(occx)+px]; yl=[min(occy)-py max(occy)+py];
reds=[linspace(1,0.55,64)', linspace(0.93,0,64)', linspace(0.90,0.05,64)'];
orange=[0.95 0.55 0.15];

fig = figure('Color','w','Units','inches','Position',[1 1 15 12]);
tl = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
for k=1:numel(order)
    ax=nexttile(tl); hold(ax,'on'); colormap(ax,reds);
    scatter(ax, L.z0(~isF), L.z1(~isF), 5, [0.72 0.72 0.72],'filled','MarkerFaceAlpha',0.15,'HandleVisibility','off');
    scatter(ax, L.z0(isF), L.z1(isF), 26, orange,'filled','MarkerFaceAlpha',0.6,'MarkerEdgeColor','w','LineWidth',0.3);
    m = sys==order(k);
    if any(m)
        nb=40; xe=linspace(xl(1),xl(2),nb+1); ye=linspace(yl(1),yl(2),nb+1);
        N=histcounts2(M.z0(m),M.z1(m),xe,ye); N(N<2)=0;
        xc=(xe(1:end-1)+xe(2:end))/2; yc=(ye(1:end-1)+ye(2:end))/2;
        him=imagesc(ax,xc,yc,N'); set(ax,'YDir','normal'); set(him,'AlphaData',double(N'>0)*0.85);
        cen=[mean(M.z0(m)) mean(M.z1(m))];
        scatter(ax,cen(1),cen(2),160,'k','filled','Marker','x','LineWidth',2.5);
    end
    xlim(ax,xl); ylim(ax,yl);
    title(ax,sprintf('%s (n=%d)',titles(k),nnz(m)),'FontName','Arial','FontSize',22,'FontWeight','bold','Interpreter','none');
    pubstyle(ax); ax.Title.FontSize=22;
    if k>2, xlabel(ax,'z0'); end
    if mod(k,2)==1, ylabel(ax,'z1'); end
end
title(tl,'FGFR2 MD on the v9.1 latent — 6UAN-referenced (Marco Q4)', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');
out = fullfile(here,'figures_matlab','marco_fgfr2_md_density');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
