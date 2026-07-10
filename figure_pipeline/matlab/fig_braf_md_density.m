% BRAF V600E MD (Clayton/Shen 2025) projected on the latent: 3x3 grid
% (Monomer/Dimer/MixedDimer x apo/LY/PHI1). MD frame density (red) over the
% static-PDB background; orange = BRAF static chains; X = MD centroid;
% apo->drug arrow shows the conformational shift on inhibitor binding.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
L = readtable(fullfile(base,'v91_full_kinome_CORRECT.csv'));
isBraf = upper(string(L.gene))=="BRAF";
M = readtable(fullfile(base,'md_projection','braf_md_v91_all_frames.csv'));
sys = string(M.system);

rowOf = strings(height(M),1);
rowOf(contains(sys,"MixedDimer")) = "MixedDimer";
rowOf(contains(sys,"Monomer"))    = "Monomer";
rowOf(rowOf=="" & contains(sys,"Dimer")) = "Dimer";
colOf = strings(height(M),1);
colOf(contains(sys,"apo"))  = "apo";
colOf(contains(sys,"LY"))   = "LY";
colOf(contains(sys,"PHI1")) = "PHI1";

rows = ["Monomer","Dimer","MixedDimer"]; cols = ["apo","LY","PHI1"];

% plotting region from BRAF static + MD occupancy, padded by ~8% of the data range
occx = [L.z0(isBraf); M.z0]; occy = [L.z1(isBraf); M.z1];
padx = 0.08*(max(occx)-min(occx)); pady = 0.08*(max(occy)-min(occy));
xl = [min(occx)-padx max(occx)+padx]; yl = [min(occy)-pady max(occy)+pady];

% apo centroid per row (for drug-panel arrows)
apoC = nan(numel(rows),2);
for r = 1:numel(rows)
    m = rowOf==rows(r) & colOf=="apo";
    if any(m), apoC(r,:) = [mean(M.z0(m)) mean(M.z1(m))]; end
end

reds = [linspace(1,0.55,64)', linspace(0.93,0,64)', linspace(0.90,0.05,64)'];
orange = [0.95 0.55 0.15];

fig = figure('Color','w','Units','inches','Position',[1 1 16 15]);
tl = tiledlayout(fig,numel(rows),numel(cols),'TileSpacing','compact','Padding','compact');
for r = 1:numel(rows)
    for c = 1:numel(cols)
        ax = nexttile(tl); hold(ax,'on'); colormap(ax,reds);
        % background: all static chains (grey) + BRAF static (orange)
        scatter(ax, L.z0(~isBraf), L.z1(~isBraf), 5, [0.72 0.72 0.72], ...
                'filled','MarkerFaceAlpha',0.18,'HandleVisibility','off');
        scatter(ax, L.z0(isBraf), L.z1(isBraf), 22, orange, 'filled', ...
                'MarkerFaceAlpha',0.55,'MarkerEdgeColor','w','LineWidth',0.3);
        m = rowOf==rows(r) & colOf==cols(c);
        ttl = sprintf('%s %s (n=%d)', rows(r), cols(c), nnz(m));
        if any(m)
            nb = 45;
            xe = linspace(xl(1),xl(2),nb+1); ye = linspace(yl(1),yl(2),nb+1);
            N = histcounts2(M.z0(m), M.z1(m), xe, ye);
            N(N<2) = 0;
            xc = (xe(1:end-1)+xe(2:end))/2; yc = (ye(1:end-1)+ye(2:end))/2;
            him = imagesc(ax, xc, yc, N'); set(ax,'YDir','normal');
            set(him,'AlphaData', double(N'>0)*0.85);
            cen = [mean(M.z0(m)) mean(M.z1(m))];
            if cols(c)~="apo" && all(~isnan(apoC(r,:)))
                a = apoC(r,:);
                quiver(ax, a(1),a(2), cen(1)-a(1),cen(2)-a(2),0,'Color','k', ...
                       'LineWidth',2,'MaxHeadSize',0.5,'HandleVisibility','off');
                scatter(ax, a(1),a(2), 120, 'w','filled','MarkerEdgeColor','k','LineWidth',1.2);
                text(ax,(a(1)+cen(1))/2,(a(2)+cen(2))/2, ...
                     sprintf(' |\\Delta|=%.1f',norm(cen-a)), ...
                     'FontName','Arial','FontSize',20,'FontWeight','bold');
            end
            scatter(ax, cen(1),cen(2), 160, 'k','filled','Marker','x','LineWidth',2.5);
        end
        xlim(ax,xl); ylim(ax,yl);
        title(ax,ttl,'FontName','Arial','FontSize',22,'FontWeight','bold');
        pubstyle(ax); ax.Title.FontSize=22;
        if c==1, ylabel(ax,'z1'); end
        if r==numel(rows), xlabel(ax,'z0'); end
    end
end
title(tl,'BRAF V600E MD projected on the latent', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');

out = fullfile(here,'figures_matlab','braf_md_density_on_v91_latent');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
