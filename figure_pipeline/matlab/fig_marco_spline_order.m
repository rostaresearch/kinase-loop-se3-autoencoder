% Marco follow-up #5: is cubic the right spline order? Left: leave-one-out
% held-out Ca error vs spline order (U-shape, min at 2-3). Right: how much the
% final 27-point representation changes vs cubic. Cubic is near-optimal.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'marco_followup','spline_order_loo.csv'));
orders = [1 2 3 4 5];
navy=[0.192 0.373 0.557]; hi=[0.85 0.42 0.10];

fig = figure('Color','w','Units','inches','Position',[1 1 16 7.5]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% Panel A: LOO error distribution per order (boxchart), cubic highlighted
axA = nexttile(tl); hold(axA,'on');
for i=1:numel(orders)
    v = T.loo_mean_per_seg(T.order==orders(i));
    c = navy; if orders(i)==3, c=hi; end
    bc = boxchart(axA, orders(i)*ones(size(v)), v, 'BoxFaceColor',c, 'MarkerStyle','none','BoxWidth',0.6);
end
% median line
meds = arrayfun(@(k) median(T.loo_mean_per_seg(T.order==k)), orders);
plot(axA, orders, meds, '-o','Color',[0.3 0.3 0.3],'LineWidth',2,'MarkerFaceColor','w','HandleVisibility','off');
set(axA,'XTick',orders); xlabel(axA,'spline order'); ylabel(axA,'leave-one-out C\alpha error (Å)');
title(axA,'Loop-fit accuracy vs order'); pubstyle(axA); axA.Title.FontSize=24;
ylim(axA,[0 max(T.loo_mean_per_seg)*0.6]);
text(axA,3,meds(3)+0.15,'cubic (current)','Color',hi,'FontName','Arial','FontSize',20,'HorizontalAlignment','center');

% Panel B: representation change vs cubic (from run output)
axB = nexttile(tl); hold(axB,'on');
repdiff = [0.4021 0.2146 0.0 0.2762 0.6768];   % median 27-pt RMSD vs cubic
b = bar(axB, orders, repdiff, 0.6, 'FaceColor',navy,'EdgeColor','none');
b.FaceColor='flat'; b.CData(3,:)=hi;
set(axB,'XTick',orders); xlabel(axB,'spline order'); ylabel(axB,'27-pt RMSD vs cubic (Å)');
title(axB,'Representation change vs order'); pubstyle(axB); axB.Title.FontSize=24;
text(axB,0.06,0.88,{'all \leq 0.7 Å —', 'small vs 2.3 Å', 'recon RMSD'},'Units','normalized', ...
     'FontName','Arial','FontSize',20,'Color',[0.4 0.4 0.4],'VerticalAlignment','top');
axtoolbar(axA,{}); axtoolbar(axB,{});   % keep interactive toolbar out of the export

title(tl,'Spline order sensitivity — cubic is near-optimal (Marco Q5)', ...
      'FontName','Arial','FontSize',26,'FontWeight','bold');
out = fullfile(here,'figures_matlab','marco_spline_order');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s\n', out);
