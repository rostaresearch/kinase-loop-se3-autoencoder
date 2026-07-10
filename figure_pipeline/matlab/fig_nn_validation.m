% Latent-neighbour validation: for each probe chain, median structural RMSD to
% its k latent nearest-neighbours vs to random chains. Points below the y=x line
% mean latent neighbours are structurally closer than random -> the latent is
% structurally meaningful.
here = fileparts(mfilename('fullpath')); base = fileparts(here);
T = readtable(fullfile(base,'extended','nn_validation.csv'));
xnn = max(T.median_rmsd_latent_neighbours, 1e-2);
xrnd = max(T.median_rmsd_random_chains, 1e-2);
frac = mean(xnn < xrnd)*100;

fig = figure('Color','w','Units','inches','Position',[1 1 11 9]);
ax = axes(fig); hold(ax,'on');
lim = [max(0.3, min([xnn;xrnd])*0.8), max([xnn;xrnd])*1.25];   % robust floor
plot(ax, lim, lim, '--', 'Color',[0.3 0.3 0.3], 'LineWidth',1.8,'HandleVisibility','off');
scatter(ax, xnn, xrnd, 60, [0.192 0.373 0.557], 'filled', 'MarkerFaceAlpha',0.45);
set(ax,'XScale','log','YScale','log'); xlim(ax,lim); ylim(ax,lim);
xlabel(ax,'RMSD to latent neighbours (Å)');
ylabel(ax,'RMSD to random chains (Å)');
title(ax,'Latent neighbours are structurally closer');
pubstyle(ax); titlegap(ax); axis(ax,'square'); axtoolbar(ax,{});
text(ax,0.05,0.95,sprintf('%.0f%% closer to latent\nneighbours than random\n(above y=x, n=%d)',frac,height(T)), ...
     'Units','normalized','FontName','Arial','FontSize',22,'VerticalAlignment','top');

out = fullfile(here,'figures_matlab','nn_validation');
exportgraphics(fig,[out '.png'],'Resolution',600);
exportgraphics(fig,[out '.pdf'],'ContentType','vector');
fprintf('wrote %s (%.0f%% below y=x)\n', out, frac);
