function pubstyle(ax)
% Apply the lab publication style to an axes:
%  - all text >= 22 pt (sans-serif), no top/right spine, no grid,
%    outward ticks, axis line width 1.8.
% Use titlegap(ax) afterwards on SINGLE-panel figures to float the title
% into a croppable whitespace band.
if nargin < 1, ax = gca; end
set(ax, 'FontName','Arial', 'FontSize',24, 'LineWidth',1.8, ...
        'Box','off', 'TickDir','out', 'TickLength',[0.018 0.018], ...
        'XGrid','off','YGrid','off', 'Color','white', ...
        'XMinorTick','off','YMinorTick','off');
ax.XAxis.FontSize = 24; ax.YAxis.FontSize = 24;
ax.XLabel.FontSize = 24; ax.YLabel.FontSize = 24;
ax.Title.FontSize  = 26; ax.Title.FontWeight = 'bold';
end
