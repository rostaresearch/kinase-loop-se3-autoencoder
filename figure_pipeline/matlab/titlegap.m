function titlegap(ax)
% Float the axes title into a clear whitespace band above the plot box so
% the title can be cropped out cleanly for publication. Single-panel only.
if nargin < 1, ax = gca; end
ax.Units = 'normalized';
p = ax.Position;
% Lift the axes bottom so the large (24-pt) x-label + tick labels are not
% clipped at the figure edge; shrink the height so the top of the plot box
% stays put (same croppable whitespace band for the floated title).
lift = 0.08;
ax.Position = [p(1) p(2)+lift p(3) p(4)*0.88 - lift];
ax.Title.Units = 'normalized';
ax.Title.VerticalAlignment = 'bottom';
ax.Title.Position(2) = 1.12;     % well above the box top -> croppable gap
end
