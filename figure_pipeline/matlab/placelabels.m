function placelabels(ax, x, y, labels, fsize)
% Place point labels next to (x,y) and de-overlap them by iterative
% vertical nudging, with a thin leader line from each point to its
% (moved) label. Side-aware: points in the right third get left-placed
% labels. Handles linear or log axes. Never uses text below 22 pt.
if nargin < 5, fsize = 22; end
n = numel(labels);
if n == 0, return; end
x = x(:); y = y(:);

xl = xlim(ax); yl = ylim(ax);
isxlog = strcmp(ax.XScale,'log'); isylog = strcmp(ax.YScale,'log');
% data -> normalized [0,1] within axes
    function u = toN(v, lo, hi, islog)
        if islog, u = (log10(v)-log10(lo))./(log10(hi)-log10(lo));
        else,     u = (v-lo)./(hi-lo); end
    end
% normalized -> data
    function v = fromN(u, lo, hi, islog)
        if islog, v = 10.^(log10(lo)+u.*(log10(hi)-log10(lo)));
        else,     v = lo + u.*(hi-lo); end
    end

px = toN(x, xl(1), xl(2), isxlog);
py = toN(y, yl(1), yl(2), isylog);
rightside = px > 0.62;
lx = px + 0.015;  lx(rightside) = px(rightside) - 0.015;
ly = py;

axpos = getpixelposition(ax);
hh = (fsize*1.4) / max(axpos(4),1);     % ~line height in norm units
minsep = 1.05*hh;

[~,ord] = sort(ly); lys = ly(ord);
for it = 1:300
    moved = false;
    for k = 2:n
        if lys(k)-lys(k-1) < minsep
            s = (minsep-(lys(k)-lys(k-1)))/2 + 1e-4;
            lys(k-1) = lys(k-1)-s; lys(k) = lys(k)+s; moved = true;
        end
    end
    lys = min(max(lys,0.03),0.97);
    if ~moved, break; end
end
ly(ord) = lys;

hold(ax,'on');
for k = 1:n
    xd = fromN(lx(k), xl(1), xl(2), isxlog);
    yd = fromN(ly(k), yl(1), yl(2), isylog);
    if rightside(k), ha = 'right'; else, ha = 'left'; end
    plot(ax, [x(k) xd], [y(k) yd], '-', 'Color',[0.55 0.55 0.55], ...
         'LineWidth',0.8, 'HandleVisibility','off');
    text(ax, xd, yd, char(labels(k)), 'FontName','Arial','FontSize',fsize, ...
         'HorizontalAlignment',ha, 'VerticalAlignment','middle', ...
         'Interpreter','none', 'Clipping','off');
end
end
