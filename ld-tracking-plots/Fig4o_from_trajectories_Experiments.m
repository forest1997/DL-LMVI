%% Reproduce Fig. 4o directly from deposited trajectory coordinates
% The script uses 2D centroid trajectories. One experiment is one paired point.
% It reproduces the current figure's legacy MSD-based inclusion rule.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
dataFile = fullfile(scriptDir, 'Fig4o_Reproducibility_Experiments.xlsx');
assert(isfile(dataFile), 'Missing trajectory table: %s', dataFile);

pixelSizeUm = 0.2;
frameIntervalS = 0.48;
msdMaxLag = 20;
minMsdLags = 5;

T = readtable(dataFile, 'Sheet', 'Trajectory Points', ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
required = ["ExperimentID","ParticleID","Frame","X_pixel","Y_pixel","Region"];
assert(all(ismember(required, string(T.Properties.VariableNames))), ...
    'Trajectory table is missing one or more required columns.');
T.ExperimentID = string(T.ExperimentID);

keys = unique(T(:, {'ExperimentID','ParticleID'}), 'rows', 'stable');
nTracks = height(keys);
trackRows = cell(nTracks, 1);

for k = 1:nTracks
    experimentID = keys.ExperimentID(k);
    particleID = keys.ParticleID(k);
    tr = T(T.ExperimentID == experimentID & T.ParticleID == particleID, :);
    tr = sortrows(tr, 'Frame');

    x = double(tr.X_pixel);
    y = double(tr.Y_pixel);
    frames = double(tr.Frame);
    dx = diff(x);
    dy = diff(y);
    stepPx = hypot(dx, dy);
    frameGaps = diff(frames);
    if all(frameGaps == frameGaps(1))
        dtFrames = frameGaps(1);
    else
        dtFrames = mean(frameGaps);
    end
    meanSpeedPxPerFrame = mean(stepPx) / dtFrames;

    maxLag = min(msdMaxLag, floor(numel(x) / 2));
    msd = nan(maxLag, 1);
    for lag = 1:maxLag
        msd(lag) = mean((x(1+lag:end)-x(1:end-lag)).^2 + ...
                        (y(1+lag:end)-y(1:end-lag)).^2);
    end

    include = true;
    exclusionReason = "";
    monotonicRatio = NaN;
    if numel(msd) >= minMsdLags
        laterMsd = msd(2:end);
        if any(laterMsd <= 0)
            include = false;
            exclusionReason = "Non-positive MSD in fitted lag range";
        else
            monotonicRatio = mean(diff(laterMsd) >= 0);
            if monotonicRatio < 0.3
                include = false;
                exclusionReason = "MSD monotonicity ratio below 0.3";
            end
        end
    end

    trackRows{k} = table(experimentID, particleID, string(tr.Region(1)), numel(x), ...
        frames(1), frames(end), mean(stepPx), meanSpeedPxPerFrame, ...
        meanSpeedPxPerFrame * pixelSizeUm / frameIntervalS, ...
        sum(stepPx), hypot(x(end)-x(1), y(end)-y(1)), ...
        monotonicRatio, include, exclusionReason, ...
        'VariableNames', {'ExperimentID','ParticleID','Region','TrackLength_frames', ...
        'StartFrame','EndFrame','MeanStep_px','MeanSpeed_px_per_frame', ...
        'MeanSpeed_um_per_s','TotalPathLength_px','NetDisplacement_px', ...
        'MSDMonotonicityRatio','IncludedInCurrentFig4o','ExclusionReason'});
end

trackSummary = vertcat(trackRows{:});
included = trackSummary(trackSummary.IncludedInCurrentFig4o, :);
experimentIDs = unique(included.ExperimentID, 'stable');
experimentRows = cell(numel(experimentIDs), 1);

for k = 1:numel(experimentIDs)
    experimentID = experimentIDs(k);
    current = included(included.ExperimentID == experimentID, :);
    nuc = current.MeanSpeed_um_per_s(current.Region == "Nucleus-proximal");
    mem = current.MeanSpeed_um_per_s(current.Region == "Membrane-proximal");
    assert(~isempty(nuc) && ~isempty(mem), 'Experiment %s lacks one spatial category.', char(experimentID));
    nucMean = mean(nuc);
    memMean = mean(mem);
    experimentRows{k} = table(experimentID, nucMean, memMean, nucMean-memMean, numel(nuc), numel(mem), ...
        'VariableNames', {'ExperimentID','NucleusProximal_ExperimentMean_um_per_s', ...
        'MembraneProximal_ExperimentMean_um_per_s','Difference_NucleusMinusMembrane_um_per_s', ...
        'NucleusProximal_TrajectoryCount','MembraneProximal_TrajectoryCount'});
end

experimentMeans = vertcat(experimentRows{:});
delta = experimentMeans.Difference_NucleusMinusMembrane_um_per_s;
pValue = exactSignedRankTwoSided(delta);

writetable(trackSummary, fullfile(scriptDir, 'Fig4o_TrajectorySummary_recomputed.csv'));
writetable(experimentMeans, fullfile(scriptDir, 'Fig4o_ExperimentMeans_recomputed.csv'));

fprintf('All linked trajectories: %d\n', height(trackSummary));
fprintf('Trajectories included in current Fig. 4o: %d\n', height(included));
fprintf('Excluded by legacy MSD rule: %d\n', sum(~trackSummary.IncludedInCurrentFig4o));
fprintf('Paired experiments: %d\n', height(experimentMeans));
fprintf('Nucleus-proximal median experiment mean: %.6f um/s\n', ...
    median(experimentMeans.NucleusProximal_ExperimentMean_um_per_s));
fprintf('Membrane-proximal median experiment mean: %.6f um/s\n', ...
    median(experimentMeans.MembraneProximal_ExperimentMean_um_per_s));
fprintf('Two-sided exact Wilcoxon signed-rank p = %.4f\n', pValue);

plotFig4o(experimentMeans, pValue, scriptDir);

function p = exactSignedRankTwoSided(d)
    d = d(isfinite(d) & d ~= 0);
    absD = abs(d(:));
    [sortedD, order] = sort(absD);
    ranksSorted = zeros(size(sortedD));
    i = 1;
    while i <= numel(sortedD)
        j = i;
        while j < numel(sortedD) && sortedD(j+1) == sortedD(i)
            j = j + 1;
        end
        ranksSorted(i:j) = mean(i:j);
        i = j + 1;
    end
    ranks = zeros(size(ranksSorted));
    ranks(order) = ranksSorted;
    observed = min(sum(ranks(d > 0)), sum(ranks(d < 0)));
    totalRank = sum(ranks);
    allW = zeros(2^numel(d), 1);
    for mask = 0:(2^numel(d)-1)
        chosen = logical(bitget(mask, 1:numel(d)));
        wPlus = sum(ranks(chosen));
        allW(mask+1) = min(wPlus, totalRank-wPlus);
    end
    p = mean(allW <= observed + 1e-12);
end

function plotFig4o(C, pValue, outDir)
    nuc = C.NucleusProximal_ExperimentMean_um_per_s;
    mem = C.MembraneProximal_ExperimentMean_um_per_s;
    colNuc = [44 127 184] / 255;
    colMem = [39 174 96] / 255;

    fig = figure('Color','k','Position',[220 120 520 600]);
    ax = axes(fig,'Color','k'); hold(ax,'on');
    ax.Position = [0.24 0.31 0.70 0.61];
    set(ax,'XColor','w','YColor','w','FontName','Arial','FontSize',14, ...
        'TickDir','out','Box','off','LineWidth',1.35);
    grid(ax,'on'); ax.GridColor = [0.75 0.75 0.75]; ax.GridAlpha = 0.08;
    ax.GridLineStyle = '--';

    x1 = ones(size(nuc));
    x2 = 2 * ones(size(mem));
    for i = 1:numel(nuc)
        plot(ax,[x1(i) x2(i)],[nuc(i) mem(i)],'-','Color',[0.55 0.55 0.55], ...
            'LineWidth',1.1);
    end
    drawBox(ax,1,nuc,colNuc);
    drawBox(ax,2,mem,colMem);
    scatter(ax,x1,nuc,38,colNuc,'filled','MarkerFaceAlpha',0.78);
    scatter(ax,x2,mem,38,colMem,'filled','MarkerFaceAlpha',0.78);

    set(ax,'XTick',[1 2],'XTickLabel',{'Nucleus-proximal','Membrane-proximal'});
    xtickangle(ax,45); xlim(ax,[0.55 2.45]);
    ylabel(ax,'Average speed (\mum/s)','Color','w','FontSize',16);
    values = [nuc; mem]; span = max(values)-min(values);
    ylim(ax,[max(0,min(values)-0.18*span), max(values)+0.55*span]);
    yl = ylim(ax); yRange = diff(yl);
    y1 = max(values) + 0.10*yRange; y2 = y1 + 0.04*yRange;
    plot(ax,[1 1 2 2],[y1 y2 y2 y1],'w-','LineWidth',1.1);
    text(ax,1.5,y2+0.016*yRange,sprintf('p = %.4f',pValue), ...
        'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'Color','w','FontSize',13,'FontWeight','bold');

    exportgraphics(fig,fullfile(outDir,'Fig4o_reproduced.png'), ...
        'Resolution',300,'BackgroundColor','black');
    exportgraphics(fig,fullfile(outDir,'Fig4o_reproduced.pdf'), ...
        'ContentType','vector','BackgroundColor','black');
    savefig(fig,fullfile(outDir,'Fig4o_reproduced.fig'));
end

function drawBox(ax,x0,data,col)
    q = prctile(data,[25 50 75]); q1=q(1); med=q(2); q3=q(3);
    iqrValue = q3-q1;
    low = min(data(data >= q1-1.5*iqrValue));
    high = max(data(data <= q3+1.5*iqrValue));
    halfWidth = 0.15;
    line(ax,[x0 x0],[q3 high],'Color',col,'LineWidth',1.3);
    line(ax,[x0 x0],[q1 low],'Color',col,'LineWidth',1.3);
    line(ax,[x0-0.09 x0+0.09],[high high],'Color',col,'LineWidth',1.3);
    line(ax,[x0-0.09 x0+0.09],[low low],'Color',col,'LineWidth',1.3);
    patch(ax,[x0-halfWidth x0+halfWidth x0+halfWidth x0-halfWidth], ...
        [q1 q1 q3 q3],col,'FaceColor','none','EdgeColor',col,'LineWidth',1.5);
    plot(ax,[x0-halfWidth x0+halfWidth],[med med],'-','Color',col,'LineWidth',1.8);
end
