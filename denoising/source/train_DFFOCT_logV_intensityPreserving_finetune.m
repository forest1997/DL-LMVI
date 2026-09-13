%% train_DFFOCT_logV_intensityPreserving_finetune.m
% Quantitative-map training with frame-count-normalized dynamic intensity:
%   short [H_Hz, log10(V_power)] -> long [H_Hz, log10(V_power)]
% V_power is already frame-count normalized; do not divide by N^2 again.
%
% RGB is not used for training or quantitative evaluation in this pipeline.

clear; clc; close all;

%% ===================== 1) Paths ==========================================
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir);
map48Dir = fullfile(projectRoot, 'data', 'quant_maps', 'short_acquisition');
map512Dir = fullfile(projectRoot, 'data', 'quant_maps', 'long_reference');
holdoutCsv = fullfile(projectRoot, 'config', 'selected_35_images.csv');

pretrainedModel = fullfile(projectRoot, 'output', 'training', 'DFFOCT_resUNet_TimeNormalizedV_dlnet.mat');
outModel = fullfile(projectRoot, 'output', 'training', 'DFFOCT_resUNet_TimeNormalizedV_IntensityPreserving_dlnet.mat');

%% ===================== 2) Training configuration ==========================
cfg.patchSize = [256 256];
cfg.overlapInfer = 96;
cfg.batchSize = 6;
cfg.epochs = 16;
cfg.stepsPerEpoch = 60;
cfg.valEverySteps = 60;
cfg.earlyStopPatience = 6;
cfg.earlyStopMinDelta = 1e-4;
cfg.preloadData = true;
cfg.selectionHScaleHz = 10;
cfg.selectionBiasWeight = 0.5;
cfg.selectionVMeanWeight = 1.0;

cfg.lr0 = 3e-5;
cfg.lrDropPeriod = 8;
cfg.lrDropFactor = 0.5;
cfg.beta1 = 0.9;
cfg.beta2 = 0.999;

% Loss weights in normalized quantitative-map space.
cfg.wH = 1.0;
cfg.wLogV = 1.0;
cfg.wGrad = 0.03;
cfg.wVSymRel = 0.25;
cfg.wVMean = 0.80;
cfg.charbEps = 1e-3;
cfg.gradClip = 1.0;

% H is a physical frequency. Keep this fixed across train/inference.
% If samplingRateHz is not 100 Hz, change the upper bound accordingly.
cfg.hNormRangeHz = [0 50];

% Dynamic intensity uses log transform and global train-set statistics.
cfg.vEps = 1e-12;
cfg.statsMaxPixelsPerFile = 120000;
cfg.validMaskPercentile = 20;    % fixed from train targets, used for H/V metrics

% Patch sampling.
cfg.minValidFraction = 0.05;
cfg.maxTries = 30;

% Augmentation. These preserve H/V values and spatial correspondence.
cfg.augFlipLR = true;
cfg.augFlipUD = true;
cfg.augRot90 = true;

cfg.useGPU = canUseGPU();

% Training-time RGB previews are display-only. Metrics still use H_Hz/V_power.
cfg.showPreview = true;
cfg.savePreview = true;
cfg.previewDir = fullfile(projectRoot, 'output', 'training', 'preview_finetune');

if cfg.savePreview
    ensureDir(cfg.previewDir);
end
ensureDir(fileparts(outModel));

%% ===================== 3) Build paired list ===============================
[inputs, targets] = buildPairsQuant(map48Dir, map512Dir);
[inputs, targets, holdoutX, holdoutY] = excludeHeldOutPairs( ...
    inputs, targets, holdoutCsv);
n = numel(inputs);
fprintf("Found %d development pairs after reserving %d held-out test pairs.\n", ...
    n, numel(holdoutX));
assert(n >= 8, "Too few paired samples. Check map folders and naming.");

%% ===================== 4) Train/validation split ==========================
rng(0);
idx = randperm(n);
nTrain = floor(0.85 * n);
trainIdx = idx(1:nTrain);
valIdx = idx(nTrain+1:end);

trainX = inputs(trainIdx); trainY = targets(trainIdx);
valX = inputs(valIdx);     valY = targets(valIdx);

fprintf("Train=%d, Val=%d\n", numel(trainX), numel(valX));

%% ===================== 5) Optional in-memory cache =======================
if cfg.preloadData
    fprintf("Preloading quantitative maps into memory...\n");
    trainDataX = preloadQuantList(trainX);
    trainDataY = preloadQuantList(trainY);
    valDataX = preloadQuantList(valX);
    valDataY = preloadQuantList(valY);
else
    trainDataX = trainX;
    trainDataY = trainY;
    valDataX = valX;
    valDataY = valY;
end

%% ===================== 6) Fixed normalization from train set ==============
normStats = computeQuantNormStats(trainDataX, trainDataY, cfg);
fprintf("Normalization:\n");
fprintf("  H range: [%.4g %.4g] Hz\n", normStats.hMinHz, normStats.hMaxHz);
fprintf("  log10(V) mean/std: %.4g / %.4g\n", normStats.logVMean, normStats.logVStd);
fprintf("  valid log10(V) threshold: %.4g\n", normStats.validLogVMin);

%% ===================== 7) Load the validated primary checkpoint ==========
assert(isfile(pretrainedModel), "Pretrained checkpoint not found: %s", pretrainedModel);
checkpoint = load(pretrainedModel, "dlnet");
dlnet = checkpoint.dlnet;

if cfg.useGPU
    dlnet = dlupdate(@gpuArray, dlnet);
end

%% ===================== 8) Adam state =====================================
trailingAvg = [];
trailingAvgSq = [];
globalStep = 0;
checksWithoutImprovement = 0;
stopTraining = false;
valHistory = table();

% Register the original validated checkpoint as the baseline so fine-tuning
% can never replace it with a quantitatively worse model.
baselineStats = validateQuantSet(dlnet, valDataX, valDataY, normStats, cfg, 0);
bestVal = baselineStats.loss;
bestSelectionScore = baselineStats.hMAEMeanHz / cfg.selectionHScaleHz + ...
    baselineStats.vRelMAEMean + cfg.selectionBiasWeight * ...
    (abs(baselineStats.hBiasMeanHz) / cfg.selectionHScaleHz + ...
    abs(baselineStats.vRelBiasMean)) + cfg.selectionVMeanWeight * ...
    abs(log(max(baselineStats.vMeanRatio, 1e-6)));
baselineRow = table(0, 0, 0, NaN, baselineStats.loss, bestSelectionScore, ...
    baselineStats.hBiasMeanHz, baselineStats.hMAEMeanHz, ...
    100*baselineStats.vRelBiasMean, 100*baselineStats.vRelMAEMean, ...
    100*(baselineStats.vMeanRatio-1), 'VariableNames', ...
    {'step','epoch','learningRate','trainLoss','valLoss','selectionScore', ...
    'H_bias_Hz','H_MAE_Hz','V_rel_bias_pct','V_rel_MAE_pct','V_mean_bias_pct'});
valHistory = [valHistory; baselineRow];
dlnetOnDevice = dlnet;
dlnet = dlupdate(@gather, dlnet);
save(outModel, "dlnet", "cfg", "normStats", "pretrainedModel", ...
    "map48Dir", "map512Dir", "trainX", "trainY", "valX", "valY", ...
    "holdoutX", "holdoutY", "bestVal", "bestSelectionScore", ...
    "valHistory", '-v7.3');
dlnet = dlnetOnDevice;
writetable(valHistory, fullfile(fileparts(outModel), ...
    'training_history_logV_intensityPreserving.csv'));
fprintf("Baseline score=%.4g Hbias=%.4gHz VrelBias=%.3f%% VrelMAE=%.3f%% VmeanBias=%.3f%%\n", ...
    bestSelectionScore, baselineStats.hBiasMeanHz, ...
    100*baselineStats.vRelBiasMean, 100*baselineStats.vRelMAEMean, ...
    100*(baselineStats.vMeanRatio-1));

%% ===================== 9) Training loop ==================================
for epoch = 1:cfg.epochs
    lr = cfg.lr0 * cfg.lrDropFactor^(floor((epoch-1) / cfg.lrDropPeriod));

    tic;
    runningLoss = 0;

    for step = 1:cfg.stepsPerEpoch
        globalStep = globalStep + 1;

        [Xbatch, Ybatch, Mbatch] = makeMiniBatchQuant(trainDataX, trainDataY, normStats, cfg);
        dlX = dlarray(Xbatch, "SSCB");
        dlY = dlarray(Ybatch, "SSCB");
        dlM = dlarray(Mbatch, "SSCB");

        if cfg.useGPU
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
            dlM = gpuArray(dlM);
        end

        [loss, grads] = dlfeval(@modelGradientsQuant, dlnet, dlX, dlY, dlM, cfg, normStats);
        grads = dlupdate(@(g) clipGrad(g, cfg.gradClip), grads);

        [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, grads, ...
            trailingAvg, trailingAvgSq, globalStep, lr, cfg.beta1, cfg.beta2);

        runningLoss = runningLoss + double(gather(extractdata(loss)));

        if mod(globalStep, cfg.valEverySteps) == 0 && ~isempty(valDataX)
            valStats = validateQuantSet(dlnet, valDataX, valDataY, normStats, cfg, globalStep);
            selectionScore = valStats.hMAEMeanHz / cfg.selectionHScaleHz + ...
                valStats.vRelMAEMean + cfg.selectionBiasWeight * ...
                (abs(valStats.hBiasMeanHz) / cfg.selectionHScaleHz + ...
                abs(valStats.vRelBiasMean)) + cfg.selectionVMeanWeight * ...
                abs(log(max(valStats.vMeanRatio, 1e-6)));
            fprintf("[E%03d S%05d] lr=%.2e trainLoss=%.4g valLoss=%.4g score=%.4g Hbias=%.4gHz VrelBias=%.3f%% VrelMAE=%.3f%% VmeanBias=%.3f%%\n", ...
                epoch, globalStep, lr, runningLoss / step, valStats.loss, ...
                selectionScore, valStats.hBiasMeanHz, 100*valStats.vRelBiasMean, ...
                100*valStats.vRelMAEMean, 100*(valStats.vMeanRatio-1));

            row = table(globalStep, epoch, lr, runningLoss / step, ...
                valStats.loss, selectionScore, valStats.hBiasMeanHz, valStats.hMAEMeanHz, ...
                100*valStats.vRelBiasMean, 100*valStats.vRelMAEMean, ...
                100*(valStats.vMeanRatio-1), ...
                'VariableNames', {'step','epoch','learningRate','trainLoss', ...
                'valLoss','selectionScore','H_bias_Hz','H_MAE_Hz', ...
                'V_rel_bias_pct','V_rel_MAE_pct','V_mean_bias_pct'});
            valHistory = [valHistory; row]; %#ok<AGROW>
            writetable(valHistory, fullfile(fileparts(outModel), ...
                'training_history_logV_intensityPreserving.csv'));

            if selectionScore < bestSelectionScore - cfg.earlyStopMinDelta
                bestVal = valStats.loss;
                bestSelectionScore = selectionScore;
                checksWithoutImprovement = 0;
                dlnetOnDevice = dlnet;
                dlnet = dlupdate(@gather, dlnet);
                save(outModel, "dlnet", "cfg", "normStats", ...
                    "map48Dir", "map512Dir", "trainX", "trainY", ...
                    "valX", "valY", "holdoutX", "holdoutY", ...
                    "bestVal", "bestSelectionScore", "valHistory", '-v7.3');
                dlnet = dlnetOnDevice;
                fprintf("  Saved BEST quantitative model (score=%.4g, val=%.4g) -> %s\n", ...
                    bestSelectionScore, bestVal, outModel);
            else
                checksWithoutImprovement = checksWithoutImprovement + 1;
                fprintf("  No validation improvement: %d/%d\n", ...
                    checksWithoutImprovement, cfg.earlyStopPatience);
                if checksWithoutImprovement >= cfg.earlyStopPatience
                    stopTraining = true;
                    break;
                end
            end
        end
    end

    fprintf("Epoch %d/%d done. avgLoss=%.4g time=%.1fs\n", ...
        epoch, cfg.epochs, runningLoss / cfg.stepsPerEpoch, toc);
    if stopTraining
        fprintf("Early stopping at epoch %d.\n", epoch);
        break;
    end
end

fprintf("Intensity-preserving fine-tuning finished. Best score=%.4g, associated val loss=%.4g\n", ...
    bestSelectionScore, bestVal);

if isfile(outModel)
    bestCheckpoint = load(outModel, 'dlnet');
    dlnet = bestCheckpoint.dlnet;
    if cfg.useGPU
        dlnet = dlupdate(@gpuArray, dlnet);
    end
end
if ~isempty(valDataX)
    valStats = validateQuantSet(dlnet, valDataX, valDataY, normStats, cfg, globalStep);
    fprintf("Best-checkpoint validation: valLoss=%.4g Hbias=%.4gHz VrelBias=%.3f%% VrelMAE=%.3f%% VmeanBias=%.3f%%\n", ...
        valStats.loss, valStats.hBiasMeanHz, 100*valStats.vRelBiasMean, ...
        100*valStats.vRelMAEMean, 100*(valStats.vMeanRatio-1));
end

%% ========================================================================
%% Local functions
%% ========================================================================

function tf = canUseGPU()
    try
        tf = (gpuDeviceCount > 0);
    catch
        tf = false;
    end
end

function ensureDir(folderName)
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
end

function g = clipGrad(g, thr)
    if isempty(g) || ~isnumeric(g)
        return;
    end
    g = max(min(g, thr), -thr);
end

function [inputs, targets] = buildPairsQuant(map48Dir, map512Dir)
    d = dir(fullfile(map48Dir, "*.mat"));
    names = string({d.name});
    names = names(~[d.isdir]);
    names = names(contains(names, "Range48", "IgnoreCase", true));

    inputs = strings(1, numel(names));
    targets = strings(1, numel(names));
    matchCount = 0;

    fprintf("Scanning %d candidate MAT files in Range48 folder...\n", numel(names));

    for i = 1:numel(names)
        n48 = names(i);
        n512 = replace(n48, "_Range48_quant", "_Range512_quant");
        n512 = replace(n512, "_range48_quant", "_range512_quant");
        if n512 == n48
            n512 = replace(n512, "Range48", "Range512");
        end

        f48 = fullfile(map48Dir, n48);
        f512 = fullfile(map512Dir, n512);

        if isfile(f512)
            matchCount = matchCount + 1;
            inputs(matchCount) = string(f48);
            targets(matchCount) = string(f512);
        else
            fprintf("No match for:\n  %s\nExpected:\n  %s\n", n48, n512);
        end
    end

    inputs = inputs(1:matchCount);
    targets = targets(1:matchCount);

    fprintf("Matched %d pairs.\n", numel(inputs));
end

function [devX, devY, holdoutX, holdoutY] = excludeHeldOutPairs(inputs, targets, holdoutCsv)
    lines = readlines(holdoutCsv);
    lines = lines(strlength(strtrim(lines)) > 0);
    assert(numel(lines) >= 2, "Held-out CSV is empty: %s", holdoutCsv);
    header = split(strtrim(lines(1)), ",");
    baseCol = find(strcmpi(header, "baseName"), 1, "first");
    assert(~isempty(baseCol), "baseName column not found: %s", holdoutCsv);

    holdoutNames = strings(numel(lines)-1, 1);
    for i = 2:numel(lines)
        parts = split(strtrim(lines(i)), ",");
        holdoutNames(i-1) = parts(baseCol);
    end

    inputBase = strings(numel(inputs), 1);
    for i = 1:numel(inputs)
        [~, name] = fileparts(inputs(i));
        inputBase(i) = erase(string(name), "_Range48_quant");
        inputBase(i) = erase(inputBase(i), "_range48_quant");
    end
    isHoldout = ismember(inputBase, holdoutNames);
    missing = holdoutNames(~ismember(holdoutNames, inputBase));
    assert(isempty(missing), "Held-out image is unavailable: %s", strjoin(missing, ", "));

    holdoutX = inputs(isHoldout);
    holdoutY = targets(isHoldout);
    devX = inputs(~isHoldout);
    devY = targets(~isHoldout);
end

function normStats = computeQuantNormStats(trainX, trainY, cfg)
    rng(1);
    logVUnion = [];
    logVTarget = [];

    for i = 1:numel(trainX)
        mx = getQuantMap(trainX, i);
        my = getQuantMap(trainY, i);

        logVx = sampleLogV(mx.V_power, cfg.vEps, cfg.statsMaxPixelsPerFile);
        logVy = sampleLogV(my.V_power, cfg.vEps, cfg.statsMaxPixelsPerFile);

        logVUnion = [logVUnion; logVx; logVy]; %#ok<AGROW>
        logVTarget = [logVTarget; logVy]; %#ok<AGROW>
    end

    logVUnion = logVUnion(isfinite(logVUnion));
    logVTarget = logVTarget(isfinite(logVTarget));
    assert(~isempty(logVUnion), "No finite V values found for normalization.");

    normStats = struct();
    normStats.hMinHz = cfg.hNormRangeHz(1);
    normStats.hMaxHz = cfg.hNormRangeHz(2);
    normStats.logVMean = mean(logVUnion);
    normStats.logVStd = std(logVUnion);
    if normStats.logVStd < 1e-6 || ~isfinite(normStats.logVStd)
        normStats.logVStd = 1;
    end
    normStats.validLogVMin = prctile(logVTarget, cfg.validMaskPercentile);
    normStats.vEps = cfg.vEps;
    normStats.vTransform = "log10";
    normStats.vPowerNormalization = "frameCountSquared";
    normStats.version = "quant-time-normalized-log-v-norm-v2";
end

function s = sampleLogV(V, vEps, maxPixels)
    V = double(V);
    logV = log10(max(V(:), vEps));
    logV = logV(isfinite(logV));
    if numel(logV) > maxPixels
        idx = randperm(numel(logV), maxPixels);
        logV = logV(idx);
    end
    s = logV(:);
end

function maps = loadQuantMaps(filename)
    S = load(filename, 'maps');
    if ~isfield(S, 'maps')
        error('MAT file does not contain a maps struct: %s', filename);
    end
    maps = S.maps;
    expectedFrames = 512;
    if contains(string(filename), 'Range48', 'IgnoreCase', true)
        expectedFrames = 48;
    end
    validateDFFOCTInputMaps(maps, expectedFrames);
    maps.H_Hz = single(maps.H_Hz);
    maps.V_power = single(maps.V_power);
    if isfield(maps, 'S_invHz')
        maps.S_invHz = single(maps.S_invHz);
    else
        maps.S_invHz = [];
    end
    if isfield(maps, 'validMask')
        maps.validMask = logical(maps.validMask);
    else
        maps.validMask = isfinite(maps.H_Hz) & isfinite(maps.V_power) & maps.V_power > 0;
    end
end

function data = preloadQuantList(files)
    data = cell(size(files));
    for i = 1:numel(files)
        data{i} = loadQuantMaps(files(i));
    end
end

function maps = getQuantMap(list, index)
    if iscell(list)
        maps = list{index};
    else
        maps = loadQuantMaps(list(index));
    end
end

function [Xbatch, Ybatch, Mbatch] = makeMiniBatchQuant(listX, listY, normStats, cfg)
    ps = cfg.patchSize;
    B = cfg.batchSize;

    Xbatch = zeros(ps(1), ps(2), 2, B, "single");
    Ybatch = zeros(ps(1), ps(2), 2, B, "single");
    Mbatch = zeros(ps(1), ps(2), 1, B, "single");

    n = numel(listX);

    for b = 1:B
        id = randi(n);
        mx = getQuantMap(listX, id);
        my = getQuantMap(listY, id);

        [X, Y, M] = makeNormalizedPair(mx, my, normStats);
        [px, py, pm] = samplePatchQuant(X, Y, M, ps, cfg);
        [px, py, pm] = augmentTriplet(px, py, pm, cfg);

        Xbatch(:,:,:,b) = px;
        Ybatch(:,:,:,b) = py;
        Mbatch(:,:,:,b) = pm;
    end
end

function [X, Y, M] = makeNormalizedPair(mx, my, normStats)
    [mx, my] = cropToCommonSize(mx, my);
    X = normalizeQuantMaps(mx, normStats);
    Y = normalizeQuantMaps(my, normStats);
    M = makeTargetMask(my, normStats);
end

function [mx, my] = cropToCommonSize(mx, my)
    H = min(size(mx.H_Hz, 1), size(my.H_Hz, 1));
    W = min(size(mx.H_Hz, 2), size(my.H_Hz, 2));

    mx.H_Hz = mx.H_Hz(1:H, 1:W);
    mx.V_power = mx.V_power(1:H, 1:W);
    mx.validMask = mx.validMask(1:H, 1:W);

    my.H_Hz = my.H_Hz(1:H, 1:W);
    my.V_power = my.V_power(1:H, 1:W);
    my.validMask = my.validMask(1:H, 1:W);
end

function X = normalizeQuantMaps(maps, normStats)
    H = double(maps.H_Hz);
    V = double(maps.V_power);

    Hn = ((H - normStats.hMinHz) ./ ...
        (normStats.hMaxHz - normStats.hMinHz + eps)) * 2 - 1;
    logV = log10(max(V, normStats.vEps));
    logVn = (logV - normStats.logVMean) ./ normStats.logVStd;

    Hn(~isfinite(Hn)) = 0;
    logVn(~isfinite(logVn)) = 0;

    X = single(cat(3, Hn, logVn));
end

function M = makeTargetMask(maps, normStats)
    H = double(maps.H_Hz);
    V = double(maps.V_power);
    logV = log10(max(V, normStats.vEps));

    M = maps.validMask & isfinite(H) & isfinite(V) & ...
        logV >= normStats.validLogVMin;
    M = single(M);
end

function [px, py, pm] = samplePatchQuant(X, Y, M, ps, cfg)
    [H, W, ~] = size(X);
    ph = ps(1);
    pw = ps(2);

    padH = max(0, ph - H);
    padW = max(0, pw - W);
    if padH > 0 || padW > 0
        X = padarray(X, [padH padW], "symmetric", "post");
        Y = padarray(Y, [padH padW], "symmetric", "post");
        M = padarray(M, [padH padW], 0, "post");
        [H, W, ~] = size(X);
    end

    bestScore = -inf;
    bestRC = [1 1];

    for t = 1:cfg.maxTries
        r = randi(H - ph + 1);
        c = randi(W - pw + 1);
        patchM = M(r:r+ph-1, c:c+pw-1, :);
        validFrac = mean(patchM(:));
        patchY = Y(r:r+ph-1, c:c+pw-1, 2);
        textureVar = var(patchY(:));
        score = validFrac + 0.05 * textureVar;

        if score > bestScore
            bestScore = score;
            bestRC = [r c];
        end
        if validFrac >= cfg.minValidFraction
            break;
        end
    end

    r = bestRC(1);
    c = bestRC(2);
    px = X(r:r+ph-1, c:c+pw-1, :);
    py = Y(r:r+ph-1, c:c+pw-1, :);
    pm = M(r:r+ph-1, c:c+pw-1, :);
end

function [X, Y, M] = augmentTriplet(X, Y, M, cfg)
    if cfg.augFlipLR && rand < 0.5
        X = fliplr(X); Y = fliplr(Y); M = fliplr(M);
    end
    if cfg.augFlipUD && rand < 0.5
        X = flipud(X); Y = flipud(Y); M = flipud(M);
    end
    if cfg.augRot90
        k = randi(4) - 1;
        if k > 0
            X = rot90(X, k); Y = rot90(Y, k); M = rot90(M, k);
        end
    end
end

function dlnet = buildResUNet_dlnet_Quant(patchSize, inChannels, outChannels) %#ok<DEFNU>
    assert(inChannels == outChannels, ...
        'Residual output add requires inChannels == outChannels.');

    in = imageInputLayer([patchSize inChannels], ...
        "Normalization", "none", "Name", "input");

    enc1 = convBlockGN(32, "enc1");
    pool1 = maxPooling2dLayer(2, "Stride", 2, "Name", "pool1");

    enc2 = convBlockGN(64, "enc2");
    pool2 = maxPooling2dLayer(2, "Stride", 2, "Name", "pool2");

    enc3 = convBlockGN(128, "enc3");
    pool3 = maxPooling2dLayer(2, "Stride", 2, "Name", "pool3");

    enc4 = convBlockGN(256, "enc4");
    pool4 = maxPooling2dLayer(2, "Stride", 2, "Name", "pool4");

    bott = convBlockGN(512, "bott");

    up4 = transposedConv2dLayer(2, 256, "Stride", 2, "Name", "up4");
    cat4 = depthConcatenationLayer(2, "Name", "cat4");
    dec4 = convBlockGN(256, "dec4");

    up3 = transposedConv2dLayer(2, 128, "Stride", 2, "Name", "up3");
    cat3 = depthConcatenationLayer(2, "Name", "cat3");
    dec3 = convBlockGN(128, "dec3");

    up2 = transposedConv2dLayer(2, 64, "Stride", 2, "Name", "up2");
    cat2 = depthConcatenationLayer(2, "Name", "cat2");
    dec2 = convBlockGN(64, "dec2");

    up1 = transposedConv2dLayer(2, 32, "Stride", 2, "Name", "up1");
    cat1 = depthConcatenationLayer(2, "Name", "cat1");
    dec1 = convBlockGN(32, "dec1");

    residual = convolution2dLayer(1, outChannels, ...
        "Padding", "same", "Name", "residual");
    add = additionLayer(2, "Name", "add");

    lgraph = layerGraph();
    lgraph = addLayers(lgraph, in);
    lgraph = addLayers(lgraph, enc1); lgraph = addLayers(lgraph, pool1);
    lgraph = addLayers(lgraph, enc2); lgraph = addLayers(lgraph, pool2);
    lgraph = addLayers(lgraph, enc3); lgraph = addLayers(lgraph, pool3);
    lgraph = addLayers(lgraph, enc4); lgraph = addLayers(lgraph, pool4);
    lgraph = addLayers(lgraph, bott);
    lgraph = addLayers(lgraph, up4); lgraph = addLayers(lgraph, cat4); lgraph = addLayers(lgraph, dec4);
    lgraph = addLayers(lgraph, up3); lgraph = addLayers(lgraph, cat3); lgraph = addLayers(lgraph, dec3);
    lgraph = addLayers(lgraph, up2); lgraph = addLayers(lgraph, cat2); lgraph = addLayers(lgraph, dec2);
    lgraph = addLayers(lgraph, up1); lgraph = addLayers(lgraph, cat1); lgraph = addLayers(lgraph, dec1);
    lgraph = addLayers(lgraph, residual);
    lgraph = addLayers(lgraph, add);

    lgraph = connectLayers(lgraph, "input", "enc1_conv1");
    lgraph = connectLayers(lgraph, "enc1_relu2", "pool1");
    lgraph = connectLayers(lgraph, "pool1", "enc2_conv1");
    lgraph = connectLayers(lgraph, "enc2_relu2", "pool2");
    lgraph = connectLayers(lgraph, "pool2", "enc3_conv1");
    lgraph = connectLayers(lgraph, "enc3_relu2", "pool3");
    lgraph = connectLayers(lgraph, "pool3", "enc4_conv1");
    lgraph = connectLayers(lgraph, "enc4_relu2", "pool4");
    lgraph = connectLayers(lgraph, "pool4", "bott_conv1");

    lgraph = connectLayers(lgraph, "bott_relu2", "up4");
    lgraph = connectLayers(lgraph, "up4", "cat4/in1");
    lgraph = connectLayers(lgraph, "enc4_relu2", "cat4/in2");
    lgraph = connectLayers(lgraph, "cat4", "dec4_conv1");

    lgraph = connectLayers(lgraph, "dec4_relu2", "up3");
    lgraph = connectLayers(lgraph, "up3", "cat3/in1");
    lgraph = connectLayers(lgraph, "enc3_relu2", "cat3/in2");
    lgraph = connectLayers(lgraph, "cat3", "dec3_conv1");

    lgraph = connectLayers(lgraph, "dec3_relu2", "up2");
    lgraph = connectLayers(lgraph, "up2", "cat2/in1");
    lgraph = connectLayers(lgraph, "enc2_relu2", "cat2/in2");
    lgraph = connectLayers(lgraph, "cat2", "dec2_conv1");

    lgraph = connectLayers(lgraph, "dec2_relu2", "up1");
    lgraph = connectLayers(lgraph, "up1", "cat1/in1");
    lgraph = connectLayers(lgraph, "enc1_relu2", "cat1/in2");
    lgraph = connectLayers(lgraph, "cat1", "dec1_conv1");

    lgraph = connectLayers(lgraph, "dec1_relu2", "residual");
    lgraph = connectLayers(lgraph, "residual", "add/in1");
    lgraph = connectLayers(lgraph, "input", "add/in2");

    dlnet = dlnetwork(lgraph);
end

function layers = convBlockGN(numF, prefix)
    layers = [
        convolution2dLayer(3, numF, "Padding", "same", "Name", prefix + "_conv1")
        groupNormalizationLayer(8, "Name", prefix + "_gn1")
        reluLayer("Name", prefix + "_relu1")
        convolution2dLayer(3, numF, "Padding", "same", "Name", prefix + "_conv2")
        groupNormalizationLayer(8, "Name", prefix + "_gn2")
        reluLayer("Name", prefix + "_relu2")
    ];
end

function [loss, grads] = modelGradientsQuant(dlnet, dlX, dlY, dlM, cfg, normStats)
    dlPred = forward(dlnet, dlX);

    denom = sum(dlM, "all") + eps;

    errH = (dlPred(:,:,1,:) - dlY(:,:,1,:)) .* dlM;
    errV = (dlPred(:,:,2,:) - dlY(:,:,2,:)) .* dlM;

    lossH = sum(sqrt(errH.^2 + cfg.charbEps^2), "all") ./ denom;
    lossLogV = sum(sqrt(errV.^2 + cfg.charbEps^2), "all") ./ denom;

    % Linear-V terms are computed without the common 10^logVMean factor.
    % The symmetric relative error protects dim structures, while the
    % per-patch log mean ratio prevents suppression of bright V peaks.
    exponentScale = log(10) * normStats.logVStd;
    VpredScaled = exp(exponentScale * dlPred(:,:,2,:));
    VtargetScaled = exp(exponentScale * dlY(:,:,2,:));
    symRelV = 2 * (VpredScaled - VtargetScaled) ./ ...
        (VpredScaled + VtargetScaled + 1e-6);
    lossVSymRel = sum(sqrt(symRelV.^2 + cfg.charbEps^2) .* dlM, "all") ./ denom;
    sumVpred = sum(sum(VpredScaled .* dlM, 1), 2);
    sumVtarget = sum(sum(VtargetScaled .* dlM, 1), 2);
    logMeanRatio = log((sumVpred + 1e-6) ./ (sumVtarget + 1e-6));
    lossVMean = mean(sqrt(logMeanRatio.^2 + cfg.charbEps^2), "all");

    [dxP, dyP] = imageGradients(dlPred);
    [dxT, dyT] = imageGradients(dlY);
    gradLoss = mean(sqrt((dxP - dxT).^2 + cfg.charbEps^2), "all") + ...
        mean(sqrt((dyP - dyT).^2 + cfg.charbEps^2), "all");

    loss = cfg.wH * lossH + cfg.wLogV * lossLogV + cfg.wGrad * gradLoss + ...
        cfg.wVSymRel * lossVSymRel + cfg.wVMean * lossVMean;
    grads = dlgradient(loss, dlnet.Learnables);
end

function [dx, dy] = imageGradients(dlI)
    dx = dlI(:,2:end,:,:) - dlI(:,1:end-1,:,:);
    dy = dlI(2:end,:,:,:) - dlI(1:end-1,:,:,:);
end

function valStats = validateQuantSet(dlnet, valX, valY, normStats, cfg, globalStep)
    nVal = numel(valX);
    losses = nan(nVal, 1);
    sumHError = 0;
    sumHAbsError = 0;
    sumLogVError = 0;
    sumVRelError = 0;
    sumVRelAbsError = 0;
    sumVPred = 0;
    sumVRef = 0;
    nPixels = 0;

    preview = [];
    for id = 1:nVal
        mx = getQuantMap(valX, id);
        my = getQuantMap(valY, id);
        [mx, my] = cropToCommonSize(mx, my);

        X = normalizeQuantMaps(mx, normStats);
        Y = normalizeQuantMaps(my, normStats);
        M = makeTargetMask(my, normStats);
        Ypred = predictTiledQuant(dlnet, X, cfg.patchSize, ...
            cfg.overlapInfer, cfg.useGPU);
        losses(id) = maskedArrayLoss(Ypred, Y, M, cfg);

        [Hpred, Vpred, logVpred] = denormalizeQuantChannels(Ypred, normStats);
        Href = double(my.H_Hz);
        Vref = double(my.V_power);
        logVref = log10(max(Vref, normStats.vEps));
        mask = M > 0.5 & isfinite(Hpred) & isfinite(Vpred) & ...
            isfinite(Href) & isfinite(Vref);

        hError = Hpred(mask) - Href(mask);
        logVError = logVpred(mask) - logVref(mask);
        vRelError = (Vpred(mask) - Vref(mask)) ./ ...
            max(Vref(mask), normStats.vEps);
        sumHError = sumHError + sum(hError, 'omitnan');
        sumHAbsError = sumHAbsError + sum(abs(hError), 'omitnan');
        sumLogVError = sumLogVError + sum(logVError, 'omitnan');
        sumVRelError = sumVRelError + sum(vRelError, 'omitnan');
        sumVRelAbsError = sumVRelAbsError + sum(abs(vRelError), 'omitnan');
        sumVPred = sumVPred + sum(Vpred(mask), 'omitnan');
        sumVRef = sumVRef + sum(Vref(mask), 'omitnan');
        nPixels = nPixels + nnz(mask);

        if id == 1 && shouldShowOrSavePreview(cfg)
            preview = makeTrainingTripletPreview(mx, Hpred, Vpred, my, normStats);
        end
    end

    denom = max(nPixels, 1);
    valStats = struct();
    valStats.loss = mean(losses, 'omitnan');
    valStats.hBiasMeanHz = sumHError / denom;
    valStats.hMAEMeanHz = sumHAbsError / denom;
    valStats.logVBiasMean = sumLogVError / denom;
    valStats.vRelBiasMean = sumVRelError / denom;
    valStats.vRelMAEMean = sumVRelAbsError / denom;
    valStats.vMeanRatio = sumVPred / max(sumVRef, normStats.vEps);

    if ~isempty(preview)
        titleText = sprintf('Step %06d | validation example 1 | short / predicted / long', ...
            globalStep);
        if isfield(cfg, 'showPreview') && cfg.showPreview
            figure(1001); clf;
            imshow(preview);
            title(titleText, 'Interpreter', 'none');
            drawnow;
        end
        if isfield(cfg, 'savePreview') && cfg.savePreview
            ensureDir(cfg.previewDir);
            previewName = sprintf('logV_step_%06d_validation_preview.png', globalStep);
            imwrite(preview, fullfile(cfg.previewDir, previewName));
        end
    end
end

function tf = shouldShowOrSavePreview(cfg)
    tf = (isfield(cfg, 'showPreview') && cfg.showPreview) || ...
        (isfield(cfg, 'savePreview') && cfg.savePreview);
end

function preview = makeTrainingTripletPreview(mx, Hpred, Vpred, my, ~)
    opts = struct();
    opts.Style = "oldHsvgui";
    opts.HDisplayRangeHz = [];
    opts.ConstantSaturation = 0.85;
    opts.VUseLog = false;
    opts.VDisplayRange = [];

    rgbIn = quantMapsToRgbPreview(mx.H_Hz, mx.S_invHz, mx.V_power, opts);
    rgbPred = quantMapsToRgbPreview(Hpred, [], Vpred, opts);
    rgbRef = quantMapsToRgbPreview(my.H_Hz, my.S_invHz, my.V_power, opts);

    preview = makeTripletImage(rgbIn, rgbPred, rgbRef);
end

function tripImg = makeTripletImage(Ileft, Imid, Iright)
    Ileft = im2single(Ileft);
    Imid = im2single(Imid);
    Iright = im2single(Iright);

    H = max([size(Ileft,1), size(Imid,1), size(Iright,1)]);
    W1 = size(Ileft,2);
    W2 = size(Imid,2);
    W3 = size(Iright,2);

    gap = 10;
    tripImg = ones(H, W1 + gap + W2 + gap + W3, 3, 'single');

    tripImg(1:size(Ileft,1), 1:W1, :) = Ileft;
    tripImg(1:size(Imid,1), W1+gap+(1:W2), :) = Imid;
    tripImg(1:size(Iright,1), W1+gap+W2+gap+(1:W3), :) = Iright;
end

function loss = maskedArrayLoss(Ypred, Y, M, cfg)
    denom = sum(M(:)) + eps;
    errH = (Ypred(:,:,1) - Y(:,:,1)) .* M;
    errV = (Ypred(:,:,2) - Y(:,:,2)) .* M;
    lossH = sum(sqrt(errH(:).^2 + cfg.charbEps^2)) / denom;
    lossV = sum(sqrt(errV(:).^2 + cfg.charbEps^2)) / denom;
    loss = cfg.wH * lossH + cfg.wLogV * lossV;
end

function Iout = predictTiledQuant(dlnet, Iin, patchSize, overlap, useGPU)
    Iin = single(Iin);
    [H, W, C] = size(Iin);

    ph = patchSize(1);
    pw = patchSize(2);
    sh = ph - overlap;
    sw = pw - overlap;
    assert(sh > 0 && sw > 0, "overlap must be smaller than patchSize.");

    padBottom = max(0, ph - H);
    padRight = max(0, pw - W);
    Ipad = padarray(Iin, [padBottom padRight], "symmetric", "post");
    [Hp, Wp, ~] = size(Ipad);

    rStarts = unique([1:sh:(Hp-ph+1), Hp-ph+1]);
    cStarts = unique([1:sw:(Wp-pw+1), Wp-pw+1]);

    acc = zeros(Hp, Wp, C, "single");
    wsum = zeros(Hp, Wp, "single");
    win = hann2d(ph, pw, C);

    for r = rStarts
        for c = cStarts
            patch = Ipad(r:r+ph-1, c:c+pw-1, :);
            dlX = dlarray(reshape(patch, [ph pw C 1]), "SSCB");
            if useGPU
                dlX = gpuArray(dlX);
            end

            dlP = predict(dlnet, dlX);
            pred = gather(extractdata(dlP(:,:,:,1)));

            acc(r:r+ph-1, c:c+pw-1, :) = acc(r:r+ph-1, c:c+pw-1, :) + pred .* win;
            wsum(r:r+ph-1, c:c+pw-1) = wsum(r:r+ph-1, c:c+pw-1) + win(:,:,1);
        end
    end

    IpadOut = acc ./ max(wsum, 1e-6);
    Iout = IpadOut(1:H, 1:W, :);
end

function w = hann2d(H, W, C)
    wy = hann1d(H);
    wx = hann1d(W);
    w2 = single(wy * wx');
    w2 = max(w2, 1e-3);
    w = repmat(w2, 1, 1, C);
end

function w = hann1d(N)
    if N == 1
        w = 1;
        return;
    end
    n = (0:N-1)';
    w = 0.5 - 0.5 * cos(2*pi*n/(N-1));
end

function [H, V, logV] = denormalizeQuantChannels(Y, normStats)
    Hn = double(Y(:,:,1));
    logVn = double(Y(:,:,2));
    H = ((Hn + 1) / 2) * (normStats.hMaxHz - normStats.hMinHz) + normStats.hMinHz;
    logV = logVn * normStats.logVStd + normStats.logVMean;
    V = 10.^logV - normStats.vEps;
    V = max(V, 0);
end
