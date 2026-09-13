function results = run_DFFOCT_denoising(useGPU, outputDir)
% Apply the trained residual U-Net to short-acquisition quantitative maps.
% Input MAT files must contain a struct named "maps" with H_Hz and V_power.
% The model operates on [H_Hz, log10(V_power)] internally and saves H_Hz and
% linear V_power outputs. RGB images are display-only old-HSV-style previews.

%% Paths
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

modelFile = fullfile(projectRoot, "model", ...
    "DFFOCT_resUNet_TimeNormalizedV_IntensityPreserving_dlnet.mat");
inputDir = fullfile(projectRoot, "data", "quant_maps", "short_acquisition");
if nargin < 1 || isempty(useGPU)
    useGPU = canUseGPU();
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(projectRoot, "output", "denoised");
end
assert(isscalar(useGPU) && (islogical(useGPU) || ismember(useGPU, [0 1])), ...
    'useGPU must be a logical scalar.');
assert(~useGPU || canUseGPU(), 'No supported GPU is available. Use false for CPU.');
assert(isfile(modelFile), 'Missing supplied checkpoint: %s', modelFile);
addpath(scriptDir);
mapOutputDir = fullfile(outputDir, "quantitative_maps");
rgbOutputDir = fullfile(outputDir, "rgb_preview");

ensureDir(mapOutputDir);
ensureDir(rgbOutputDir);

%% Load model and fixed training normalization
S = load(modelFile);
assert(isfield(S, 'dlnet'), 'Checkpoint does not contain dlnet.');
assert(isfield(S, 'cfg'), 'Checkpoint does not contain cfg.');
assert(isfield(S, 'normStats'), 'Checkpoint does not contain normStats.');

dlnet = S.dlnet;
cfg = S.cfg;
normStats = S.normStats;
assert(isfield(normStats, 'vPowerNormalization') && ...
    strcmp(string(normStats.vPowerNormalization), 'frameCountSquared'), ...
    'This entry point requires a frame-count-normalized checkpoint.');

if useGPU
    dlnet = dlupdate(@gpuArray, dlnet);
else
    dlnet = dlupdate(@gather, dlnet);
end

%% Run inference
files = dir(fullfile(inputDir, '*.mat'));
assert(~isempty(files), 'No MAT files found in %s.', inputDir);

fprintf('Model: %s\n', modelFile);
fprintf('Input files: %d\n', numel(files));
fprintf('GPU enabled: %d\n', useGPU);
results = struct('inputFile', {}, 'outputFile', {}, 'rgbFile', {}, 'anchorScale', {});

for i = 1:numel(files)
    inputFile = fullfile(files(i).folder, files(i).name);
    [~, stem] = fileparts(files(i).name);
    stem = string(stem);
    fprintf('[%d/%d] %s\n', i, numel(files), stem);

    inputMaps = loadQuantMaps(inputFile);
    X = normalizeQuantMaps(inputMaps, normStats);
    Y = predictTiledQuant(dlnet, X, cfg.patchSize, cfg.overlapInfer, useGPU);
    [Hpred, Vpred] = denormalizeQuantChannels(Y, normStats);
    [anchorScale, anchorMask] = inputAnchoredVScale( ...
        double(inputMaps.V_power), Vpred, inputMaps.validMask, normStats.vEps);
    Vpred = Vpred * anchorScale;
    assert(all(isfinite(Hpred), 'all') && all(isfinite(Vpred), 'all'), ...
        'Inference returned non-finite values.');

    maps = struct();
    maps.H_Hz = single(Hpred);
    maps.S_invHz = [];
    maps.V_power = single(Vpred);
    maps.validMask = inputMaps.validMask & isfinite(Hpred) & isfinite(Vpred) & Vpred > 0;
    maps.sourceInput = string(files(i).name);
    [~, modelStem, modelExt] = fileparts(modelFile);
    maps.modelFile = string(modelStem) + string(modelExt);
    maps.powerMode = "amplitudeSquared";
    maps.vNormalization = "sum(abs(FFT).^2/N^2) within band";
    maps.bandHz = inputMaps.bandHz;
    maps.samplingRateHz = inputMaps.samplingRateHz;
    maps.sourceFrameCount = inputMaps.frameCount;
    maps.targetFrameCount = 512;
    maps.frameNormalize = inputMaps.frameNormalize;
    maps.inputAnchorScale = anchorScale;
    maps.inputAnchorMaskPixelCount = nnz(anchorMask);
    maps.inputAnchorPercentile = 20;
    maps.inputAnchorScaleRange = [0.5 2.0];
    maps.version = "dffoct-denoising-minimal-v2-time-normalized-intensity-preserving";

    outMat = fullfile(mapOutputDir, stem + "_denoised.mat");
    save(outMat, ...
        'maps', 'normStats', '-v7.3');

    displayOpts = struct();
    displayOpts.Style = "oldHsvgui";
    displayOpts.ConstantSaturation = 0.85;
    displayOpts.VUseLog = false;

    % S is not predicted by this two-channel network. A fixed saturation is
    % therefore used so RGB remains a transparent display representation.
    rgbPred = quantMapsToRgbPreview(Hpred, [], Vpred, displayOpts);
    outPng = fullfile(rgbOutputDir, stem + "_denoised_rgb.png");
    imwrite(rgbPred, outPng);
    results(i) = struct('inputFile', string(inputFile), 'outputFile', string(outMat), ...
        'rgbFile', string(outPng), 'anchorScale', anchorScale);
    fprintf('  Input-only V anchor factor: %.6f\n', anchorScale);
end

fprintf('Denoising complete. Results: %s\n', outputDir);
end

%% Local functions
function ensureDir(folderName)
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
end

function tf = canUseGPU()
    try
        tf = (gpuDeviceCount > 0);
    catch
        tf = false;
    end
end

function maps = loadQuantMaps(filename)
    S = load(filename, 'maps');
    assert(isfield(S, 'maps'), ...
        'MAT file does not contain a struct named maps: %s', filename);
    maps = S.maps;
    assert(isfield(maps, 'H_Hz') && isfield(maps, 'V_power'), ...
        'maps must contain H_Hz and V_power: %s', filename);
    maps.H_Hz = single(maps.H_Hz);
    maps.V_power = single(maps.V_power);
    validateDFFOCTInputMaps(maps, 48);
    if ~isfield(maps, 'validMask')
        maps.validMask = isfinite(maps.H_Hz) & isfinite(maps.V_power) & maps.V_power > 0;
    else
        maps.validMask = logical(maps.validMask);
    end
end

function [scale, mask] = inputAnchoredVScale(Vinput, Vpred, validMask, vEps)
    mask = logical(validMask) & isfinite(Vinput) & isfinite(Vpred) & ...
        Vinput > 0 & Vpred > 0;
    if ~any(mask, 'all')
        scale = 1;
        return;
    end
    logVinput = log10(max(Vinput, vEps));
    threshold = prctile(logVinput(mask), 20);
    mask = mask & logVinput >= threshold;
    scale = sum(Vinput(mask), 'omitnan') / max(sum(Vpred(mask), 'omitnan'), vEps);
    scale = min(max(scale, 0.5), 2.0);
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

function Iout = predictTiledQuant(dlnet, Iin, patchSize, overlap, useGPU)
    Iin = single(Iin);
    [height, width, channels] = size(Iin);
    patchHeight = patchSize(1);
    patchWidth = patchSize(2);
    stepHeight = patchHeight - overlap;
    stepWidth = patchWidth - overlap;
    assert(stepHeight > 0 && stepWidth > 0, ...
        'Inference overlap must be smaller than patch size.');

    padBottom = max(0, patchHeight - height);
    padRight = max(0, patchWidth - width);
    padded = padarray(Iin, [padBottom padRight], 'symmetric', 'post');
    [paddedHeight, paddedWidth, ~] = size(padded);

    rowStarts = unique([1:stepHeight:(paddedHeight-patchHeight+1), ...
        paddedHeight-patchHeight+1]);
    colStarts = unique([1:stepWidth:(paddedWidth-patchWidth+1), ...
        paddedWidth-patchWidth+1]);

    accumulator = zeros(paddedHeight, paddedWidth, channels, 'single');
    weightSum = zeros(paddedHeight, paddedWidth, 'single');
    window = hann2d(patchHeight, patchWidth, channels);

    for row = rowStarts
        for col = colStarts
            patch = padded(row:row+patchHeight-1, ...
                col:col+patchWidth-1, :);
            dlX = dlarray(reshape(patch, ...
                [patchHeight patchWidth channels 1]), 'SSCB');
            if useGPU
                dlX = gpuArray(dlX);
            end

            dlPrediction = predict(dlnet, dlX);
            prediction = gather(extractdata(dlPrediction(:,:,:,1)));
            rows = row:row+patchHeight-1;
            cols = col:col+patchWidth-1;
            accumulator(rows, cols, :) = accumulator(rows, cols, :) + ...
                prediction .* window;
            weightSum(rows, cols) = weightSum(rows, cols) + window(:,:,1);
        end
    end

    paddedOutput = accumulator ./ max(weightSum, 1e-6);
    Iout = paddedOutput(1:height, 1:width, :);
end

function w = hann2d(height, width, channels)
    wy = hann1d(height);
    wx = hann1d(width);
    w2 = single(wy * wx');
    w2 = max(w2, 1e-3);
    w = repmat(w2, 1, 1, channels);
end

function w = hann1d(nSamples)
    if nSamples == 1
        w = 1;
        return;
    end
    n = (0:nSamples-1)';
    w = 0.5 - 0.5 * cos(2*pi*n/(nSamples-1));
end

function [H, V] = denormalizeQuantChannels(Y, normStats)
    Hn = double(Y(:,:,1));
    logVn = double(Y(:,:,2));
    H = ((Hn + 1) / 2) * ...
        (normStats.hMaxHz - normStats.hMinHz) + normStats.hMinHz;
    logV = logVn * normStats.logVStd + normStats.logVMean;
    V = max(10.^logV - normStats.vEps, 0);
end
