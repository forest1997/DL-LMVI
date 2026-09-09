%% generate_DFFOCT_quantMaps_fromRaw.m
% Raw DFF/OCT time series -> quantitative H/S/V maps.
%
% The MAT files generated here are the source for quantitative training and
% evaluation. Preview PNG files are display-only and should not be used for
% biological measurements.

close all; clear; clc;

%% ===================== 1) Input/output configuration =====================
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

rootDirs = {
    fullfile(projectRoot, 'data', 'raw')
};
rootDirs = unique(rootDirs, 'stable');

outRoot = fullfile(projectRoot, 'data', 'quant_maps');
previewRoot = fullfile(outRoot, 'Preview_RGB');

overwriteExisting = false;
writePreviewPng = true;

%% ===================== 2) Acquisition and FFT settings ====================
imageWidth = 800;
imageHeight = 550;
numPixels = imageWidth * imageHeight;

samplingRateHz = 100;
fBandHz = [1 16];
% fBandHz = [1 50];
% PowerMode "legacy" matches computeHSVnewfft.m: abs(fft).^2.
% Other options are "perSample" and "amplitudeSquared".
quantOpts = struct();
quantOpts.FrameNormalize = true;
quantOpts.PowerMode = "legacy";
quantOpts.EpsPower = 1e-12;
quantOpts.OutputClass = "single";

processConfigs = {
    % {16,  '_Range16',  'Range16'}
    % {32,  '_Range32',  'Range32'}
    {48,  '_Range48',  'short_acquisition'}
    {64,  '_Range64',  'Range64'}
    {512, '_Range512', 'long_reference'}
};

strictFrameRequirement = true;

%% ===================== 3) Create output folders ===========================
if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end
if writePreviewPng && ~exist(previewRoot, 'dir')
    mkdir(previewRoot);
end

for cfgIdx = 1:numel(processConfigs)
    subFolder = processConfigs{cfgIdx}{3};
    ensureDir(fullfile(outRoot, subFolder));
    if writePreviewPng
        ensureDir(fullfile(previewRoot, subFolder));
    end
end

%% ===================== 4) Batch processing ================================
for rd = 1:numel(rootDirs)
    rootDir = rootDirs{rd};

    if ~exist(rootDir, 'dir')
        warning('Input folder does not exist, skipped: %s', rootDir);
        continue;
    end

    fprintf('\n=============================================\n');
    fprintf('Processing folder [%d/%d]: %s\n', rd, numel(rootDirs), rootDir);
    fprintf('Quantitative MAT output: %s\n', outRoot);
    fprintf('=============================================\n');

    files = dir(fullfile(rootDir, '*.raw'));
    if isempty(files)
        warning('No *.raw files found in %s.', rootDir);
        continue;
    end

    safeRootName = makeSafePathName(rootDir);

    for fi = 1:numel(files)
        fname = fullfile(files(fi).folder, files(fi).name);
        [~, baseName, ~] = fileparts(files(fi).name);

        try
            fprintf('\n[%d/%d] Reading raw file: %s\n', fi, numel(files), baseName);

            fid = fopen(fname, 'rb');
            if fid < 0
                warning('Cannot open file, skipped: %s', fname);
                continue;
            end
            rawData = fread(fid, inf, 'uint16=>double');
            fclose(fid);

            nFramesTotal = floor(numel(rawData) / numPixels);
            if nFramesTotal < 1
                warning('File has less than one full frame, skipped: %s', files(fi).name);
                clear rawData;
                continue;
            end

            rawData = rawData(1:nFramesTotal * numPixels);
            volFull = reshape(rawData, [imageWidth, imageHeight, nFramesTotal]);
            volFull = permute(volFull, [2, 1, 3]);   % [H W T]
            clear rawData;

            fprintf('   Total frames: %d\n', nFramesTotal);

            for cfgIdx = 1:numel(processConfigs)
                targetFrames = processConfigs{cfgIdx}{1};
                suffix = processConfigs{cfgIdx}{2};
                subFolder = processConfigs{cfgIdx}{3};

                if strictFrameRequirement && nFramesTotal < targetFrames
                    fprintf('   -> %s skipped: need %d frames, got %d.\n', ...
                        suffix, targetFrames, nFramesTotal);
                    continue;
                end

                currentUseFrames = min(nFramesTotal, targetFrames);
                outDir = fullfile(outRoot, subFolder);
                previewDir = fullfile(previewRoot, subFolder);

                fileStem = sprintf('%s_%s%s_quant', safeRootName, baseName, suffix);
                outMat = fullfile(outDir, [fileStem '.mat']);
                outPng = fullfile(previewDir, [fileStem '_preview.png']);

                if ~overwriteExisting && isfile(outMat)
                    fprintf('   -> %s exists, skipped.\n', suffix);
                    continue;
                end

                fprintf('   -> Computing %s using %d frames... ', suffix, currentUseFrames);

                volSlice = volFull(:, :, 1:currentUseFrames);
                maps = computeDFFOCTQuantMaps(volSlice, samplingRateHz, fBandHz, quantOpts);

                meta = struct();
                meta.sourceRawFile = fname;
                meta.sourceRoot = rootDir;
                meta.baseName = baseName;
                meta.usedFrames = currentUseFrames;
                meta.totalFrames = nFramesTotal;
                meta.imageWidth = imageWidth;
                meta.imageHeight = imageHeight;
                meta.samplingRateHz = samplingRateHz;
                meta.fBandHz = fBandHz;
                meta.quantOpts = quantOpts;
                meta.createdBy = mfilename;
                meta.createdOn = char(datetime('now'));

                save(outMat, 'maps', 'meta', '-v7.3');

                if writePreviewPng
                    rgbPreview = quantMapsToRgbPreview(maps);
                    imwrite(rgbPreview, outPng);
                end

                fprintf('done.\n');
                clear volSlice maps meta rgbPreview;
            end

            clear volFull;

        catch ME
            warning('Failed processing %s: %s', files(fi).name, ME.message);
            clear rawData volFull volSlice maps;
            continue;
        end
    end
end

fprintf('\nAll quantitative maps generated.\n');
fprintf('MAT output: %s\n', outRoot);
if writePreviewPng
    fprintf('Preview output: %s\n', previewRoot);
end

%% ===================== Local helpers ======================================
function ensureDir(folderName)
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
end

function safeName = makeSafePathName(pathName)
    safeName = regexprep(pathName, '[:\\/]+', '_');
    safeName = regexprep(safeName, '^_+', '');
end
