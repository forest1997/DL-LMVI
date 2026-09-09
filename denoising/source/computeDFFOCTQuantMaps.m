function maps = computeDFFOCTQuantMaps(data, samplingRateHz, fBandHz, opts)
% computeDFFOCTQuantMaps
% Quantitative frequency-domain maps for DFF/OCT time-series data.
%
% Inputs
%   data           [H W T] numeric volume
%   samplingRateHz frames per second
%   fBandHz        [low high] frequency band for dynamic intensity
%   opts           optional struct:
%       FrameNormalize : divide each frame by its global mean, default true
%       PowerMode      : "legacy", "perSample", or "amplitudeSquared"
%                        legacy matches the previous abs(fft).^2 behavior
%       EpsPower       : small value for numerical safety
%       OutputClass    : "single" or "double"
%
% Outputs
%   maps.H_Hz       weighted mean fluctuation frequency, in Hz
%   maps.S_invHz    inverse spectral width, same idea as old S map
%   maps.V_power    band-integrated dynamic intensity
%   maps.validMask  finite pixels with non-zero spectral power

    if nargin < 4 || isempty(opts)
        opts = struct();
    end
    opts = fillDefaults(opts);

    [rows, cols, frames] = size(data);
    if frames < 2
        error('computeDFFOCTQuantMaps:NotEnoughFrames', ...
            'At least 2 frames are required.');
    end

    data = double(data);

    if opts.FrameNormalize
        meanFrames = mean(reshape(data, [], frames), 1, 'omitnan');
        bad = ~isfinite(meanFrames) | abs(meanFrames) < eps;
        meanFrames(bad) = 1;
        data = data ./ reshape(meanFrames, 1, 1, frames);
    end

    nPositive = floor(frames / 2);
    posIdx = 2:(nPositive + 1);              % skip DC
    freqHz = (1:nPositive) * (samplingRateHz / frames);

    data2d = reshape(data, [], frames);
    fftData = fft(data2d, [], 2);
    power = abs(fftData(:, posIdx)).^2;

    switch lower(string(opts.PowerMode))
        case "legacy"
            % Keep compatibility with the original computeHSVnewfft.m.
        case "persample"
            power = power ./ frames;
        case "amplitudesquared"
            power = power ./ (frames.^2);
        otherwise
            error('Unknown PowerMode: %s', opts.PowerMode);
    end

    sumPower = sum(power, 2);
    valid = isfinite(sumPower) & sumPower > opts.EpsPower;
    safeSumPower = max(sumPower, opts.EpsPower);

    freqRow = reshape(freqHz, 1, []);
    meanFreq = sum(power .* freqRow, 2) ./ safeSumPower;

    meanFreq2 = sum(power .* (freqRow.^2), 2) ./ safeSumPower;
    varFreq = max(meanFreq2 - meanFreq.^2, 0);
    invWidth = 1 ./ sqrt(varFreq + opts.EpsPower);

    bandMask = freqHz >= fBandHz(1) & freqHz <= fBandHz(2);
    if ~any(bandMask)
        warning('computeDFFOCTQuantMaps:EmptyBand', ...
            'No FFT frequency bin falls inside [%.4g %.4g] Hz.', ...
            fBandHz(1), fBandHz(2));
        dynIntensity = zeros(size(sumPower));
    else
        dynIntensity = sum(power(:, bandMask), 2);
    end

    meanFreq(~valid) = 0;
    invWidth(~valid) = 0;
    dynIntensity(~valid) = 0;

    H_Hz = reshape(meanFreq, rows, cols);
    S_invHz = reshape(invWidth, rows, cols);
    V_power = reshape(dynIntensity, rows, cols);
    validMask = reshape(valid, rows, cols);

    if strcmpi(opts.OutputClass, 'single')
        H_Hz = single(H_Hz);
        S_invHz = single(S_invHz);
        V_power = single(V_power);
        freqHz = single(freqHz);
    end

    maps = struct();
    maps.H_Hz = H_Hz;
    maps.S_invHz = S_invHz;
    maps.V_power = V_power;
    maps.validMask = validMask;
    maps.frequencyHz = freqHz;
    maps.bandHz = fBandHz;
    maps.samplingRateHz = samplingRateHz;
    maps.frameCount = frames;
    maps.frameNormalize = opts.FrameNormalize;
    maps.powerMode = string(opts.PowerMode);
    maps.version = "quant-maps-v1";
end

function opts = fillDefaults(opts)
    if ~isfield(opts, 'FrameNormalize')
        opts.FrameNormalize = true;
    end
    if ~isfield(opts, 'PowerMode')
        opts.PowerMode = "legacy";
    end
    if ~isfield(opts, 'EpsPower')
        opts.EpsPower = 1e-12;
    end
    if ~isfield(opts, 'OutputClass')
        opts.OutputClass = "single";
    end
end
