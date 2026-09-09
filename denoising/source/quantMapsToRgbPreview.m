function rgbImage = quantMapsToRgbPreview(H_Hz, S_invHz, V_power, opts)
% quantMapsToRgbPreview
% Display-only RGB rendering for quantitative maps.
%
% This function is intentionally not used for quantitative evaluation.
% It only creates a convenient preview image from H/V maps.

    if nargin == 1 && isstruct(H_Hz)
        maps = H_Hz;
        H_Hz = maps.H_Hz;
        if isfield(maps, 'S_invHz')
            S_invHz = maps.S_invHz;
        else
            S_invHz = [];
        end
        V_power = maps.V_power;
        opts = struct();
    elseif nargin < 4 || isempty(opts)
        opts = struct();
    end

    opts = fillDefaults(opts);

    H_Hz = double(H_Hz);
    V_power = double(V_power);
    if isempty(S_invHz)
        S_invHz = [];
    else
        S_invHz = double(S_invHz);
    end

    if strcmpi(string(opts.Style), "oldHsvgui")
        rgbImage = renderOldHsvguiStyle(H_Hz, S_invHz, V_power, opts);
        return;
    end

    signalMask = isfinite(H_Hz) & isfinite(V_power) & V_power > opts.EpsV;
    if nnz(signalMask) < 10
        signalMask = isfinite(H_Hz) & isfinite(V_power);
    end

    if isempty(opts.HDisplayRangeHz)
        [hLo, hHi] = robustPercentile(H_Hz(signalMask), opts.HPercentiles);
    else
        hLo = opts.HDisplayRangeHz(1);
        hHi = opts.HDisplayRangeHz(2);
    end
    hue = clipScale(H_Hz, hLo, hHi, opts.HueRange(1), opts.HueRange(2));

    if isempty(S_invHz)
        sat = ones(size(H_Hz)) * opts.ConstantSaturation;
    else
        [sLo, sHi] = robustPercentile(S_invHz(signalMask), opts.SPercentiles);
        sat = clipScale(S_invHz, sLo, sHi, opts.SDisplayRange(1), opts.SDisplayRange(2));
    end

    if opts.VUseLog
        Vvis = log10(max(V_power, opts.EpsV));
    else
        Vvis = V_power;
    end
    if isempty(opts.VDisplayRange)
        [vLo, vHi] = robustPercentile(Vvis(isfinite(Vvis)), opts.VPercentiles);
    else
        vLo = opts.VDisplayRange(1);
        vHi = opts.VDisplayRange(2);
    end
    val = clipScale(Vvis, vLo, vHi, 0, 1);

    hue(~isfinite(hue)) = 0;
    sat(~isfinite(sat)) = 0;
    val(~isfinite(val)) = 0;

    hsvImage = cat(3, hue, sat, val);
    rgbImage = hsv2rgb(hsvImage);
    rgbImage = min(max(rgbImage, 0), 1);
end

function opts = fillDefaults(opts)
    if ~isfield(opts, 'Style')
        opts.Style = "general";
    end
    if ~isfield(opts, 'HDisplayRangeHz')
        opts.HDisplayRangeHz = [];
    end
    if ~isfield(opts, 'HPercentiles')
        opts.HPercentiles = [2 98];
    end
    if ~isfield(opts, 'HueRange')
        opts.HueRange = [0.1 0.6];
    end
    if ~isfield(opts, 'SPercentiles')
        opts.SPercentiles = [2 98];
    end
    if ~isfield(opts, 'SDisplayRange')
        opts.SDisplayRange = [0.55 0.95];
    end
    if ~isfield(opts, 'ConstantSaturation')
        opts.ConstantSaturation = 0.85;
    end
    if ~isfield(opts, 'VUseLog')
        opts.VUseLog = true;
    end
    if ~isfield(opts, 'VDisplayRange')
        opts.VDisplayRange = [];
    end
    if ~isfield(opts, 'VPercentiles')
        opts.VPercentiles = [1 99.8];
    end
    if ~isfield(opts, 'EpsV')
        opts.EpsV = 1e-12;
    end
end

function rgbImage = renderOldHsvguiStyle(H_Hz, S_invHz, V_power, opts)
    V = V_power;
    if isempty(opts.VDisplayRange)
        [vLo, vHi] = robustPercentile(V(:), [37 98]);
    else
        vLo = opts.VDisplayRange(1);
        vHi = opts.VDisplayRange(2);
    end
    V = clipScale(V, vLo, vHi, 0, 1);

    if isempty(S_invHz)
        S = ones(size(H_Hz)) * opts.ConstantSaturation;
    else
        [sLo, sHi] = robustPercentile(S_invHz(:), [2 90]);
        S = clipScale(S_invHz, sLo, sHi, 0.55, 0.95);
    end

    mask = V > 0.05 & isfinite(H_Hz);
    if nnz(mask) < 10
        mask = isfinite(H_Hz);
    end

    if isempty(opts.HDisplayRangeHz)
        [hLo, hHi] = robustPercentile(H_Hz(mask), [2 98]);
    else
        hLo = opts.HDisplayRangeHz(1);
        hHi = opts.HDisplayRangeHz(2);
    end
    H = clipScale(H_Hz, hLo, hHi, 0.1, 0.6);

    H(~isfinite(H)) = 0;
    S(~isfinite(S)) = 0;
    V(~isfinite(V)) = 0;

    rgbImage = hsv2rgb(cat(3, H, S, V));
    rgbImage = min(max(rgbImage, 0), 1);
end

function y = clipScale(x, lo, hi, outLo, outHi)
    if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
        lo = min(x(:), [], 'omitnan');
        hi = max(x(:), [], 'omitnan');
    end
    if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
        lo = 0;
        hi = 1;
    end

    x = min(max(x, lo), hi);
    y = (x - lo) ./ (hi - lo + eps);
    y = y .* (outHi - outLo) + outLo;
end

function [lo, hi] = robustPercentile(x, pct)
    x = x(isfinite(x));
    if isempty(x)
        lo = 0;
        hi = 1;
        return;
    end

    lo = prctile(x, pct(1));
    hi = prctile(x, pct(2));
    if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
        lo = min(x);
        hi = max(x);
    end
    if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
        lo = 0;
        hi = 1;
    end
end
