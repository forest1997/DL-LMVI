function report = verify_packaged_example(useGPU)
% Verify the released model and example without retraining or editing inputs.
    if nargin < 1
        useGPU = false;
    end
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'source'));
    name = "F_OAFOV3_OAcellsLFOV3_images_20251113_220325";
    inputFile = fullfile(root, 'data', 'quant_maps', 'short_acquisition', ...
        name + '_Range48_quant.mat');
    expectedFile = fullfile(root, 'output', 'denoised', 'quantitative_maps', ...
        name + '_Range48_quant_denoised.mat');
    expected = load(expectedFile, 'maps');
    input = load(inputFile, 'maps');
    validateDFFOCTInputMaps(input.maps, 48);
    manifest = readtable(fullfile(root, 'config', 'dataset_split_91_pairs.csv'), ...
        'TextType', 'string', 'Delimiter', ',', 'VariableNamingRule', 'preserve');
    assert(height(manifest) == 91 && numel(unique(manifest.baseName)) == 91);
    assert(sum(manifest.split == "train") == 47);
    assert(sum(manifest.split == "validation") == 9);
    assert(sum(manifest.split == "held_out_test") == 35);
    assert(manifest.split(manifest.baseName == name) == "held_out_test");

    wrong = input.maps;
    wrong.powerMode = 'legacy';
    mustReject(wrong, 'DFFOCT:WrongPowerMode');
    wrong = input.maps;
    wrong.frameCount = 512;
    mustReject(wrong, 'DFFOCT:AcquisitionMismatch');

    opts = struct('FrameNormalize', false, 'PowerMode', 'amplitudeSquared', ...
        'OutputClass', 'double');
    for n = [48 512]
        signal = reshape(1 + 0.2*cos(2*pi*25*(0:n-1)/100), 1, 1, n);
        normalized = computeDFFOCTQuantMaps(signal, 100, [1 50], opts);
        legacyOpts = opts;
        legacyOpts.PowerMode = 'legacy';
        legacy = computeDFFOCTQuantMaps(signal, 100, [1 50], legacyOpts);
        assert(abs(normalized.V_power - 0.01) < 1e-12);
        assert(abs(normalized.H_Hz - 25) < 1e-10);
        assert(abs(normalized.V_power - legacy.V_power/n^2) < 1e-12);
    end

    destination = tempname;
    result = run_DFFOCT_denoising(useGPU, destination);
    match = find(endsWith([result.inputFile], name + '_Range48_quant.mat'), 1);
    assert(~isempty(match));
    actual = load(result(match).outputFile, 'maps');
    hDifference = abs(double(actual.maps.H_Hz) - double(expected.maps.H_Hz));
    hError = max(hDifference, [], 'all');
    hMAE = mean(hDifference, 'all');
    vError = max(abs(double(actual.maps.V_power) - double(expected.maps.V_power)), [], 'all');
    vScale = max(double(expected.maps.V_power), [], 'all');
    % CPU and GPU convolutions need not be bit-identical. The H maximum
    % tolerance is 0.1% of its fixed 50-Hz normalization range.
    assert(hError < 0.05 && hMAE < 0.005, 'H differs from the supplied expected output.');
    assert(vError / max(vScale, 1e-12) < 1e-3, 'V differs from the expected output.');
    assert(abs(actual.maps.inputAnchorScale - expected.maps.inputAnchorScale) < 1e-3);
    assert(all(isfinite(actual.maps.H_Hz), 'all'));
    assert(all(isfinite(actual.maps.V_power) & actual.maps.V_power >= 0, 'all'));
    rgb = imread(result(match).rgbFile);
    assert(size(rgb, 3) == 3 && max(rgb, [], 'all') > min(rgb, [], 'all'));

    report = struct('passed', true, 'useGPU', logical(useGPU), ...
        'maxAbsoluteHErrorHz', hError, 'maxAbsoluteVError', vError, ...
        'meanAbsoluteHErrorHz', hMAE, 'HMaximumToleranceHz', 0.05, ...
        'HMeanToleranceHz', 0.005, 'VPeakRelativeTolerance', 1e-3, ...
        'VErrorRelativeToPeak', vError/max(vScale, 1e-12), ...
        'anchorScale', actual.maps.inputAnchorScale, 'temporaryOutput', destination);
    disp(report);
end

function mustReject(maps, expectedIdentifier)
    rejected = false;
    try
        validateDFFOCTInputMaps(maps, 48);
    catch ex
        assert(strcmp(ex.identifier, expectedIdentifier), 'Unexpected validation failure.');
        rejected = true;
    end
    assert(rejected, 'Invalid input was not rejected.');
end
