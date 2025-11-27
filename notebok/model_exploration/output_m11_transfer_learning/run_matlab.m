%% MATLAB Inference from ONNX - CLEAN SIMPLIFIED
clear; clc; close all;

%% Paths
sampleFile = 'E:\MPL_Data\TRIMMM\Day15\mm_results_D15F_S6B_1.mat';
onnxFile   = 'C:\Users\user_picm\Desktop\mmTissueFilter\notebok\model_exploration\output_m11_transfer_learning\models\best_model.onnx';
outputDir  = 'results';
srcVar     = 'M11Z';

if ~exist(outputDir,'dir'); mkdir(outputDir); end

%% Load network
try
    net = importNetworkFromONNX(onnxFile);
    isDL = isa(net,'dlnetwork');
    fprintf('Loaded network\n');
catch ME
    error('Failed to load ONNX: %s', ME.message);
end

%% Load data
S = load(sampleFile);
assert(isfield(S, srcVar), 'Variable %s not found', srcVar);
m11_raw = double(S.(srcVar));
[H0, W0] = size(m11_raw);
fprintf('Loaded %s: [%d x %d]\n', srcVar, H0, W0);

%% Preprocess - MATCH TRAINING EXACTLY
% Step 1: Percentile clipping (handles outliers)
p1  = prctile(m11_raw(:), 1);
p99 = prctile(m11_raw(:), 99);
m11_clipped = m11_raw;
m11_clipped(m11_clipped < p1)  = p1;
m11_clipped(m11_clipped > p99) = p99;
fprintf('Clipped [%.3e, %.3e] -> [%.3e, %.3e]\n', ...
    min(m11_raw(:)), max(m11_raw(:)), p1, p99);

% Step 2: Min-max normalization to [0, 1]
mn = min(m11_clipped(:));
mx = max(m11_clipped(:));
if mx > mn
    m11_norm = (m11_clipped - mn) / (mx - mn);
else
    m11_norm = zeros(size(m11_clipped));
    warning('Zero range after clipping');
end
fprintf('Normalized to [%.6f, %.6f]\n', min(m11_norm(:)), max(m11_norm(:)));

% Step 3: Resize to 512x512
inputSize = [512, 512];
m11_resized = imresize(m11_norm, inputSize, 'bilinear');

% Step 4: Replicate to 3 channels
m11_3ch = repmat(m11_resized, [1, 1, 3]);

% Step 5: ImageNet normalization
imagenetMean = [0.485, 0.456, 0.406];
imagenetStd  = [0.229, 0.224, 0.225];

x_normalized = zeros(size(m11_3ch), 'single');
for c = 1:3
    x_normalized(:, :, c) = single((m11_3ch(:, :, c) - imagenetMean(c)) / imagenetStd(c));
end
fprintf('After ImageNet norm: [%.3f, %.3f]\n', ...
    min(x_normalized(:)), max(x_normalized(:)));

%% Predict
X = reshape(x_normalized, [inputSize(1), inputSize(2), 3, 1]);
if isDL
    dlX = dlarray(X, 'SSCB');
    dlY = predict(net, dlX);
    Y = extractdata(gather(dlY));
else
    Y = predict(net, X);
end
fprintf('Prediction complete. Output shape: [%s]\n', num2str(size(Y)));

%% Post-process
assert(size(Y, 3) == 2, 'Expected 2 channels, got %d', size(Y, 3));

% Apply softmax if needed
channel_sum = sum(Y, 3);
if any(abs(channel_sum(:) - 1) > 1e-3)
    Y = Y - max(Y, [], 3);
    Y = exp(Y);
    Y = Y ./ sum(Y, 3);
end

% Extract tissue probability and mask
P_tissue = Y(:, :, 2, 1);
mask_512 = P_tissue >= 0.5;

% Resize to original dimensions
predicted_mask = imresize(mask_512, [H0, W0], 'nearest');
confidence_map = imresize(P_tissue, [H0, W0], 'bilinear');

tissue_pct = 100 * mean(predicted_mask(:));
fprintf('Tissue: %.2f%%\n', tissue_pct);

% Filtered M11 intensity (apply mask to normalized input at native size)
m11_filtered = m11_norm .* predicted_mask;

%% Visualize (clean, simplified)
figure('Position', [100, 100, 1500, 450]);

subplot(1, 3, 1);
imagesc(m11_norm); axis image off; colormap(gca, 'gray'); colorbar;
title('M11 Intensity (Normalized)');

subplot(1, 3, 2);
imagesc(predicted_mask); axis image off; colormap(gca, 'parula'); colorbar;
title('Mask');

subplot(1, 3, 3);
imagesc(m11_filtered, [0, 1]); axis image off; colormap(gca, 'gray'); colorbar;
title('M11 Intensity (Filtered)');

%% Save
[~, name, ~] = fileparts(sampleFile);
imwrite(uint8(predicted_mask) * 255, fullfile(outputDir, [name '_mask.png']));
imwrite(uint8(confidence_map * 255), fullfile(outputDir, [name '_confidence.png']));
imwrite(uint8(m11_filtered * 255), fullfile(outputDir, [name '_m11_filtered.png']));
saveas(gcf, fullfile(outputDir, [name '_panel.png']));
save(fullfile(outputDir, [name '_results.mat']), ...
    'predicted_mask', 'confidence_map', 'tissue_pct', 'P_tissue', 'm11_filtered');

fprintf('Results saved to: %s\n', outputDir);
