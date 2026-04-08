function MODEL = trainingdata()
% trainingdata.m
% ------------------------------------------------------------
% Trains 4 slot-wise SVM(ECOC) classifiers using:
%   Train = first 800 images
%   Val   = next 400 images
%
% Saves: digit_svm_model.mat
%
% Expected Train/labels.txt format (numeric matrix):
%   [img_id  d1 d2 d3 d4]
%
% IMPORTANT:
% - No training should occur during evaluate_classifier.m

    clc;
    rng(0);
    tStart = tic;

    data = importdata('Train/labels.txt');
    img_nrs = data(:,1);
    Yall    = data(:,2:end);   % Nx4

    N = size(Yall,1);
    assert(N >= 1200, 'Expected at least 1200 labeled samples.');

    idxTrain = 1:800;
    idxVal   = 801:1200;

    params = []; % FeatureExtraction defaults

    % =========================================================
    % Determine feature length D (per-slot) robustly
    % =========================================================
    fprintf('Determining feature length...\n');

    D = [];
    for t = 1:min(30, numel(idxTrain))  % try a few samples in case some fail
        k0 = img_nrs(idxTrain(t));
        I0 = imread(sprintf('Train/captcha_%04d.png', k0));

        try
            [slotFeats0, info0] = try_call_featureextraction(I0, params);

            if isstruct(info0) && isfield(info0,'failed') && info0.failed
                continue;
            end

            slotFeats0 = normalize_slotfeats(slotFeats0);

            if ~isempty(slotFeats0) && iscell(slotFeats0) && numel(slotFeats0)==4
                D = numel(slotFeats0{1});
                break;
            end
        catch
            % try next
        end
    end

    if isempty(D)
        error('Could not determine feature length: FeatureExtraction failed on initial samples.');
    end

    fprintf('Feature length per slot: %d\n', D);

    % =========================================================
    % Collect per-slot training matrices
    % =========================================================
    Xpos = cell(1,4);
    Ypos = cell(1,4);
    for p=1:4
        Xpos{p} = zeros(numel(idxTrain), D, 'single');
        Ypos{p} = round(Yall(idxTrain,p));
    end

    fprintf('Building training features (1..800)...\n');

    skipped = 0;
    for ii = 1:numel(idxTrain)
        n = idxTrain(ii);
        k = img_nrs(n);
        imgPath = sprintf('Train/captcha_%04d.png', k);

        I = imread(imgPath);

        [slotFeats, info] = try_call_featureextraction(I, params);
        slotFeats = normalize_slotfeats(slotFeats);

        if isempty(slotFeats)
            skipped = skipped + 1;
            fprintf('  [SKIP] n=%d k=%d (%s): FeatureExtraction returned empty.\n', n, k, imgPath);
            continue;
        end

        if isstruct(info) && isfield(info,'failed') && info.failed
            skipped = skipped + 1;
            fprintf('  [SKIP] n=%d k=%d (%s): %s\n', n, k, imgPath, safe_reason(info));
            continue;
        end

        % write features
        for p=1:4
            fp = slotFeats{p};
            if numel(fp) ~= D
                skipped = skipped + 1;
                fprintf('  [SKIP] n=%d k=%d: slot %d featLen=%d (expected %d)\n', n, k, p, numel(fp), D);
                break;
            end
            Xpos{p}(ii,:) = single(fp(:).');
        end

        if mod(ii,50)==0
            fprintf('  done %d/800\n', ii);
        end
    end

    fprintf('Skipped (train) samples: %d\n', skipped);

    % =========================================================
    % Train SVM ECOC per slot
    % =========================================================
    svm = cell(1,4);

    tSVM = templateSVM( ...
        'KernelFunction','rbf', ...
        'KernelScale','auto', ...
        'Standardize',true, ...
        'BoxConstraint',1);

    fprintf('Training 4 ECOC models...\n');
    for p=1:4
        y = Ypos{p};
        y = round(y);

        svm{p} = fitcecoc( ...
            Xpos{p}, y, ...
            'Learners', tSVM, ...
            'Coding','onevsall', ...
            'ClassNames', unique(y));

        fprintf('  Slot %d classes: %s\n', p, mat2str(unique(y)'));
    end

    YpredTrain = zeros(numel(idxTrain),4);
    YtrueTrain = Yall(idxTrain,:);
    
    for p = 1:4
        YpredTrain(:,p) = predict(svm{p}, Xpos{p});
    end
    
    trainAcc = mean(sum(abs(YtrueTrain - YpredTrain),2)==0);
    fprintf('Training accuracy (exact CAPTCHA): %.4f\n', trainAcc);

    % =========================================================
    % Validation
    % =========================================================
    fprintf('Validating (801..1200)...\n');
    Ytrue = round(Yall(idxVal,:));
    Ypred = zeros(size(Ytrue));

    skippedVal = 0;

    for jj = 1:numel(idxVal)
        n = idxVal(jj);
        k = img_nrs(n);
        I = imread(sprintf('Train/captcha_%04d.png', k));

        [slotFeats, info] = try_call_featureextraction(I, params);
        slotFeats = normalize_slotfeats(slotFeats);

        if isempty(slotFeats) || (isstruct(info) && isfield(info,'failed') && info.failed)
            skippedVal = skippedVal + 1;
            continue;
        end

        for p=1:4
            Ypred(jj,p) = predict(svm{p}, slotFeats{p});
        end

        % If FeatureExtraction tells us it was a 3-digit captcha, force leading 0.
        if isstruct(info) && isfield(info,'isFour') && ~info.isFour
            Ypred(jj,1) = 0;
        end
    end

    exact = sum(abs(Ytrue - Ypred),2)==0;
    accExact = mean(exact);

    accSlot = mean(Ytrue==Ypred, 1);

    MODEL = struct();
    MODEL.svm = svm;
    MODEL.params = params;
    MODEL.meta = struct();
    MODEL.meta.trainRange = [1 800];
    MODEL.meta.valRange   = [801 1200];
    MODEL.meta.accExact   = accExact;
    MODEL.meta.accSlot    = accSlot;
    MODEL.meta.featureLenPerSlot = D;
    MODEL.meta.skippedTrain = skipped;
    MODEL.meta.skippedVal   = skippedVal;

    save('digit_svm_model.mat','MODEL');

    fprintf('\n==== VALIDATION RESULTS ====\n');
    fprintf('Exact captcha accuracy: %.4f\n', accExact);
    fprintf('Slot accuracies: [%.4f %.4f %.4f %.4f]\n', accSlot(1),accSlot(2),accSlot(3),accSlot(4));
    fprintf('Skipped val samples: %d\n', skippedVal);
    fprintf('Saved: digit_svm_model.mat\n');
    toc(tStart);
end

% ======================================================================
% Helpers
% ======================================================================

function [out, info] = try_call_featureextraction(I, params)
% Calls FeatureExtraction safely regardless of whether it returns:
%   (a) one output (vector or cell), or
%   (b) two outputs ([slotFeats, info])

    info = struct('failed',false);

    try
        % try 2 outputs first
        [out, info] = FeatureExtraction(I, params);
    catch
        try
            % fallback: single output
            out = FeatureExtraction(I, params);
        catch ME
            out = [];
            info = struct('failed',true,'reason',ME.message);
        end
    end
end

function slotFeats = normalize_slotfeats(feats)
% Ensures output is 1x4 cell array of row vectors
% - if feats is already cell(1,4): keep
% - if feats is a row vector: split into 4 equal parts
% - if feats empty: return []

    slotFeats = [];

    if isempty(feats)
        return;
    end

    if iscell(feats)
        if numel(feats) ~= 4
            return;
        end
        slotFeats = feats;
        return;
    end

    if isnumeric(feats)
        v = feats(:).';
        L = numel(v);
        if mod(L,4) ~= 0
            return;
        end
        D = L/4;
        slotFeats = cell(1,4);
        for p=1:4
            i1 = (p-1)*D + 1;
            i2 = p*D;
            slotFeats{p} = v(i1:i2);
        end
        return;
    end
end

function r = safe_reason(info)
    if isstruct(info) && isfield(info,'reason') && ~isempty(info.reason)
        r = info.reason;
    else
        r = 'unknown reason';
    end
end
