function y = myclassifier(I)
% myclassifier.m
% ------------------------------------------------------------
% Loads digit_svm_model.mat (pretrained) and predicts [d1 d2 d3 d4].
% Must NOT train anything.

    persistent MODEL
    if isempty(MODEL)
        S = load('digit_svm_model.mat','MODEL');
        MODEL = S.MODEL;
    end

    params = MODEL.params;

    % Get per-slot features robustly
    [slotFeats, info] = try_call_featureextraction(I, params);
    slotFeats = normalize_slotfeats(slotFeats);

    % Default output (safe)
    y = [0 0 0 0];

    if isempty(slotFeats)
        return;
    end

    % Predict each slot
    for p=1:4
        try
            y(p) = predict(MODEL.svm{p}, slotFeats{p});
        catch
            y(p) = 0;
        end
    end

    % If FeatureExtraction identified 3-digit captcha -> force leading 0
    if isstruct(info) && isfield(info,'isFour') && ~info.isFour
        y(1) = 0;
    end
end

% ======================================================================
% Local helpers
% ======================================================================

function [out, info] = try_call_featureextraction(I, params)
    info = struct('failed',false);
    try
        [out, info] = FeatureExtraction(I, params);
    catch
        try
            out = FeatureExtraction(I, params);
        catch ME
            out = [];
            info = struct('failed',true,'reason',ME.message);
        end
    end
end

function slotFeats = normalize_slotfeats(feats)
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
    end
end
