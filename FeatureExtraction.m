function K = FeatureExtraction(I, params)
% FEATUREEXTRACTION
% ------------------------------------------------------------
% PURPOSE
%   Extract a fixed-length feature vector from a CAPTCHA image for digit
%   recognition. This function performs ONLY preprocessing + segmentation +
%   feature extraction (no training). It is safe to call inside both
%   trainingdata.m and evaluate_classifier.m.
%
% OUTPUT
%   K : 1 x (4*D) single row vector
%       K = [HOG(slot1)  HOG(slot2)  HOG(slot3)  HOG(slot4)]
%       where each slot has identical feature length D (fixed by HOG params).
%       If extraction fails at any stage, K = [] is returned (caller may skip).
%
% PIPELINE (MAIN LOGIC)
%   Stage 1 — Grayscale + normalize
%     - Convert RGB -> Gray (if needed) and convert to double in [0,1].
%
%   Stage 2 — Frequency-domain stripe suppression (FFT notch filtering)
%     - Compute 2D FFT and shift to center (fft2 + fftshift).
%     - Detect the strongest off-center peak in an annulus [r_min, r_max]
%       and its symmetric counterpart around the spectrum center.
%     - Optionally detect a second symmetric peak pair along the same line
%       direction (captures multiple stripe harmonics when present).
%     - Build a notch mask: remove small disks (radius = notchRadius) around
%       the detected peaks, and apply it in the frequency domain.
%
%   Stage 3 — Reconstruct filtered image (iFFT)
%     - Apply inverse shift + inverse FFT to get the stripe-suppressed image.
%     - Normalize with mat2gray for stable downstream thresholding.
%
%   Stage 4 — Binarization + morphology cleanup (digits foreground)
%     - Light Gaussian blur (sigma) to stabilize thresholding.
%     - Otsu threshold (graythresh/imbinarize) to get a binary image.
%     - Morph cleanup: remove speckles (bwareaopen), close small gaps
%       (imclose), slightly dilate (imdilate) for connectivity.
%     - Invert so digits are foreground (~BW), remove small speckles again,
%       and close to consolidate digit strokes.
%
%   Stage 5 — Distance-transform seeding + mask consolidation
%     - Compute distance transform D = bwdist(~BW).
%     - Choose a seed threshold using a percentile of D on foreground pixels
%       (with a minimum floor seedMinT).
%     - Extract seed = (D >= t), remove tiny seed regions, then strengthen
%       connectivity using bwmorph operations and a final close.
%     - Resulting mask BWfull is a robust "digit-only" segmentation.
%
%   Stage 5b — Deskew (projection-based)
%     - Estimate a small rotation angle by scanning a range of angles and
%       selecting the one that maximizes horizontal projection variance.
%     - Rotate BWfull by the estimated angle (crop mode) and remove empty rows.
%
%   Stage 6 — Crop tight bbox + decide 3-digit vs 4-digit
%     - Find the tight bounding box of BWfull and crop.
%     - Decide digit count using cropped bbox width:
%         isFour = (bboxWidth > BBOXW_THRESH)
%         nDigits = 4 if isFour else 3
%
%   Stage 7 — Segmentation into slots (equal-width slicing)
%     - Split the cropped binary region into nDigits equal-width vertical slices.
%     - Trim each slice to its own tight content box (removes padding).
%     - Map into 4 output slots:
%         * If 4-digit: slots 1..4 are the four slices.
%         * If 3-digit: slot1 is blank; slots 2..4 are the three slices.
%       (Downstream logic forces leading digit = 0 for the 3-digit case.)
%
%   Stage 8 — Feature extraction per slot (HOG) + concatenation
%     - For each slot:
%         * If empty/blank: use an all-zero patch (hogSize) for stable length.
%         * Else: resize to hogSize and compute extractHOGFeatures with hogCellSize.
%     - Concatenate 4 HOG vectors into one row vector K (single precision).
%
% NOTES / GUARANTEES
%   - Always returns a fixed-length feature vector when successful.
%   - 3-digit CAPTCHAs are represented with a blank first slot (stable HOG);
%     myclassifier.m/trainingdata.m enforce y(1)=0 when isFour==false.
%   - No plotting, no printing; designed for fast batch evaluation.
%   - Uses default_params() when params is not provided or empty.

    K = [];

    try
        if nargin < 2 || isempty(params)
            params = default_params();
        end

        % -------------------------
        % Stage 1: Gray
        % -------------------------
        if size(I,3) == 3
            G = rgb2gray(I);
        else
            G = I;
        end
        G = im2double(G);

        % -------------------------
        % Stage 2: FFT + peaks + notches
        % -------------------------
        F  = fft2(G);
        Fs = fftshift(F);
        Mag = abs(Fs);

        [H,W] = size(G);
        cx = floor(W/2)+1;
        cy = floor(H/2)+1;

        [pA, pA_sym] = find_first_symmetric_pair(Mag, cx, cy, params.r_min, params.r_max);
        v = [pA(1)-cx, pA(2)-cy];

        [pB, pB_sym, foundSecond] = find_second_pair_on_line_robust( ...
            Mag, cx, cy, v, params.r_min, params.r_max, [pA; pA_sym], params.suppressR);

        if foundSecond
            peaks = [pA; pA_sym; pB; pB_sym];
        else
            peaks = [pA; pA_sym];
        end

        mask = true(H,W);
        mask = apply_notches(mask, peaks, params.notchRadius);

        % -------------------------
        % Stage 3: iFFT filtered
        % -------------------------
        Fs_filt = Fs .* mask;
        G_filt  = real(ifft2(ifftshift(Fs_filt)));
        G_filt  = mat2gray(G_filt);

        % -------------------------
        % Stage 4: blur + Otsu + morph
        % -------------------------
        G_blur = imgaussfilt(G_filt, params.sigma);

        level = graythresh(G_blur);
        BW = imbinarize(G_blur, level);

        BW = bwareaopen(BW, params.bw_areaopen1);
        BW = imclose(BW, strel('disk', params.close_disk1));
        BW = imdilate(BW, strel('disk', params.dilate_disk1));

        BW = ~BW;  % digits as foreground
        BW = bwareaopen(BW, params.minAreaSpeckle);
        BW = imclose(BW, strel('disk', params.close_disk2));

        % -------------------------
        % Stage 5: distance transform seed -> BWfull
        % -------------------------
        D   = bwdist(~BW);
        fgD = D(BW);
        if isempty(fgD)
            t = params.seedMinT;
        else
            t = max(params.seedMinT, prctile(fgD, params.seedPrctile));
        end

        seed = (D >= t);
        seed = bwareaopen(seed, params.seedAreaOpen);

        BWfull = seed;
        BWfull = bwmorph(BWfull,'bridge');
        BWfull = bwmorph(BWfull,'diag');
        BWfull = bwmorph(BWfull,'fill');
        BWfull = imclose(BWfull, strel('disk', params.seedCloseDisk));

        % -------------------------
        % Stage 5b: projection-based deskew
        % -------------------------
        angleDeg = estimate_skew_projection(BWfull, params);
        BWfull = imrotate(BWfull, angleDeg, 'bilinear', 'crop');

        % Optional cleanup: drop empty rows
        rows = any(BWfull,2);
        BWfull = BWfull(rows,:);

        % -------------------------
        % Stage 6: crop bbox + decide 3/4 digits
        % -------------------------
        [r,c] = find(BWfull);
        if isempty(r)
            K = [];
            return;
        end

        r1 = min(r); r2 = max(r);
        c1 = min(c); c2 = max(c);

        bboxW = c2 - c1 + 1;
        isFour = (bboxW > params.BBOXW_THRESH);
        nDigits = 3 + isFour;

        BWseg = BWfull(r1:r2, c1:c2);
        BWseg = bwareaopen(BWseg, params.cropAreaOpen);

        % -------------------------
        % Stage 7: equal-width slicing + slot mapping
        % -------------------------
        digitPatches = equal_width_slices(BWseg, nDigits);

        patches4 = cell(1,4);
        if isFour
            patches4 = digitPatches(1:4);
        else
            patches4{1} = [];              % slot1 blank (=> leading 0 in logic)
            patches4{2} = digitPatches{1};
            patches4{3} = digitPatches{2};
            patches4{4} = digitPatches{3};
        end

        % -------------------------
        % Stage 8: HOG per slot + concatenate
        % -------------------------
        featLen = get_featLen(params);
        K = zeros(1, 4*featLen, 'single');

        for s = 1:4
            if isempty(patches4{s}) || ~any(patches4{s}(:))
                patch = zeros(params.hogSize, 'single');  % stable blank
            else
                patch = patches4{s};
                patch = im2single(patch);
                patch = imresize(patch, params.hogSize, 'nearest');
            end

            hog = extractHOGFeatures(patch, 'CellSize', params.hogCellSize);
            hog = single(hog(:).');  % row

            i1 = (s-1)*featLen + 1;
            i2 = s*featLen;
            K(i1:i2) = hog;
        end

    catch
        % If anything blows up, return empty so training can skip this sample.
        K = [];
    end
end

% =======================================================================
% Local helpers 
% =======================================================================
function params = default_params()
    params = struct();

    % FFT peak detection
    params.r_min       = 4;
    params.r_max       = 100;
    params.suppressR   = 2;
    params.notchRadius = 2;

    % blur / morphology
    params.sigma         = 0.12;
    params.bw_areaopen1  = 25;
    params.close_disk1   = 1;
    params.dilate_disk1  = 1;

    params.minAreaSpeckle = 20;
    params.close_disk2    = 1;

    % distance transform seed
    params.seedPrctile    = 55;
    params.seedMinT       = 1.5;
    params.seedAreaOpen   = 40;
    params.seedCloseDisk  = 1;

    % bbox width rule
    params.BBOXW_THRESH = 240;
    params.cropAreaOpen = 30;

    % HOG ( style)
    params.hogSize     = [28 28];
    params.hogCellSize = [7 7];

    % projection-based deskew
    params.skew_angleRange = 15;    % degrees
    params.skew_angleStep  = 0.25;  % degrees
end

function digitPatches = equal_width_slices(BWseg, nDigits)
    BWseg = logical(BWseg);
    W = size(BWseg,2);

    edges = round(linspace(1, W+1, nDigits+1));
    digitPatches = cell(1,nDigits);

    for i = 1:nDigits
        xL = edges(i);
        xR = edges(i+1)-1;

        xL = max(1, min(W, xL));
        xR = max(1, min(W, xR));
        if xR < xL, xR = xL; end

        chunk = BWseg(:, xL:xR);
        digitPatches{i} = trimBinary(chunk);
    end
end

function featLen = get_featLen(params)
    dummy = zeros(params.hogSize, 'single');
    f = extractHOGFeatures(dummy, 'CellSize', params.hogCellSize);
    featLen = numel(f);
end

function [p1, p1sym] = find_first_symmetric_pair(Mag, cx, cy, r_min, r_max)
    [H,W] = size(Mag);
    [X,Y] = meshgrid(1:W, 1:H);
    R = sqrt((X-cx).^2 + (Y-cy).^2);

    mask = (R >= r_min) & (R <= r_max);
    mask(cy, cx) = false;

    M = Mag;
    M(~mask) = -Inf;

    [~, idx] = max(M(:));
    [yy, xx] = ind2sub([H,W], idx);

    p1 = [xx yy];
    x2 = 2*cx - xx;
    y2 = 2*cy - yy;

    x2 = min(max(x2,1), W);
    y2 = min(max(y2,1), H);

    p1sym = [x2 y2];
end

function [p2, p2sym, found] = find_second_pair_on_line_robust(Mag, cx, cy, v, r_min, r_max, excludePts, suppressR)
    [H,W] = size(Mag);

    found = false;
    p2 = [NaN NaN];
    p2sym = [NaN NaN];

    if all(v==0)
        return;
    end

    v = v / norm(v);

    r_try = r_max;
    while r_try <= max(r_max, 140)
        candX = [];
        candY = [];

        for sgn = [-1 1]
            for rr = r_min:r_try
                x = round(cx + sgn*rr*v(1));
                y = round(cy + sgn*rr*v(2));
                if x>=1 && x<=W && y>=1 && y<=H
                    candX(end+1) = x; %#ok<AGROW>
                    candY(end+1) = y; %#ok<AGROW>
                end
            end
        end

        if isempty(candX)
            r_try = r_try + 20;
            continue;
        end

        scores = Mag(sub2ind([H,W], candY, candX));

        for i = 1:size(excludePts,1)
            ex = excludePts(i,1); ey = excludePts(i,2);
            bad = (abs(candX-ex) <= suppressR) & (abs(candY-ey) <= suppressR);
            scores(bad) = -Inf;
        end

        [m, ii] = max(scores);
        if ~isempty(ii) && ~isinf(m)
            p2 = [candX(ii) candY(ii)];
            x2s = 2*cx - p2(1);
            y2s = 2*cy - p2(2);
            x2s = min(max(x2s,1),W);
            y2s = min(max(y2s,1),H);
            p2sym = [x2s y2s];
            found = true;
            return;
        end

        r_try = r_try + 20;
    end
end

function mask = apply_notches(mask, peaks, radius)
    [H,W] = size(mask);
    [X,Y] = meshgrid(1:W,1:H);
    for k = 1:size(peaks,1)
        x = peaks(k,1); y = peaks(k,2);
        disk = (X-x).^2 + (Y-y).^2 <= radius^2;
        mask(disk) = 0;
    end
end

function angleDeg = estimate_skew_projection(BW, params)
% Robust skew estimation: maximize variance of horizontal projection
    angleRange = params.skew_angleRange;
    angleStep  = params.skew_angleStep;

    angles = -angleRange:angleStep:angleRange;
    scores = zeros(size(angles));

    BW = logical(BW);

    for k = 1:numel(angles)
        BWrot = imrotate(BW, angles(k), 'bilinear', 'crop');
        proj  = sum(BWrot, 2);
        scores(k) = var(proj);
    end

    [~, idx] = max(scores);
    angleDeg = angles(idx);
end

function B = trimBinary(B)
    if isempty(B) || ~any(B(:)), return; end
    rows = any(B,2);
    cols = any(B,1);
    B = B(rows, cols);
end
