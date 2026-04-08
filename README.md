# 🔐 CAPTCHA Digit Recognizer

A MATLAB-based machine learning system that automatically reads 4-digit CAPTCHA images using **SVM classifiers**, **FFT-based noise filtering**, and **HOG feature extraction**.

---

## 🧠 How It Works

```
Raw CAPTCHA Image
      │
      ▼
🔊 FFT Notch Filtering     → Removes periodic stripe noise in frequency domain
      │
      ▼
🖼️  Morphological Cleaning  → Binarizes, cleans, removes speckles
      │
      ▼
📐 Projection Deskewing    → Corrects image tilt using row-sum variance
      │
      ▼
✂️  Digit Segmentation      → Splits into 4 equal-width slots
      │
      ▼
📊 HOG Feature Extraction  → Extracts shape features per digit slot
      │
      ▼
🤖 SVM-ECOC Classification → Predicts digit (0–9) for each slot
      │
      ▼
   [d1, d2, d3, d4]
```

---

## 🗂️ Project Structure

```
captcha-digit-recognizer/
│
├── trainingdata.m          # Trains 4 slot-wise SVM(ECOC) classifiers
├── myclassifier.m          # Loads model & predicts [d1 d2 d3 d4]
├── FeatureExtraction.m     # Full image preprocessing pipeline
├── evaluate_classifier.m  # Evaluates accuracy on validation set
├── debugimage2.m           # Visual step-by-step pipeline debugger
├── digit_svm_model.mat     # Pre-trained SVM model (generated after training)
│
└── Train/
    ├── labels.txt          # Ground truth labels [img_id d1 d2 d3 d4]
    └── captcha_XXXX.png    # 1200+ CAPTCHA images
```

---

## ⚙️ Requirements

- MATLAB R2020a or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

---

## 🚀 Usage

### 1. Train the model
```matlab
MODEL = trainingdata();
```
> Trains on images 1–800, validates on 801–1200. Saves `digit_svm_model.mat`.

### 2. Classify a single CAPTCHA
```matlab
im = imread('Train/captcha_0001.png');
y = myclassifier(im);  % returns [d1 d2 d3 d4]
```

### 3. Evaluate accuracy
```matlab
% Simply run evaluate_classifier.m in MATLAB
```

### 4. Debug the pipeline visually
```matlab
debugimage2(26)                         % by image number
debugimage2('Train/captcha_0026.png')   % by full path
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Training set | Images 1 – 800 |
| Validation set | Images 801 – 1200 |
| Slot 1 accuracy | 95.50% |
| Slot 2 accuracy | 93.00% |
| Slot 3 accuracy | 93.50% |
| Slot 4 accuracy | 98.50% |

---

## 📝 Notes

- Handles both **3-digit** and **4-digit** CAPTCHAs — a leading `0` is automatically forced for 3-digit ones
- The classifier loads the pre-trained model from `digit_svm_model.mat` — **no retraining happens during evaluation**
- The visual debugger (`debugimage2.m`) shows all 14 intermediate stages of the pipeline

---

## 👤 Author

**Diyanshu Kundu**  
[GitHub](https://github.com/DiyanshuKundu)

**Venkatesh Ahouri
[GitHub](https://github.com/venkatesh-akhouri)
