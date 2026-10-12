# Leaf Classification: Basil, Chinar and Lemon

Classifying leaf images into three plant types with classical machine learning on hand-crafted image features.

Originally an exercise for a Machine Learning and Pattern Recognition course at the University of Turku. This is my individual work.

## Overview

The notebook builds a full image-classification pipeline without deep learning: feature extraction, feature analysis, model selection and an honest performance estimate with nested cross-validation.

**Data:** 184 RGB leaf images from Kaggle (Basil 60, Chinar 62, Lemon 62). Dataset: a Kaggle leaf-image dataset

**Features (13 per image)**
- 6 first-order color features: mean and variance of each RGB channel
- 4 second-order texture features: GLCM correlation (horizontal and vertical, distances 1 and 2)
- 3 template-matching features: normalized cross-correlation against one representative template per class

**Analysis:** pairplot, per-class histograms and PCA, to see which features separate the classes.

**Models:** Ridge Classifier, Random Forest and MLP, with hyperparameters chosen by GridSearchCV using stratified K-fold.

## Results

Performance estimated with nested cross-validation (inner loop: 5-fold, outer loop: 4-fold, both stratified). Accuracy is the mean over the outer folds, with the standard deviation in brackets.

| Model | Accuracy |
|---|---|
| Ridge Classifier | 0.902 (± 0.024) |
| Random Forest | 0.897 (± 0.052) |
| MLP | 0.804 (± 0.058) |

- Ridge was the most accurate and the most stable. A simple linear model suits a dataset this small.
- Lemon was the hardest class for all models, and Chinar the easiest. This matches what PCA and the histograms suggested.
- Color means were the most informative features. GLCM texture features were weak on their own.

With only 184 images, these estimates are uncertain. The notebook discusses limitations and possible improvements in more detail.

## How to run

1. Download the dataset from Kaggle and place the three class folders (`Basil`, `Chinar`, `Lemon`) in the same directory as the notebook. The images are not included in this repository.
2. Install the dependencies:
3. Open and run `leaf_classification.ipynb`.

## Files

- `leaf_classification.ipynb`: the full analysis and report
- `requirements.txt`: Python dependencies