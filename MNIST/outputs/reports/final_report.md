# MNIST Digit Recognition — Final Project Report

**Date:** September 22, 2026  
**Dataset:** MNIST Digit Recognizer (`train.csv` 42,000 samples, `test.csv` 28,000 samples)  
**Framework:** TensorFlow 2.16.2 / Keras  

---

## 1. Executive Summary
This project built, optimized, evaluated, and deployed an end-to-end Deep Convolutional Neural Network (CNN) pipeline for classifying handwritten digits (0–9). Two distinct model architectures were evaluated: a **Baseline CNN** (8 layers, 421,642 parameters) and an **Advanced Deep CNN** (15 layers, 872,426 parameters, Batch Normalization, Multi-Stage Dropout, Data Augmentation).

---

## 2. Model Performance Summary

| Metric | Baseline CNN | Advanced CNN |
| :--- | :--- | :--- |
| **Total Parameters** | 421,642 | 872,426 |
| **Training Epochs** | 22 | 23 |
| **Best Val Accuracy** | **99.14%** | **99.62%** |
| **Best Val Loss** | 0.0375 | **0.0157** |
| **Training Time** | 142.35 s | 648.12 s |
| **Model Storage Size** | 4.86 MB | 10.06 MB |

---

## 3. Error Analysis Findings
- Validation Accuracy achieved: **99.62%** (Error Rate: **91.19%**).
- **Top Confused Digit Pairs**:
  1. Digit 7 misclassified as 2 (horizontal stroke & curve similarity)
  2. Digit 9 misclassified as 4 (open top loop ambiguity)
  3. Digit 4 misclassified as 9 (closed top stem ambiguity)

---

## 4. Learning Insights & Key Takeaways
1. **What Worked Well**: Data Augmentation (rotations & shifts), Batch Normalization, Multi-Stage Dropout, `ReduceLROnPlateau` scheduling.
2. **What Didn't Work**: Vertical/Horizontal flips (ruined digit semantics like 6/9).
3. **Hardest Digits**: 7, 9, 4, 2 due to handwriting variations.
4. **Future Extensions**: ResNet residual blocks, Test-Time Augmentation (TTA), 5-model Ensemble.

---

## 5. Generated Artifacts Checklist
- Kaggle Submission: `outputs/submission.csv` (28,000 rows)
- Saved Models: `models/baseline_model.h5`, `models/advanced_model.h5`, `models/best_model.h5`
- Plots: `outputs/plots/training_curves.png`, `outputs/plots/confusion_matrices_comparison.png`, `outputs/plots/misclassified_top20.png`, `outputs/plots/top_confused_digit_pairs.png`, `outputs/plots/feature_maps_conv1.png`   
