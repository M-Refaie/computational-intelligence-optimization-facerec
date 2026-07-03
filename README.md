# Computational Intelligence — Optimization & Face Recognition

Two-milestone project from *Computational Intelligence* (CSE473), Ain Shams University: (1) nonlinear optimization algorithms implemented from scratch, and (2) a face-recognition system built on an encoder–decoder + classifier.

## Milestone 1 — Nonlinear System Optimization
Implemented and compared three optimizers **from scratch** on a nonlinear system, with **analytically derived** gradients and Hessian.
- **Gradient descent** (fixed step) · **Newton–Raphson** (gradient + Hessian) · **Steepest descent with line search**
- Validated convergence and robustness over **100 random initializations**; compared final objective values, speed, and stability.
- **Finding:** Newton–Raphson reached the lowest objective fastest; line-search GD was most robust to initial conditions.

**▶ Notebook (Colab):** https://colab.research.google.com/drive/1nTqGXxAT4VSQCnn3mwWAU7CM8y1rJNJc

## Milestone 2 — Face Recognition (Encoder–Decoder + Classifier)
- **Encoder–decoder autoencoder** learns compact latent face embeddings (MSE/BCE reconstruction, Adam, LR scheduling).
- **MLP classifier** on the embeddings for identity recognition (ReLU + dropout + softmax, cross-validated depth/width).
- **Preprocessing:** per-image / dataset normalization; **visualization** via **PCA (2D & 3D)** and **t-SNE** of the latent space, plus confusion-matrix analysis.
- Dataset: CASIA-WebFace / FRGC, 70/15/15 split.

**▶ Notebook:** [`face_recognition_autoencoder.ipynb`](face_recognition_autoencoder.ipynb) (in this repo) · [Kaggle](https://www.kaggle.com/code/mohamedrefaie26/test-1-m-1)

## Tech stack
Python · NumPy · TensorFlow / Keras · scikit-learn (PCA, metrics) · Matplotlib

## Credits
Team project at Ain Shams University (Computational Intelligence / CSE473): Mohamed Refaie, Fares Mohamed, and Khaled Amin.
