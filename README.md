# CIFAR-10 with 500 Images: Augmentation and Regularization

A small CNN with an Inception-style module, trained from scratch on only 500 CIFAR-10 images, and tested on corrupted images. The goal is to see how much augmentation and regularization help when data is tiny and the test set looks different from the training set.

## Important note about the shifted test set

The assignment says `shifted_test` is provided, but no file was given. So I **made my own stand-in** from the clean CIFAR-10 test images, using a fixed seed. It applies:

- random rotation up to ±25°
- brightness and contrast changes
- blur and noise (each on about half the images)
- a random 16×16 block-out (about 25% of the image)
- for the "hidden" corruption, my own guess: grayscale or channel swap

The real hidden corruption is unknown, so my shifted-test numbers will not match the official file. Treat them as a relative comparison between methods, not as official scores. If the real file is available, replace `shifted_test_images` and `shifted_test_labels` in the notebook and rerun.

## Setup and rules followed

- No pretrained weights, trained from scratch
- At most 1,000,000 trainable parameters (the model is far below that, see `model.summary()`)
- At least one Inception-style module (parallel 1×1, 3×3, 5×5 branches with 1×1 reductions, plus a pooling branch)
- 500 training images (50 per class) split into 400 train / 100 validation, with a fixed seed
- All choices made on the validation split only
- Augmentation applied on the fly, to training data only (nothing is saved to disk)
- Max 60 epochs, fixed seeds, 3 seeds (1, 2, 3), reported as mean ± std

## What is in the notebook

| Part | What it does |
|---|---|
| A | Baseline with no augmentation, learning curves, diagnosis |
| B | Augmentation strategy and written answers |
| C | Regularization choices and where each one goes |
| D | Ablation table over 7 configurations, 3 seeds each |
| E | Per-class accuracy, confusion matrix, most confused classes, hidden-corruption test |
| F | Test-time augmentation (bonus, not done) |

Note: Part E question 3 (explaining 8 misclassified images) is written from the plot in the notebook.

## Results (mean ± std over 3 seeds, accuracy in %)

**Part A baseline (no augmentation)**

| Set | Accuracy |
|---|---|
| Train | 50.3 ± 2.4 |
| Validation | 32.7 ± 3.9 |
| Clean test | 33.7 ± 1.4 |
| Shifted test (my version) | 15.4 ± 0.9 |

The baseline is both overfitting (train much higher than val) and affected by covariate shift (clean test much higher than shifted test).

**Part D ablation**

| Configuration | Val | Clean test | Shifted test |
|---|---|---|---|
| No augmentation | 29.0 ± 5.0 | 32.1 ± 1.2 | 14.2 ± 0.7 |
| Geometric only | 31.3 ± 5.2 | 31.1 ± 3.1 | 13.5 ± 0.8 |
| Photometric only | 29.0 ± 1.4 | 33.2 ± 0.6 | 18.2 ± 0.6 |
| Geometric + photometric | 28.3 ± 4.8 | 29.7 ± 1.7 | 17.3 ± 1.7 |
| + Cutout | 24.7 ± 2.5 | 28.6 ± 1.5 | 17.7 ± 0.9 |
| + Mixup + label smoothing | 23.7 ± 3.8 | 26.4 ± 2.3 | 17.9 ± 0.5 |
| Final model | 28.0 ± 9.9 | 31.0 ± 10.5 | 20.1 ± 5.6 |

Photometric augmentation gave the clearest gain (shifted test 14.2 to 18.2). The final model has the best mean (20.1) but a large standard deviation, so its gain over photometric-only is not clearly real. The first six rows use a plain model; the final row adds batchnorm, dropout 0.3, weight decay, cosine learning-rate schedule and early stopping.

**Part E (best final-model seed, chosen by validation accuracy)**

- Per-class accuracy on shifted test ranges from 10.5% (horse) to 40.3% (bird). The model sends many classes (airplane, horse, ship) to "bird".
- Hidden-corruption stand-in (clean test images): final model 38.7% clean, 29.2% grayscale, 23.9% channel swap. No-augmentation model: 34.1% clean, 26.9% grayscale, 16.8% channel swap.

Standard deviations in this study are large because validation is only 100 images, so small differences should be read as hints, not proof.

## How to reproduce

1. Open the notebook on Kaggle (GPU recommended, there are 21 training runs in Part D) or locally with Jupyter.
2. Install the packages if needed:
   ```
   pip install numpy matplotlib scikit-learn scipy tensorflow
   ```
3. Run the cells top to bottom, in order. CIFAR-10 downloads automatically through `keras.datasets.cifar10` (about 170 MB, saved after the first time).
4. Run the shifted-test cell **once**, before the clean test images are divided by 255. It saves `shifted_test_images.npy` and `shifted_test_labels.npy` so it does not need to be regenerated.
5. Run Part D and Part E after the model and augmentation cells.

All random seeds are fixed, but results can still differ slightly between machines and TensorFlow versions.

## Files

- `notebook.ipynb`: all parts with outputs (rename to match your actual file)
- `README.md`: this file
