# Arabic Handwritten Character Recognition

## Project Overview

This project is developed for the CAI421 Deep Learning course at Princess Nourah bint Abdulrahman University.

The project investigates how well Convolutional Neural Networks (CNNs) trained on adult Arabic handwriting can recognize handwritten characters produced by children.

Two datasets are used:

- **AHCD:** Arabic Handwritten Characters Dataset, containing adult handwriting samples.
- **Hijja:** A dataset containing Arabic handwriting samples from children.

## Models

Two CNN architectures are implemented and compared.

**1. Baseline CNN**
- Two convolutional layers with ReLU activation
- Max-pooling layers
- Flatten layer
- Softmax classification layer

**2. Proposed CNN**
- Three convolutional layers with ReLU activation
- Batch normalization
- Max-pooling
- Global average pooling
- Dropout
- Softmax classification layer

Both models classify 28 Arabic character classes using grayscale images of size 32 × 32 pixels.

## Implemented Experiments

The notebook includes:

1. Downloading and preprocessing the AHCD and Hijja datasets.
2. Preparing training, validation, and testing data.
3. Training the baseline CNN and proposed CNN on AHCD.
4. Evaluating both models on the AHCD test set.
5. Evaluating both models on Hijja without fine-tuning.
6. Reporting accuracy, macro precision, macro recall, and macro F1-score.

## Planned Experiments

Future experiments will investigate fine-tuning the AHCD-trained model using 5%, 10%, and 25% of Hijja training data.

The objective is to measure how different amounts of children's handwriting data affect recognition performance.

These fine-tuning experiments are planned and are not included in the current notebook.

## Requirements

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- KaggleHub

## How to Run

1. Open `Checkpoint_2.ipynb` in Google Colab.
2. Enable a GPU runtime if available.
3. Run the notebook cells in order.
4. Allow the notebook to download AHCD and clone the Hijja repository.
5. Run preprocessing, training, and evaluation.
6. Review the generated evaluation results.

The notebook uses random seed 42 and saves trained models, processed data, and evaluation results in the `processed` directory.

## Dataset Sources

AHCD:
https://www.kaggle.com/datasets/mloey1/ahcd1

Hijja:
https://github.com/israksu/Hijja2

The datasets are obtained from their respective sources. Dataset licensing and redistribution conditions should be checked before distributing dataset files.

## Project Status

Checkpoint 2: Baseline and proposed CNN training, cross-dataset evaluation, and preliminary results.

Fine-tuning experiments and the final analysis remain future work.
Checkpoint 2: Baseline and proposed CNN training, cross-dataset evaluation, and preliminary results.

Fine-tuning experiments and the final analysis remain future work.
