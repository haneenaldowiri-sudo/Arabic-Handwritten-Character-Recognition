# Cross-Dataset Evaluation and Fine-Tuning of CNNs for Arabic Handwritten Character Recognition

CAI 421 Deep Learning, section 53U, Princess Nourah bint Abdulrahman University.
Authors: Renad Alghazi, Ghala Khalid, Hanin Aldowiri, Sara Alsuwailem.

We train two CNNs (a baseline and a proposed CNN) on adult handwriting (AHCD), test them on children's
handwriting (Hijja), and measure how much fine-tuning with 5%, 10% and 25% of the Hijja training set recovers.

## Files
- `Arabic_Handwritten_Character_Recognition.ipynb`: the full notebook (data loading, preprocessing, models, training, fine-tuning, error analysis).
- `requirements.txt`: dependencies.

## Data
- AHCD: Kaggle dataset `mloey1/ahcd1` (downloaded automatically with `kagglehub` in the notebook).
- Hijja: https://github.com/israksu/Hijja2 (cloned automatically in the notebook).

## How to reproduce
1. Open `Arabic_Handwritten_Character_Recognition.ipynb` in Google Colab and set Runtime > Change runtime type > T4 GPU.
2. Run all cells from top to bottom (Runtime > Run all). Total time is about 10 minutes.
3. Outputs are saved in the `processed/` folder: `results_zero_shot.csv`, `results_multiseed.csv`,
   `per_letter_accuracy.csv`, `confusion_matrix.png`, `per_letter_accuracy.png`.

## Settings
- Seeds: 42 (data splits, single runs), and 42, 43, 44 (multi-seed results).
- Training: Adam, learning rate 1e-3, batch size 64, up to 30 epochs, early stopping on validation loss (patience 5).
- Fine-tuning: all layers trainable, Adam, learning rate 1e-4, batch size 32, 20 epochs, stratified Hijja subsets.
- Splits: AHCD 12,096 train / 1,344 validation / 3,360 test; Hijja (isolated letters) 9,880 train / 2,471 test.

## Note on reproducibility
GPU training in TensorFlow is not fully deterministic, so single runs can differ by a few points
(especially the zero-shot accuracy on Hijja). Reported results are the mean and standard deviation over three seeds.
