# Model Card — Breast Tumor Classifier

## Model Summary

| Field | Value |
|---|---|
| Task | Binary classification |
| Output classes | Benign and malignant |
| Current committed model | Logistic Regression |
| Input size | 30 numerical features |
| Training records | Derived from a 569-record dataset |
| Serialization | Python pickle |
| Application | Flask web application |

## Intended Use

This model is intended for:

- Machine-learning education
- Demonstrating a tabular classification workflow
- Demonstrating model serialization and Flask inference
- Portfolio and software-engineering review

It is not intended for:

- Clinical diagnosis
- Medical screening
- Treatment decisions
- Self-diagnosis
- Use as a regulated medical device
- Replacement of pathology, imaging, or professional medical evaluation

## Dataset

The repository contains a WDBC-format dataset with:

- 569 rows
- 30 numerical predictive features
- One binary diagnosis target
- One identifier column

The features describe cell-nucleus characteristics calculated from digitized images of fine-needle aspirates of breast masses.

Reference: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Recorded Evaluation

The committed notebook records:

| Metric | Result |
|---|---:|
| Training accuracy | 94.73% |
| Test accuracy | 92.98% |

These values came from one 80/20 split with `random_state=2`.

## Important Evaluation Gaps

The committed notebook does not record:

- Precision
- Recall or sensitivity
- Specificity
- F1 score
- ROC-AUC
- Confusion matrix
- Cross-validation statistics
- Confidence intervals
- Probability calibration
- Subgroup or fairness evaluation
- External validation

Accuracy alone is insufficient for assessing a medical classification model.

## Known Technical Limitations

1. The original notebook trains unscaled Logistic Regression and records a convergence warning.
2. The original notebook contains a machine-specific absolute dataset path.
3. The original model was evaluated on one split.
4. The dataset is small relative to real clinical deployment requirements.
5. The application accepts manually entered measurements and does not validate whether values are medically plausible.
6. Pickle artifacts should be loaded only from trusted sources.
7. The feedback database is local SQLite storage and may not persist on ephemeral hosting platforms.

## Recommended Next Evaluation

- Use stratified cross-validation
- Report class-wise recall and precision
- Prioritize malignant-class recall analysis
- Add a confusion matrix and ROC curve
- Evaluate probability calibration
- Compare Logistic Regression, SVM, Random Forest, and calibrated models
- Track experiments and model versions
- Validate on an external dataset
- Add input distribution and drift checks

## Ethical and Safety Notes

False negatives could create inappropriate reassurance, while false positives could create unnecessary distress. The interface must clearly state that predictions are educational and that medical decisions require qualified healthcare professionals.
