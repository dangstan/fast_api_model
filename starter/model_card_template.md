# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Daniel Benjamim created the model. It is a XGBoost Classifier using the default hyperparameters.

## Intended Use

This model should be used as a classification model on publicly available Census Bureau data to predict if someone's salary is below or over 50K.

## Training Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

The original data set has 32561 rows, but some outliers and rows with unidentifed classes values were removed, resulting in a dataset with 30162 rows. An oversampling was made to to adjust the imbalanced target class and a K-cross split was made, which divided the data into five equal-size sets.

To use the data for training, the numerical features were normalized, the categorical features were dummized and the target class was binarized.

## Evaluation Data

 To use the data for evaluation, the numerical features were normalized, the categorical features were dummized and the target class was binarized.

## Metrics

The model was evaluated using precision, recall and F1 score. The value is 0.8960.

## Ethical Considerations



## Caveats and Recommendations


