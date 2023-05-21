# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model was developed by Daniel Benjamim using XGBoost Classifier with default hyperparameters. The final model was selected after performing grid search.

## Intended Use

This model is intended for classification tasks on publicly available Census Bureau data. Its purpose is to predict whether someone's salary is below or over 50K.

## Training Data

The training data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

The original dataset contained 32,561 rows, but outliers and rows with unidentified class values were removed, resulting in a dataset with 30,162 rows. To address the imbalanced target class, an oversampling technique was applied. The data was then divided into five equal-sized sets using K-fold cross-validation.

Before training, the numerical features were normalized, the categorical features were converted to dummy variables, and the target class was binarized.

## Evaluation Data

Similar to the training data, the evaluation data underwent preprocessing steps, including normalization of numerical features, conversion of categorical features to dummy variables, and binarization of the target class.

## Metrics

The model was evaluated using precision, recall, and F1 score. The obtained F1 score is 0.7127.

## Ethical Considerations

...

## Caveats and Recommendations

...