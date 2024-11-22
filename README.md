# Diabetes Progression Prediction Project

## Overview
This project aims to predict diabetes progression using clinical data from a dataset provided by Stanford University's Machine Learning Repository. We utilized both **Linear Regression** and **XGBoost** models to understand how patient features like **BMI**, **Blood Pressure**, and **Serum Measurements** impact disease progression.

The project was conducted as part of an academic course (CS675: Introduction to Data Science) and involved comprehensive **Exploratory Data Analysis (EDA)**, followed by building and evaluating regression models.

## Features & Tools Used
- **Dataset**: Diabetes dataset with 442 records and 10 baseline features, plus a progression target (`y`).
- **Models Used**: Linear Regression and XGBoost Regressor.
- **Libraries**: Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib.
- **Tasks**:
  - Predict `y` using individual features, pairs of features, and all features.
  - Compute **Training MSE** and **Validation MSE** for different training set sizes.
  - Compare results between **Linear Regression** and **XGBoost**.

## Project Structure
- **`data/`**: Contains the dataset (diabetes.data).
- **`notebooks/`**: Jupyter notebooks used for EDA and model building.
- **`src/`**: Python scripts for feature selection, regression modeling, and metrics calculation.
- **`plots/`**: Graphs generated during EDA and model evaluation.
- **`README.md`**: Project documentation (this file).

## Installation & Setup
1. Clone this repository to your local machine:
   ```sh
   git clone https://github.com/username/diabetes-progression-prediction.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Start Jupyter Notebook for exploration:
   ```sh
   jupyter notebook
   ```

## How It Works
1. **Data Loading & Preprocessing**:
   - The dataset is loaded and cleaned for analysis, including handling missing values and outliers.
2. **Exploratory Data Analysis (EDA)**:
   - Visualization of correlations between features using **scatter plots**, **pair plots**, and **heatmaps**.
3. **Modeling**:
   - **Linear Regression** and **XGBoost Regressor** models were trained and evaluated for predicting diabetes progression.
4. **Performance Metrics**:
   - The models' performance was measured using **Mean Squared Error (MSE)**, and results were visualized for easy comparison.

## Key Findings
- **Feature Importance**: The features **BMI** and **S5** (serum level) were found to be most significant in predicting disease progression.
- **Model Comparison**: While **XGBoost** generally produced lower MSE values, it often indicated signs of overfitting. **Linear Regression** showed better generalization but slightly higher MSE values.
- **Training Size Analysis**: Validation MSE stabilized as training size increased for both models, with **Linear Regression** showing smoother convergence.

## Usage
To experiment with the models or run your own analyses, use the Jupyter notebooks provided in the `notebooks/` folder. Modify the parameters to test different feature sets or adjust the model hyperparameters.

## Results
- **Linear Regression**:
  - More interpretable and generalizable, but had higher error compared to XGBoost.
  - Suitable for scenarios where model transparency is important.
- **XGBoost**:
  - Lower MSE but demonstrated potential overfitting.
  - Useful when maximizing predictive accuracy, but caution is needed for generalization.

## Conclusion
This project highlights the trade-off between **interpretability** (Linear Regression) and **predictive performance** (XGBoost). The choice between models depends on the specific requirements, such as the need for transparency versus accuracy.

## Future Work
- **Hyperparameter Tuning**: Use grid search or random search to fine-tune the **XGBoost** model.
- **Cross-Validation**: Implement **k-fold cross-validation** for a more robust performance estimate.
- **Feature Engineering**: Investigate additional feature transformations or scaling techniques.

## References
- **Dataset Source**: [Stanford Machine Learning Repository](https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data)
- **Least Angle Regression Paper**: Bradley Efron, Trevor Hastie, Iain Johnstone, and Robert Tibshirani (2004), "Least Angle Regression."

## Contact
For any questions or feedback, please reach out to me on my github :)

