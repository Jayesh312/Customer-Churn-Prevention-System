# Customer Churn Prevention System

An end-to-end machine learning pipeline and interactive web application designed to predict, explain, and prevent customer churn. This system leverages advanced gradient boosting (LightGBM) for high-accuracy predictions and Game Theory (SHAP) to provide interpretable, real-time insights into why a specific customer is at risk.

**Live Application:** [View Streamlit Dashboard](https://customer-churn-prevention-system-nqvotioh3qjcijacv35idg.streamlit.app/)

---

## Repository Structure

| File Name | Description |
| :--- | :--- |
| `app.py` | Streamlit application interface, real-time prediction engine, and SHAP explanation visualizer. |
| `requirements.txt` | Python dependency configuration file for environment setup and deployment. |
| `Telco_Churn_Advanced_Pipeline.ipynb` | Jupyter Notebook with EDA, feature engineering, pipeline construction, and model tuning. |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Raw customer churn dataset used for model training and evaluation. |
| `README.md` | Project documentation, mathematical overviews, and setup instructions. |

---

## Mathematical and Statistical Foundations

### 1. Light Gradient Boosting Machine (LightGBM)
The primary classifier is a LightGBM model. Gradient boosting iteratively constructs decision trees to minimize residual errors from previous iterations.

At iteration $t$, the objective function to minimize using a second-order Taylor expansion is:
$$ \text{Obj}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t) $$

Where:
* $g_i = \partial_{\hat{y}_i^{(t-1)}} L(y_i, \hat{y}_i^{(t-1)})$ represents the first-order gradient of the loss function.
* $h_i = \partial_{\hat{y}_i^{(t-1)}}^2 L(y_i, \hat{y}_i^{(t-1)})$ represents the second-order Hessian.
* $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$ is the regularization term penalizing tree complexity with $T$ leaves and leaf weights $w$.

LightGBM utilizes **Gradient-based One-Side Sampling (GOSS)** to filter data instances based on gradients, retaining those with large gradients and randomly sampling those with small gradients to speed up execution without compromising statistical accuracy.

### 2. SHapley Additive exPlanations (SHAP)
Model interpretability is achieved via TreeSHAP, based on cooperative game theory. SHAP computes the exact marginal contribution of each feature across all possible feature combinations.

The SHAP value $\phi_i$ for feature $i$ is calculated as:
$$ \phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (n - |S| - 1)!}{n!} (v(S \cup \{i\}) - v(S)) $$

Where:
* $N$ is the complete set of features ($n = |N|$).
* $S$ is a feature subset excluding feature $i$.
* $v(S)$ is the model's output given feature subset $S$.

The sum of all feature SHAP values equals the total deviation of the individual prediction from the expected baseline model prediction.

### 3. Evaluation Metrics
Because customer churn datasets exhibit class imbalance, standard accuracy is an uninformative metric. The system prioritizes:

* **Precision ($P$):** Proportion of flagged churners who actually churned.
  $$ P = \frac{TP}{TP + FP} $$
* **Recall ($R$):** Proportion of actual churners correctly identified.
  $$ R = \frac{TP}{TP + FN} $$
* **PR-AUC (Precision-Recall Area Under Curve):** Measures trade-off across all probability thresholds, ensuring high recall on churners while minimizing false alarms.

---
