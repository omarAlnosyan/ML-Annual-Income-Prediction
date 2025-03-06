# **Machine Learning Classifiers for Predicting Annual Income**

## 📌 **Project Overview**
This project aims to predict whether an individual's **annual income** exceeds **$50,000 (>50K) or not (≤50K)** using machine learning. We applied **Support Vector Machine (SVM), Decision Tree, and k-Nearest Neighbors (KNN)** classifiers to analyze the dataset and compare their performance. The main focus is to **handle class imbalance, optimize model performance, and reduce overfitting**.

## 📊 **Dataset Information**
We used the **UCI Census Income Dataset (1994 US Census)**, which contains **demographic and employment-related attributes** of individuals.

- **Training Data:** 32,560 samples  
- **Testing Data:** 16,281 samples  
- **Features:** 15 (Categorical & Numerical)  
- **Target Variable:** Binary classification (`≤50K` or `>50K`)

### **Feature Categories:**
- **Numerical:** Age, fnlwgt, Education-num, Capital-gain, Capital-loss, Hours-per-week
- **Categorical:** Workclass, Education, Marital-status, Occupation, Relationship, Race, Sex, Native-country

## ⚙️ **Installation & Setup**
Clone the repository and install the required dependencies:
```bash
# Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
```

## 🚀 **How to Run the Project**
Run the Jupyter Notebook to train and evaluate the models:
```bash
jupyter notebook Project_SVM_DT_KNN_adult_DB_01_2025.ipynb
```

Or, run the Python script (if available):
```bash
python main.py
```

## 📈 **Model Performance & Comparison**
The models were evaluated based on **Accuracy, Precision, Recall, and F1-score**.

| Model          | Accuracy | Precision | Recall | F1-Score |
|---------------|---------|-----------|--------|---------|
| **SVM**      | 84.4%   | 0.86      | 0.69   | 0.77    |
| **Decision Tree** | 84.5% | 0.86      | 0.70   | 0.77    |
| **KNN (k=7)**| 82.9%   | 0.66      | 0.58   | 0.61    |

### **Which Model is Best and Why?**
- **Decision Tree** performed the best with an accuracy of **84.5%** due to its ability to handle both categorical and numerical data effectively. It provides interpretability but is prone to **overfitting**.
- **SVM** had nearly the same accuracy (**84.4%**) but demonstrated better generalization, handling **non-linear decision boundaries** well using the **RBF kernel**.
- **KNN** was the weakest performer (**82.9%**) as it struggled with the large dataset size and class imbalance, making distance-based classification less effective.

## 🔄 **How We Handled Overfitting**
Overfitting occurs when a model performs well on training data but poorly on unseen test data. Here’s how we tackled it:

### **1️⃣ Decision Tree (Prone to Overfitting)**
✅ **Solution:**
- **Limited max depth to 3** → Prevents excessive branching and overfitting.
- **Minimum samples per leaf = 5** → Ensures that each node split is meaningful.

### **2️⃣ Support Vector Machine (SVM)**
✅ **Solution:**
- Used **regularization parameter (C = 1.0)** → Prevents overfitting by balancing margin size and misclassification.
- Applied **RBF kernel** → Captures non-linear relationships while preventing high variance.

### **3️⃣ k-Nearest Neighbors (KNN)**
✅ **Solution:**
- Used **k = 7** (instead of a smaller value) → Reduces sensitivity to noise and variance.
- Applied **feature scaling (MinMaxScaler)** → Ensures fair distance measurement between features.

### **4️⃣ General Techniques**
✅ **Cross-Validation (5-Fold CV)** → Ensures that models generalize well across multiple data splits.
✅ **Feature Scaling** → Prevents bias in models like KNN and SVM.
✅ **Train-Test Split (70%-30%)** → Helps evaluate the model on unseen data.

## 🔄 **Future Improvements**
- Use **SMOTE (Synthetic Minority Over-sampling Technique)** to balance classes.
- Optimize SVM and Decision Tree **hyperparameters**.
- Experiment with **Random Forest and Gradient Boosting** models.
- Implement **deep learning** approaches for better accuracy.


