# 🚗 Used Car Price Prediction

A machine learning project to predict the price of used cars based on structured datasets. Multiple models were trained and compared, with a final blended model selected for submission.

This is a machine learning project that I had to do in the context of a "Business Analytics and Data Science" course. The goal of this project was to train 3 different models on a train.csv dataset, tune one of them and submit the final model on a private Kaggle competition (class competition) to see who's model would perform the best on unseen data. 

The fine-tuned Random Forest Regressor was retained for the Kaggle submission.

---

## 📂 Project Structure

├── data/                  # Cleaned and raw data (ignored from GitHub) \
├── Notebooks/             # Jupyter notebooks for each step \
├── Results/                # Trained models (.joblib files) \
├── Scripts/                   # Python scripts (e.g., processing.py) \
├── requirements.txt       # List of required Python packages \
├── README.md              # Project overview 

---

## 📊 Models Used
- Linear Regression
- Random Forest
- XGBoost
- Blender (ensemble of above)

---

## 🧠 Technologies
- Python
- pandas, NumPy, scikit-learn
- XGBoost, joblib
- Jupyter

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run the submission.ipynb notebook

3. Results are stored in the submission.csv file






