# Student Performance Prediction (ML Mini Project)

A beginner-friendly Machine Learning mini project that predicts a student's likely **Pass/Fail** result from attendance, study hours, previous exam score, assignment score, and internal assessment score.

## Algorithm
**Random Forest Classifier** (an ensemble of decision trees).

## Project files
- `app.py` — Streamlit user interface and prediction
- `train_model.py` — trains, evaluates, and saves the model
- `data/student_performance.csv` — generated sample dataset (1,200 rows)
- `requirements.txt` — Python packages

## Run in VS Code (Windows)

Open a terminal in this project folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit will show a local URL (usually `http://localhost:8501`). Open it in your browser.

If `python` does not work, try `py` instead.

## Features
1. Attendance percentage
2. Study hours per day
3. Previous exam score
4. Assignment score
5. Internal assessment score

## ML workflow
1. Load CSV with Pandas
2. Split features (`X`) and target (`y`)
3. Use an 80/20 train-test split
4. Train Random Forest
5. Evaluate with accuracy, classification report, and confusion matrix
6. Save model using Joblib
7. Load the trained model in Streamlit for predictions

## Important accuracy note
The included CSV is **synthetic/generated for demonstration**. The achieved score is printed when you run `python train_model.py`; it is not guaranteed to be 80% or higher and should not be presented as a result from real student records. For a research/college submission, replace it with a documented real dataset and report the measured test metrics honestly.

## Future scope
- Predict grade bands (A/B/C) instead of only Pass/Fail
- Add charts for feature importance
- Use a real student dataset
- Save prediction history in a database
