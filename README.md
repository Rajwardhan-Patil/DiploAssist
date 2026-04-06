# Diploma to Engineering DSE Admission Predictor

ML-powered web app to predict engineering colleges for Direct Second Year admission based on diploma percentage.

## Dataset
- `data/colleges.csv`: Sample cutoff data for ~25 Maharashtra colleges (fictional but realistic).
- Model trained with RandomForestClassifier (scikit-learn).

## Setup & Run (Windows)

1. **Virtual Environment**:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Train Model**:
   ```
   python train_model.py
   ```
   Outputs: model.joblib, label_encoder.joblib (accuracy ~0.8-1.0 on small dataset).

4. **Run Server**:
   ```
   python app.py
   ```

5. **Open Browser**:
   http://localhost:5000

## Usage
- Enter diploma % (50-100).
- Get top 5 college recommendations with match % (from model predict_proba).

## Model Details
- Input: percentage (single feature).
- Output: College classification probabilities.
- Top probs = likely eligible colleges (model learns cutoff boundaries).

## Files
- `app.py`: Flask backend.
- `train_model.py`: Trains/saves model.
- `templates/index.html`: Frontend.
- `static/*`: CSS/JS.
- Update `data/colleges.csv` for custom data, re-train.

Enjoy predicting your college admissions!
