import os

import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from fpdf import FPDF
from joblib import load

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load large trained model and encoders with new features using absolute paths
try:
    model = load(os.path.join(BASE_DIR, 'model_huge.joblib'))
    le_college = load(os.path.join(BASE_DIR, 'college_encoder_huge.joblib'))
    le_caste = load(os.path.join(BASE_DIR, 'caste_encoder_huge.joblib'))
    le_branch = load(os.path.join(BASE_DIR, 'branch_encoder_huge.joblib'))
    le_gender = load(os.path.join(BASE_DIR, 'gender_encoder_huge.joblib'))  # Using huge version
    le_quota = load(os.path.join(BASE_DIR, 'quota_encoder_huge.joblib'))   # Using huge version
    print("OK: Model and encoders loaded successfully.")
except Exception as e:
    print(f"Error loading model/encoders: {e}")
    model = None
    le_college = le_caste = le_branch = le_gender = le_quota = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model files not loaded on server.'}), 500
    
    try:
        data = request.json
        
        # 1. Clean and Parse Inputs from JS
        # We use .strip().upper() because your CSV used 'OPEN', 'CS', etc.
        percentage = float(data.get('percentage', 0))
        caste = str(data.get('caste', '')).strip().upper()
        branch = str(data.get('branch', '')).strip().upper()
        gender = str(data.get('gender', 'M')).strip().upper()
        quota = str(data.get('quota', 'MS')).strip().upper()
        
        # 2. Transform strings to codes using the LOADED encoders
        try:
            caste_code = le_caste.transform([caste])[0]
            branch_code = le_branch.transform([branch])[0]
            gender_code = le_gender.transform([gender])[0]
            quota_code = le_quota.transform([quota])[0]
        except ValueError as e:
            # This happens if 'caste' or 'branch' isn't in your training data
            print(f"Label Error: {e}")
            return jsonify({'error': f"Invalid selection: {str(e)}"}), 400

        # 3. Create the input array in the EXACT same order as training
        # Training order: [cutoff_percentage, caste_code, branch_code, gender_code, quota_code]
        X_input = np.array([[percentage, caste_code, branch_code, gender_code, quota_code]])
        
        # 4. Get probabilities
        probabilities = model.predict_proba(X_input)[0]
        
        # 5. Extract Top 10
        top_indices = np.argsort(probabilities)[-10:][::-1]
        recommendations = []
        
        # Column idx of predict_proba matches model.classes_[idx] (encoded college id), not raw index order
        classes = model.classes_
        for idx in top_indices:
            label = int(classes[idx])
            college_name = le_college.inverse_transform([label])[0]
            prob = float(probabilities[idx])
            # Four decimals so small top-10 values are not shown as 0.00%
            recommendations.append({
                'college': college_name,
                'probability': round(prob * 100, 4)
            })
            
        return jsonify({'recommendations': recommendations})

    except Exception as e:
        # Check your Python terminal for this output!
        print(f"Server-side Prediction Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


def _pdf_safe(text: str) -> str:
    """fpdf2 core Helvetica is Latin-1; strip/replace anything else."""
    if not text:
        return ''
    return str(text).encode('latin-1', 'replace').decode('latin-1')


def _build_recommendations_pdf(profile: dict, recommendations: list) -> bytes:
    """Server-side PDF (no browser CDN). fpdf2 core fonts = Latin-1."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(14, 14, 14)
    pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_w, 8, _pdf_safe('DSE admission predictor - recommendations'))
    pdf.ln(2)
    pdf.set_font('helvetica', '', 10)
    lines = [
        f"Diploma percentage: {profile.get('percentage', '')}%",
        f"Category: {profile.get('caste_label', profile.get('caste', ''))}",
        f"Branch: {profile.get('branch_label', profile.get('branch', ''))}",
        f"Gender: {profile.get('gender_label', profile.get('gender', ''))}",
        f"Quota: {profile.get('quota_label', profile.get('quota', ''))}",
    ]
    for line in lines:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, 6, _pdf_safe(line))
    pdf.ln(3)
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_w, 7, _pdf_safe('Top 10 colleges (model scores)'))
    pdf.ln(1)
    pdf.set_font('helvetica', '', 10)
    for i, rec in enumerate(recommendations, start=1):
        college = str(rec.get('college', ''))
        prob = rec.get('probability', '')
        block = f"{i}. {college} - {prob}% match"
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, 6, _pdf_safe(block))
        pdf.ln(1)
    pdf.ln(4)
    pdf.set_font('helvetica', 'I', 8)
    pdf.set_text_color(90, 90, 90)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        usable_w,
        5,
        _pdf_safe(
            'Disclaimer: Scores come from a machine learning model trained on synthetic data. '
            'They are not official CAP round cutoffs or admission guarantees.'
        ),
    )
    return bytes(pdf.output())


@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json(silent=True) or {}
        recs = data.get('recommendations')
        if not recs or not isinstance(recs, list):
            return jsonify({'error': 'No recommendations to export. Run Predict first.'}), 400
        profile = data.get('profile') or {}
        pdf_bytes = _build_recommendations_pdf(profile, recs)
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': 'attachment; filename=college_recommendations.pdf',
                'Cache-Control': 'no-store',
            },
        )
    except Exception as e:
        print(f'PDF export error: {e}')
        return jsonify({'error': 'Could not build PDF.'}), 500


if __name__ == '__main__':
    # Setting use_reloader=False prevents the double-loading crash
    app.run(debug=True, use_reloader=False, port=5000)

