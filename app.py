import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from functools import wraps

import numpy as np
from flask import (
    Flask,
    Response,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from fpdf import FPDF
from joblib import load
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Deployments (e.g. Render persistent disk): set DIPLOASSIST_DATABASE_PATH=/path/to/diploassist.db
# or DIPLOASSIST_INSTANCE_DIR=/path/to/writable/dir for instance/diploassist.db under that dir.
_db_override = os.environ.get('DIPLOASSIST_DATABASE_PATH')
if _db_override:
    DATABASE_PATH = os.path.abspath(_db_override)
    INSTANCE_DIR = os.path.dirname(DATABASE_PATH)
else:
    INSTANCE_DIR = os.environ.get(
        'DIPLOASSIST_INSTANCE_DIR', os.path.join(BASE_DIR, 'instance')
    )
    INSTANCE_DIR = os.path.abspath(INSTANCE_DIR)
    DATABASE_PATH = os.path.join(INSTANCE_DIR, 'diploassist.db')


def _normalize_database_url(url: str) -> str:
    u = (url or '').strip()
    if u.startswith('postgres://'):
        u = 'postgresql://' + u[len('postgres://') :]
    return u


# Free hosted Postgres (e.g. Neon): set DATABASE_URL on Render. Omit for local SQLite.
DATABASE_URL = _normalize_database_url(os.environ.get('DATABASE_URL', ''))
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2

    _DB_INTEGRITY_ERRORS = (sqlite3.IntegrityError, psycopg2.IntegrityError)
else:
    _DB_INTEGRITY_ERRORS = (sqlite3.IntegrityError,)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sql_adapt(sql: str) -> str:
    if USE_POSTGRES:
        return sql.replace('?', '%s')
    return sql


def get_db():
    if 'db' not in g:
        if USE_POSTGRES:
            from psycopg2.extras import RealDictCursor

            g.db = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        else:
            os.makedirs(INSTANCE_DIR, exist_ok=True)
            g.db = sqlite3.connect(DATABASE_PATH)
            g.db.row_factory = sqlite3.Row
    return g.db


def db_execute(sql: str, params=None, *, commit: bool = False):
    params = params if params is not None else ()
    db = get_db()
    cur = db.cursor()
    cur.execute(_sql_adapt(sql), params)
    if commit:
        db.commit()
    return cur


def _row_dict(row):
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return {k: row[k] for k in row.keys()}


@app.teardown_appcontext
def close_db(_exc=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    if USE_POSTGRES:
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(320) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                percentage DOUBLE PRECISION,
                caste TEXT,
                branch TEXT,
                gender TEXT,
                quota TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                profile_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            'CREATE INDEX IF NOT EXISTS idx_runs_user ON prediction_runs (user_id, created_at DESC)'
        )
        conn.commit()
        conn.close()
    else:
        os.makedirs(INSTANCE_DIR, exist_ok=True)
        conn = sqlite3.connect(DATABASE_PATH)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                percentage REAL,
                caste TEXT,
                branch TEXT,
                gender TEXT,
                quota TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                profile_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_runs_user ON prediction_runs (user_id, created_at DESC);
            """
        )
        conn.commit()
        conn.close()


init_db()

if USE_POSTGRES:
    print('OK: Using PostgreSQL (DATABASE_URL).')
else:
    print('OK: Using SQLite at', DATABASE_PATH)


def get_current_user():
    uid = session.get('user_id')
    if not uid:
        return None
    cur = db_execute('SELECT id, email, full_name, created_at FROM users WHERE id = ?', (uid,))
    return _row_dict(cur.fetchone())


@app.context_processor
def inject_auth():
    return {'current_user': get_current_user()}


def login_required_page(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please sign in to continue.', 'error')
            return redirect(url_for('signin', next=request.path))
        return f(*args, **kwargs)

    return wrapped


def login_required_api(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            return jsonify({'error': 'Sign in required.'}), 401
        return f(*args, **kwargs)

    return wrapped


_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def _validate_signup(email: str, password: str, confirm: str, full_name: str) -> list[str]:
    errors = []
    if not email:
        errors.append('Email is required.')
    elif not _EMAIL_RE.match(email):
        errors.append('Enter a valid email address.')
    if not password or len(password) < 8:
        errors.append('Password must be at least 8 characters.')
    if password != confirm:
        errors.append('Passwords do not match.')
    if full_name and len(full_name) > 120:
        errors.append('Name is too long.')
    return errors


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/predictor')
def predictor():
    default_profile = None
    user = get_current_user()
    if user:
        cur = db_execute(
            'SELECT percentage, caste, branch, gender, quota FROM user_preferences WHERE user_id = ?',
            (user['id'],),
        )
        row = cur.fetchone()
        if row:
            default_profile = {
                'percentage': row['percentage'],
                'caste': row['caste'],
                'branch': row['branch'],
                'gender': row['gender'],
                'quota': row['quota'],
            }
    return render_template('index.html', default_profile=default_profile)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if get_current_user():
        return redirect(url_for('predictor'))
    if request.method == 'GET':
        return render_template('signup.html')
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''
    confirm = request.form.get('confirm_password') or ''
    full_name = (request.form.get('full_name') or '').strip()
    errors = _validate_signup(email, password, confirm, full_name)
    if errors:
        for err in errors:
            flash(err, 'error')
        return (
            render_template(
                'signup.html',
                form_email=email,
                form_full_name=full_name,
            ),
            400,
        )
    try:
        db_execute(
            'INSERT INTO users (email, password_hash, full_name, created_at) VALUES (?, ?, ?, ?)',
            (email, generate_password_hash(password), full_name or None, _utc_now_iso()),
            commit=True,
        )
    except _DB_INTEGRITY_ERRORS:
        flash('An account with this email already exists.', 'error')
        return (
            render_template(
                'signup.html',
                form_email=email,
                form_full_name=full_name,
            ),
            400,
        )
    cur = db_execute('SELECT id FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    session['user_id'] = row['id']
    session.permanent = True
    flash('Welcome! Your account is ready.', 'success')
    return redirect(url_for('predictor'))


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if get_current_user():
        return redirect(url_for('predictor'))
    next_url = request.args.get('next') or ''
    if request.method == 'GET':
        return render_template('signin.html', next_url=next_url)
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''
    next_url = request.form.get('next') or ''
    if not email or not password:
        flash('Enter your email and password.', 'error')
        return render_template('signin.html', form_email=email, next_url=next_url), 400
    cur = db_execute('SELECT id, password_hash FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    if not row or not check_password_hash(row['password_hash'], password):
        flash('Invalid email or password.', 'error')
        return render_template('signin.html', form_email=email, next_url=next_url), 400
    session['user_id'] = row['id']
    session.permanent = True
    flash('Signed in successfully.', 'success')
    if next_url and next_url.startswith('/') and not next_url.startswith('//'):
        return redirect(next_url)
    return redirect(url_for('predictor'))


@app.post('/logout')
def logout():
    session.clear()
    flash('You have been signed out.', 'success')
    return redirect(url_for('landing'))


@app.route('/dashboard')
@login_required_page
def dashboard():
    uid = session['user_id']
    cur_p = db_execute(
        'SELECT percentage, caste, branch, gender, quota, updated_at FROM user_preferences WHERE user_id = ?',
        (uid,),
    )
    prefs = cur_p.fetchone()
    cur_r = db_execute(
        """
        SELECT id, profile_json, recommendations_json, created_at
        FROM prediction_runs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 40
        """,
        (uid,),
    )
    runs = cur_r.fetchall()
    history = []
    for r in runs:
        try:
            profile = json.loads(r['profile_json'])
            recs = json.loads(r['recommendations_json'])
        except (json.JSONDecodeError, TypeError):
            continue
        history.append(
            {
                'id': r['id'],
                'created_at': r['created_at'],
                'profile': profile,
                'recommendations': recs,
                'top_college': recs[0]['college'] if recs else None,
            }
        )
    return render_template(
        'dashboard.html',
        prefs=_row_dict(prefs),
        history=history,
    )


@app.get('/api/me')
def api_me():
    u = get_current_user()
    if not u:
        return jsonify({'user': None})
    return jsonify({'user': {'id': u['id'], 'email': u['email'], 'full_name': u['full_name']}})


@app.post('/api/save_prediction')
@login_required_api
def api_save_prediction():
    data = request.get_json(silent=True) or {}
    profile = data.get('profile')
    recs = data.get('recommendations')
    if not isinstance(profile, dict) or not isinstance(recs, list) or not recs:
        return jsonify({'error': 'Missing profile or recommendations.'}), 400
    uid = session['user_id']
    db_execute(
        'INSERT INTO prediction_runs (user_id, profile_json, recommendations_json, created_at) VALUES (?, ?, ?, ?)',
        (uid, json.dumps(profile), json.dumps(recs), _utc_now_iso()),
        commit=True,
    )
    return jsonify({'ok': True})


@app.post('/api/preferences')
@login_required_api
def api_preferences():
    data = request.get_json(silent=True) or {}
    pct = data.get('percentage')
    try:
        pct_f = float(pct) if pct is not None else None
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid percentage.'}), 400
    caste = (data.get('caste') or '').strip().upper() or None
    branch = (data.get('branch') or '').strip().upper() or None
    gender = (data.get('gender') or '').strip().upper() or None
    quota = (data.get('quota') or '').strip().upper() or None
    if pct_f is not None and (pct_f < 50 or pct_f > 100):
        return jsonify({'error': 'Percentage must be between 50 and 100.'}), 400
    uid = session['user_id']
    ex = 'EXCLUDED' if USE_POSTGRES else 'excluded'
    db_execute(
        f"""
        INSERT INTO user_preferences (user_id, percentage, caste, branch, gender, quota, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            percentage = {ex}.percentage,
            caste = {ex}.caste,
            branch = {ex}.branch,
            gender = {ex}.gender,
            quota = {ex}.quota,
            updated_at = {ex}.updated_at
        """,
        (uid, pct_f, caste, branch, gender, quota, _utc_now_iso()),
        commit=True,
    )
    return jsonify({'ok': True})


@app.delete('/api/history/<int:run_id>')
@login_required_api
def api_delete_history(run_id):
    uid = session['user_id']
    cur = db_execute(
        'DELETE FROM prediction_runs WHERE id = ? AND user_id = ?', (run_id, uid), commit=True
    )
    if cur.rowcount == 0:
        return jsonify({'error': 'Not found.'}), 404
    return jsonify({'ok': True})


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
