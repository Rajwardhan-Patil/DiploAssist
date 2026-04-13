import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from functools import wraps

import numpy as np
import pandas as pd
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

college_stats = None
college_segment_stats = {}
try:
    _stats_df = load(os.path.join(BASE_DIR, 'huge_colleges_data.pkl'))
    if not isinstance(_stats_df, pd.DataFrame):
        _stats_df = pd.DataFrame(_stats_df)
    required_cols = {'college_name', 'cutoff_percentage', 'caste', 'branch', 'gender', 'quota'}
    if required_cols.issubset(set(_stats_df.columns)):
        for c in ('caste', 'branch', 'gender', 'quota', 'college_name'):
            _stats_df[c] = _stats_df[c].astype(str).str.strip().str.upper()
        college_stats = (
            _stats_df.groupby('college_name')['cutoff_percentage']
            .agg(['mean', 'std', 'min', 'max', 'count'])
            .to_dict('index')
        )
        grp = (
            _stats_df.groupby(['college_name', 'caste', 'branch', 'gender', 'quota'])[
                'cutoff_percentage'
            ]
            .agg(['mean', 'std', 'min', 'max', 'count'])
            .reset_index()
        )
        for _, row in grp.iterrows():
            k = (
                row['college_name'],
                row['caste'],
                row['branch'],
                row['gender'],
                row['quota'],
            )
            college_segment_stats[k] = {
                'mean': float(row['mean']),
                'std': float(row['std']) if pd.notna(row['std']) else np.nan,
                'min': float(row['min']),
                'max': float(row['max']),
                'count': int(row['count']),
            }
        print('OK: Loaded cutoff stats for realism scoring.')
    else:
        print('Warning: huge_colleges_data.pkl missing required columns, using raw model only.')
except Exception as e:
    print(f'Warning: Could not load cutoff stats ({e}). Using raw model only.')


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
    return {'current_user': get_current_user(), 'current_year': datetime.now().year}


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
@login_required_page
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
    return render_template(
        'dashboard.html',
        prefs=_row_dict(prefs),
    )


@app.get('/api/me')
def api_me():
    u = get_current_user()
    if not u:
        return jsonify({'user': None})
    return jsonify({'user': {'id': u['id'], 'email': u['email'], 'full_name': u['full_name']}})


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


@app.route('/predict', methods=['POST'])
@login_required_api
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

        # 5. Apply realism factors from historical cutoff distributions.
        classes = model.classes_
        adjusted = np.zeros_like(probabilities, dtype=float)
        for idx, p in enumerate(probabilities):
            label = int(classes[idx])
            college_name = le_college.inverse_transform([label])[0]
            factor = _realism_factor(college_name, percentage, caste, branch, gender, quota)
            adjusted[idx] = float(p) * factor
        total_adj = float(adjusted.sum())
        if total_adj > 0:
            adjusted = adjusted / total_adj
        else:
            adjusted = probabilities

        # 6. Extract top 10 and normalize within shortlist for practical ranking.
        top_indices = np.argsort(adjusted)[-10:][::-1]
        top_total = float(adjusted[top_indices].sum())
        recommendations = []
        for idx in top_indices:
            label = int(classes[idx])
            college_name = le_college.inverse_transform([label])[0]
            score = float(adjusted[idx])
            shortlist_pct = (score / top_total * 100.0) if top_total > 0 else 0.0
            shortlist_pct = float(np.clip(shortlist_pct, 0.0, 99.0))
            recommendations.append(
                {
                    'college': college_name,
                    'probability': round(shortlist_pct, 2),
                }
            )

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


def _realism_factor(
    college_name: str, percentage: float, caste: str, branch: str, gender: str, quota: str
) -> float:
    if college_stats is None:
        return 1.0
    seg_key = (college_name.upper(), caste, branch, gender, quota)
    seg = college_segment_stats.get(seg_key)
    base = college_stats.get(college_name.upper())
    stats = seg if seg and seg.get('count', 0) >= 5 else base
    if not stats:
        return 1.0
    mu = float(stats.get('mean', percentage))
    sigma = float(stats.get('std', np.nan))
    sigma = sigma if np.isfinite(sigma) and sigma > 1.5 else 3.0
    min_cut = float(stats.get('min', mu - (2 * sigma)))
    max_cut = float(stats.get('max', mu + (2 * sigma)))
    # Smoothly penalize colleges with much higher expected cutoff than user percentage.
    z = (percentage - mu) / sigma
    factor = 0.2 + (1.0 / (1.0 + np.exp(-z)))
    if percentage < (min_cut - 3.0):
        factor *= 0.35
    elif percentage < (min_cut - 1.5):
        factor *= 0.65
    elif percentage > (max_cut + 4.0):
        factor *= 1.05
    return float(np.clip(factor, 0.05, 1.25))


def _build_recommendations_pdf(profile: dict, recommendations: list) -> bytes:
    """Server-side PDF (no browser CDN). fpdf2 core fonts = Latin-1."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(14, 14, 14)
    pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    # Simple text logo on top-right corner
    pdf.set_font('helvetica', 'B', 11)
    pdf.set_text_color(61, 139, 253)
    pdf.set_xy(pdf.w - pdf.r_margin - 36, 10)
    pdf.cell(36, 6, _pdf_safe('DiploAssist'), align='R')
    pdf.set_text_color(0, 0, 0)
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
