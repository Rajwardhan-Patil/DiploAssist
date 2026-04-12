let lastRecommendations = null;
let lastProfile = null;

const isLoggedIn = typeof window.__PREDICTOR_AUTH__ !== 'undefined' && window.__PREDICTOR_AUTH__ === true;

function applyDefaultProfileFromServer() {
    const el = document.getElementById('defaultProfileJson');
    if (!el) return;
    let data;
    try {
        data = JSON.parse(el.textContent);
    } catch {
        return;
    }
    if (data == null || typeof data !== 'object') return;
    if (data.percentage != null && document.getElementById('percentage')) {
        const p = Number(data.percentage);
        if (Number.isFinite(p)) {
            document.getElementById('percentage').value = p;
            const r = document.getElementById('percentageRange');
            if (r) r.value = String(Math.min(100, Math.max(50, p)));
        }
    }
    if (data.caste) document.getElementById('caste').value = data.caste;
    if (data.branch) document.getElementById('branch').value = data.branch;
    if (data.gender) document.getElementById('gender').value = data.gender;
    if (data.quota) document.getElementById('quota').value = data.quota;
}

const predictBtn = document.getElementById('predictBtn');
const predictBtnText = predictBtn.querySelector('.btn-predict__text');
const predictSpinner = predictBtn.querySelector('.btn-predict__spinner');
const percentageInput = document.getElementById('percentage');
const percentageRange = document.getElementById('percentageRange');
const pctReadout = document.getElementById('pctReadout');
const stepTracker = document.getElementById('stepTracker');

function showToast(message, isError) {
    const root = document.getElementById('toastRoot');
    if (!root) return;
    const t = document.createElement('div');
    t.className = 'app-toast' + (isError ? ' app-toast--error' : '');
    t.textContent = message;
    root.appendChild(t);
    requestAnimationFrame(() => t.classList.add('app-toast--visible'));
    setTimeout(() => {
        t.classList.remove('app-toast--visible');
        setTimeout(() => t.remove(), 280);
    }, 4200);
}

function setSteps(activeStep) {
    const items = stepTracker.querySelectorAll('.step-tracker__item');
    items.forEach((el) => {
        const n = parseInt(el.dataset.step, 10);
        el.classList.remove('step-tracker__item--active', 'step-tracker__item--done');
        if (n < activeStep) el.classList.add('step-tracker__item--done');
        if (n === activeStep) el.classList.add('step-tracker__item--active');
    });
}

function selectedLabel(selectId) {
    const sel = document.getElementById(selectId);
    if (!sel || sel.selectedIndex < 0) return '';
    return sel.options[sel.selectedIndex].text.trim();
}

function syncPctFromRange() {
    percentageInput.value = percentageRange.value;
    updateLivePreview();
}

function syncRangeFromInput() {
    let v = parseFloat(percentageInput.value);
    if (Number.isFinite(v)) {
        v = Math.min(100, Math.max(50, v));
        percentageRange.value = String(v);
        percentageInput.value = v;
    }
    updateLivePreview();
}

function updateLivePreview() {
    const raw = parseFloat(percentageInput.value);
    const pct = Number.isFinite(raw) ? raw : null;
    pctReadout.textContent = pct != null ? `${pct.toFixed(1)}%` : '—';

    const caste = document.getElementById('caste').value;
    const branch = document.getElementById('branch').value;
    const main = document.getElementById('previewMain');
    const chips = document.getElementById('previewChips');

    if (pct == null) {
        main.textContent = 'Set your diploma percentage with the slider or the number field.';
        chips.innerHTML = '';
        return;
    }
    if (!caste) {
        main.textContent = 'Select your reservation category to continue.';
        chips.innerHTML = '';
        return;
    }
    if (!branch) {
        main.textContent = 'Select your preferred branch to see a full preview.';
        chips.innerHTML = '';
        return;
    }

    main.textContent = `We will score colleges for a ${pct.toFixed(1)}% diploma, ${selectedLabel('caste')}, ${selectedLabel('branch')}.`;

    const parts = [
        selectedLabel('gender'),
        selectedLabel('quota'),
    ];
    chips.innerHTML = parts
        .filter(Boolean)
        .map((p) => `<li>${escapeHtml(p)}</li>`)
        .join('');
}

function getProfileForExport() {
    return {
        percentage: percentageInput.value,
        caste: document.getElementById('caste').value,
        branch: document.getElementById('branch').value,
        gender: document.getElementById('gender').value,
        quota: document.getElementById('quota').value,
        caste_label: selectedLabel('caste'),
        branch_label: selectedLabel('branch'),
        gender_label: selectedLabel('gender'),
        quota_label: selectedLabel('quota'),
    };
}

percentageRange.addEventListener('input', syncPctFromRange);
percentageInput.addEventListener('input', syncRangeFromInput);
['caste', 'branch', 'gender', 'quota'].forEach((id) => {
    document.getElementById(id).addEventListener('change', updateLivePreview);
});

applyDefaultProfileFromServer();
syncPctFromRange();
setSteps(1);

predictBtn.addEventListener('click', async function () {
    const caste = document.getElementById('caste').value;
    const branch = document.getElementById('branch').value;
    const gender = document.getElementById('gender').value || 'M';
    const quota = document.getElementById('quota').value || 'MS';
    const percentage = parseFloat(percentageInput.value);

    if (isNaN(percentage) || percentage < 50 || percentage > 100) {
        showToast('Enter a valid percentage between 50 and 100.', true);
        return;
    }
    if (!caste) {
        showToast('Please select your category.', true);
        return;
    }
    if (!branch) {
        showToast('Please select your preferred branch.', true);
        return;
    }

    setSteps(2);
    predictBtn.disabled = true;
    predictBtnText.textContent = 'Scoring colleges…';
    predictSpinner.classList.remove('d-none');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ percentage, caste, branch, gender, quota }),
        });

        let data;
        try {
            data = await response.json();
        } catch {
            showToast('Invalid server response. Is the app running?', true);
            return;
        }

        if (!response.ok) {
            showToast(data.error || 'Prediction failed.', true);
            setSteps(1);
            return;
        }
        if (!data.recommendations || !Array.isArray(data.recommendations)) {
            showToast('Unexpected response from server.', true);
            setSteps(1);
            return;
        }

        lastRecommendations = data.recommendations;
        lastProfile = getProfileForExport();
        displayResults(data.recommendations);
        setSteps(3);
        showToast('Here are your top 10 matches.');

        const remember = document.getElementById('rememberProfile');
        if (isLoggedIn && remember && remember.checked) {
            const p = parseFloat(percentageInput.value);
            fetch('/api/preferences', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    percentage: Number.isFinite(p) ? p : null,
                    caste: document.getElementById('caste').value,
                    branch: document.getElementById('branch').value,
                    gender: document.getElementById('gender').value,
                    quota: document.getElementById('quota').value,
                }),
            }).catch(() => {});
        }
    } catch (err) {
        console.error(err);
        showToast('Network error. Check the server.', true);
        setSteps(1);
    } finally {
        predictBtn.disabled = false;
        predictBtnText.textContent = 'Get top 10 colleges';
        predictSpinner.classList.add('d-none');
    }
});

function displayResults(recommendations) {
    const resultsDiv = document.getElementById('results');
    const collegeList = document.getElementById('collegeList');

    collegeList.innerHTML = '';

    recommendations.forEach((rec, index) => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center flex-wrap gap-2';
        li.style.animationDelay = `${index * 0.06}s`;
        const pctLabel = formatProbabilityPercent(rec.probability);
        li.innerHTML = `
            <div class="d-flex align-items-center flex-grow-1" style="min-width: 200px;">
                <span class="rank-pill">${index + 1}</span>
                <span><strong>${escapeHtml(rec.college)}</strong></span>
            </div>
            <span class="badge-prob">${escapeHtml(pctLabel)}</span>
        `;
        collegeList.appendChild(li);
    });

    resultsDiv.classList.remove('d-none');
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.onclick = exportToPDF;
    }
    const saveBtn = document.getElementById('saveHistoryBtn');
    if (saveBtn) {
        saveBtn.onclick = saveRunToHistory;
        saveBtn.disabled = false;
    }
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function saveRunToHistory() {
    if (!isLoggedIn || !lastRecommendations || !lastRecommendations.length) {
        showToast('Nothing to save.', true);
        return;
    }
    const saveBtn = document.getElementById('saveHistoryBtn');
    if (saveBtn) saveBtn.disabled = true;
    try {
        const res = await fetch('/api/save_prediction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                profile: lastProfile || getProfileForExport(),
                recommendations: lastRecommendations,
            }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            showToast(data.error || 'Could not save. Try signing in again.', true);
            return;
        }
        showToast('Saved to your dashboard history.');
    } catch (e) {
        console.error(e);
        showToast('Network error.', true);
    } finally {
        if (saveBtn) saveBtn.disabled = false;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatProbabilityPercent(p) {
    const n = Number(p);
    if (!Number.isFinite(n) || n <= 0) {
        return '0% match';
    }
    if (n >= 0.01) {
        return `${n.toFixed(2)}% match`;
    }
    return `${n.toFixed(4)}% match`;
}

async function exportToPDF() {
    if (!lastRecommendations || !lastRecommendations.length) {
        showToast('Run a prediction before exporting.', true);
        return;
    }
    const exportBtn = document.getElementById('exportBtn');
    exportBtn.disabled = true;
    try {
        const res = await fetch('/export_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                profile: lastProfile || getProfileForExport(),
                recommendations: lastRecommendations,
            }),
        });
        if (!res.ok) {
            let msg = 'PDF export failed.';
            try {
                const err = await res.json();
                if (err.error) msg = err.error;
            } catch {
                /* ignore */
            }
            showToast(msg, true);
            return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'college_recommendations.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        showToast('PDF downloaded successfully.');
    } catch (e) {
        console.error(e);
        showToast('Could not download PDF.', true);
    } finally {
        exportBtn.disabled = false;
    }
}
