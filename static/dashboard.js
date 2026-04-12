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

document.getElementById('prefsForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('prefsSaveBtn');
    const status = document.getElementById('prefsStatus');
    const pct = parseFloat(document.getElementById('dp_percentage').value);
    const body = {
        percentage: Number.isFinite(pct) ? pct : null,
        caste: document.getElementById('dp_caste').value,
        branch: document.getElementById('dp_branch').value,
        gender: document.getElementById('dp_gender').value,
        quota: document.getElementById('dp_quota').value,
    };
    btn.disabled = true;
    status.textContent = '';
    try {
        const res = await fetch('/api/preferences', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            showToast(data.error || 'Could not save preferences.', true);
            return;
        }
        status.textContent = 'Saved. The predictor will use these defaults on your next visit.';
        showToast('Default profile saved.');
    } catch (err) {
        console.error(err);
        showToast('Network error.', true);
    } finally {
        btn.disabled = false;
    }
});

document.querySelectorAll('.history-delete').forEach((btn) => {
    btn.addEventListener('click', async () => {
        const id = btn.getAttribute('data-run-id');
        if (!id || !confirm('Delete this saved run?')) return;
        btn.disabled = true;
        try {
            const res = await fetch(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                showToast(data.error || 'Could not delete.', true);
                btn.disabled = false;
                return;
            }
            const item = btn.closest('.accordion-item');
            if (item) item.remove();
            showToast('Run removed.');
            if (!document.querySelector('.dashboard-acc__item')) {
                window.location.reload();
            }
        } catch (err) {
            console.error(err);
            showToast('Network error.', true);
            btn.disabled = false;
        }
    });
});
