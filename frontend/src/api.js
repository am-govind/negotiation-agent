/**
 * API Client — shared fetch wrapper with auth support.
 */
const API_BASE = '/api';

function getAuthHeader() {
    const token = localStorage.getItem('admin_token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function apiFetch(endpoint, options = {}, requireAuth = false) {
    const url = `${API_BASE}${endpoint}`;
    const headers = {
        'Content-Type': 'application/json',
        ...(requireAuth ? getAuthHeader() : {}),
        ...(options.headers || {}),
    };
    const config = { ...options, headers };
    const resp = await fetch(url, config);

    if (resp.status === 401 && requireAuth) {
        // Token expired or invalid — redirect to login
        localStorage.removeItem('admin_token');
        localStorage.removeItem('admin_user');
        window.location.href = '/login.html';
        throw new Error('Session expired. Please login again.');
    }

    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

// ── Public Endpoints ─────────────────────────────────────────
export async function getHealth() {
    return apiFetch('/health');
}

export async function getCategories() {
    return apiFetch('/categories');
}

export async function getStates() {
    return apiFetch('/states');
}

export async function startNegotiation(productCategory, customerState, sellerState) {
    return apiFetch('/negotiate/start', {
        method: 'POST',
        body: JSON.stringify({
            product_category: productCategory,
            customer_state: customerState,
            seller_state: sellerState,
        }),
    });
}

export async function sendMessage(sessionId, message) {
    return apiFetch('/negotiate/message', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, message }),
    });
}

export async function getSessionStatus(sessionId) {
    return apiFetch(`/negotiate/${sessionId}/status`);
}

// ── Admin Endpoints (require auth) ───────────────────────────
export async function getAdminSessions() {
    return apiFetch('/admin/sessions', {}, true);
}

export async function getAdminSession(sessionId) {
    return apiFetch(`/admin/sessions/${sessionId}`, {}, true);
}

export async function getAdminAnalytics() {
    return apiFetch('/admin/analytics', {}, true);
}

export async function getAdminConfig() {
    return apiFetch('/admin/config', {}, true);
}

export async function updateAdminConfig(config) {
    return apiFetch('/admin/config', {
        method: 'PUT',
        body: JSON.stringify(config),
    }, true);
}
