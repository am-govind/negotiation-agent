/**
 * Login Page — Authenticate admin and redirect to dashboard.
 */
import './styles/global.css';
import './styles/admin.css';

const form = document.getElementById('login-form');
const errorEl = document.getElementById('login-error');
const loginBtn = document.getElementById('login-btn');

// If already logged in, redirect to dashboard
if (localStorage.getItem('admin_token')) {
    window.location.href = '/admin.html';
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorEl.textContent = '';
    loginBtn.disabled = true;
    loginBtn.textContent = '⏳ Signing in...';

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        const resp = await fetch('/api/admin/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'Invalid credentials');
        }

        const data = await resp.json();
        localStorage.setItem('admin_token', data.token);
        localStorage.setItem('admin_user', data.username);
        window.location.href = '/admin.html';
    } catch (err) {
        errorEl.textContent = '❌ ' + err.message;
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = '🔓 Sign In';
    }
});
