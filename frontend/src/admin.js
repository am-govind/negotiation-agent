/**
 * Admin Dashboard — analytics, sessions, config (auth-protected).
 */
import './styles/global.css';
import './styles/admin.css';
import { getAdminAnalytics, getAdminSessions, getAdminSession, getAdminConfig, updateAdminConfig } from './api.js';

// ── Auth Check ───────────────────────────────────────────────
const token = localStorage.getItem('admin_token');
if (!token) {
    window.location.href = '/login.html';
}

// ── DOM Elements ─────────────────────────────────────────────
const statTotal = document.getElementById('stat-total');
const statClosed = document.getElementById('stat-closed');
const statWinrate = document.getElementById('stat-winrate');
const statMargin = document.getElementById('stat-margin');
const statRevenue = document.getElementById('stat-revenue');
const statRounds = document.getElementById('stat-rounds');
const sessionsContainer = document.getElementById('sessions-container');
const saveConfigBtn = document.getElementById('save-config-btn');
const configStatus = document.getElementById('config-status');
const logoutBtn = document.getElementById('logout-btn');

// Show admin username
const adminUser = localStorage.getItem('admin_user') || 'Admin';
const adminNameEl = document.getElementById('admin-name');
if (adminNameEl) adminNameEl.textContent = adminUser;

// ── Navigation & Logout ──────────────────────────────────────
const backLinks = document.querySelectorAll('.back-link');

function executeLogout() {
    localStorage.removeItem('admin_token');
    localStorage.removeItem('admin_user');
}

if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
        executeLogout();
        window.location.href = '/login.html';
    });
}

// Auto-logout when going back to chat
backLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        executeLogout();
        // Allow the default link navigation to proceed to '/'
    });
});

// ── Load Analytics ───────────────────────────────────────────
async function loadAnalytics() {
    try {
        const data = await getAdminAnalytics();

        statTotal.textContent = data.total_negotiations;
        statClosed.textContent = data.deals_closed;
        statWinrate.textContent = data.win_rate_pct + '%';
        statMargin.textContent = data.avg_margin_retained_pct + '%';
        statRevenue.textContent = '$' + data.total_revenue.toLocaleString();
        statRounds.textContent = data.avg_rounds;

        if (data.sessions && data.sessions.length > 0) {
            renderSessionsTable(data.sessions);
            renderCharts(data.sessions);
        }
    } catch (err) {
        console.error('Failed to load analytics:', err);
        if (err.message.includes('Session expired')) return;
        statTotal.textContent = '⚠️';
    }
}

function renderSessionsTable(sessions) {
    const table = document.createElement('table');
    table.className = 'sessions-table';
    table.innerHTML = `
    <thead>
      <tr>
        <th>Status</th>
        <th>Category</th>
        <th>Final Price</th>
        <th>Rounds</th>
        <th>Outcome</th>
        <th>Time</th>
      </tr>
    </thead>
    <tbody>
      ${sessions.map(s => `
        <tr style="cursor: pointer" onclick="window.viewSession('${s.session_id}')" title="Click to view chat history">
          <td><span class="status-dot ${s.outcome}"></span></td>
          <td>${(s.category || '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
          <td><span class="price-badge" style="font-size:12px;padding:2px 10px">$${(s.final_price || 0).toFixed(2)}</span></td>
          <td>${s.rounds || 0}</td>
          <td style="text-transform:capitalize">${s.outcome || '—'}</td>
          <td>${s.timestamp ? new Date(s.timestamp).toLocaleString() : '—'}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
    sessionsContainer.innerHTML = '';
    sessionsContainer.appendChild(table);
}

function renderCharts(sessions) {
    // Warm color palette for charts
    const revenueCtx = document.getElementById('revenue-chart');
    if (revenueCtx && sessions.length > 0) {
        const closedSessions = sessions.filter(s => s.outcome === 'closed');
        new Chart(revenueCtx, {
            type: 'bar',
            data: {
                labels: closedSessions.map((_, i) => `Deal ${i + 1}`),
                datasets: [{
                    label: 'Revenue ($)',
                    data: closedSessions.map(s => s.final_price || 0),
                    backgroundColor: 'rgba(255, 179, 71, 0.6)',
                    borderColor: '#ffb347',
                    borderWidth: 1,
                    borderRadius: 8,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#6e6e85' }, grid: { display: false } },
                    y: { ticks: { color: '#6e6e85', callback: v => '$' + v }, grid: { color: 'rgba(255,255,255,0.04)' } },
                },
            },
        });
    }

    // Outcome doughnut with vibrant colors
    const outcomeCtx = document.getElementById('outcome-chart');
    if (outcomeCtx && sessions.length > 0) {
        const counts = {};
        sessions.forEach(s => {
            const o = s.outcome || 'unknown';
            counts[o] = (counts[o] || 0) + 1;
        });
        const colorMap = {
            closed: '#69f0ae',
            abandoned: '#ff6b6b',
            active: '#ffb347',
            unknown: '#6e6e85'
        };
        new Chart(outcomeCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(counts).map(k => k.charAt(0).toUpperCase() + k.slice(1)),
                datasets: [{
                    data: Object.values(counts),
                    backgroundColor: Object.keys(counts).map(k => colorMap[k] || '#6e6e85'),
                    borderWidth: 0,
                }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#b0b0c8', padding: 16 } },
                },
            },
        });
    }
}

// ── Load Config ──────────────────────────────────────────────
async function loadConfig() {
    try {
        const config = await getAdminConfig();
        document.getElementById('cfg-floor').value = config.floor_price_discount;
        document.getElementById('cfg-markup').value = config.opening_markup;
        document.getElementById('cfg-rounds').value = config.max_rounds;
        document.getElementById('cfg-margin').value = config.min_profit_margin_pct;
    } catch (err) {
        console.error('Failed to load config:', err);
    }
}

saveConfigBtn.addEventListener('click', async () => {
    try {
        saveConfigBtn.disabled = true;
        const config = {
            floor_price_discount: parseFloat(document.getElementById('cfg-floor').value),
            opening_markup: parseFloat(document.getElementById('cfg-markup').value),
            max_rounds: parseInt(document.getElementById('cfg-rounds').value),
            min_profit_margin_pct: parseFloat(document.getElementById('cfg-margin').value),
        };
        await updateAdminConfig(config);
        configStatus.textContent = '✅ Saved!';
        setTimeout(() => { configStatus.textContent = ''; }, 3000);
    } catch (err) {
        configStatus.textContent = '❌ ' + err.message;
    } finally {
        saveConfigBtn.disabled = false;
    }
});

// ── Modal Logic ──────────────────────────────────────────────
window.viewSession = async function (sessionId) {
    const modal = document.getElementById('chat-modal');
    if (!modal) return;

    const msgContainer = document.getElementById('modal-messages');
    const sidLabel = document.getElementById('modal-session-id');

    sidLabel.textContent = `(${sessionId})`;
    msgContainer.innerHTML = '<div style="text-align:center; color:#888;">Loading...</div>';
    modal.style.display = 'flex';

    try {
        const data = await getAdminSession(sessionId);
        msgContainer.innerHTML = '';
        if (!data.messages || data.messages.length === 0) {
            msgContainer.innerHTML = '<div style="color:#aaa;">No messages in this session.</div>';
            return;
        }

        data.messages.forEach(m => {
            let roleClass = 'assistant';
            let roleName = 'Sales Agent';

            if (m.role === 'user' || m.role === 'human') {
                roleClass = 'user';
                roleName = 'Customer';
            } else if (m.role === 'tool' || m.role === 'system') {
                roleClass = 'tool';
                roleName = 'System Event';
            }
            const bubble = document.createElement('div');
            bubble.className = 'admin-msg-row';

            // Simple escape
            const div = document.createElement('div');
            div.textContent = m.content;
            const textContent = div.innerHTML.replace(/\n/g, '<br>');

            bubble.innerHTML = `
                <div class="admin-msg-role ${roleClass}">${roleName}</div>
                <div class="admin-msg-bubble">${textContent}</div>
            `;
            msgContainer.appendChild(bubble);
        });
    } catch (err) {
        msgContainer.innerHTML = `<div style="color:var(--coral)">Failed to load session: ${err.message}</div>`;
    }
};

document.getElementById('close-modal-btn')?.addEventListener('click', () => {
    document.getElementById('chat-modal').style.display = 'none';
});

// ── Boot ─────────────────────────────────────────────────────
loadAnalytics();
loadConfig();
