/**
 * Customer Chat App — Main entry point.
 */
import './styles/global.css';
import './styles/chat.css';
import { getCategories, getStates, startNegotiation, sendMessage } from './api.js';

// ── State ────────────────────────────────────────────────────
let sessionId = null;
let isNegotiating = false;

// ── DOM Elements ─────────────────────────────────────────────
const categorySelect = document.getElementById('category-select');
const buyerState = document.getElementById('buyer-state');
const sellerState = document.getElementById('seller-state');
const startBtn = document.getElementById('start-btn');
const resetBtn = document.getElementById('reset-btn');
const welcomeScreen = document.getElementById('welcome-screen');
const messagesEl = document.getElementById('messages');
const chatInputArea = document.getElementById('chat-input-area');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const dealBanner = document.getElementById('deal-banner');

// ── Initialize ───────────────────────────────────────────────
async function init() {
  try {
    const [catData, stateData] = await Promise.all([getCategories(), getStates()]);

    // Populate categories
    categorySelect.innerHTML = catData.categories.map(c =>
      `<option value="${c.id}">${c.name}</option>`
    ).join('');

    // Populate states
    const stateOptions = stateData.states.map(s =>
      `<option value="${s.code}">${s.name} (${s.code})</option>`
    ).join('');
    buyerState.innerHTML = stateOptions;
    sellerState.innerHTML = stateOptions;

    // Default to São Paulo
    buyerState.value = 'SP';
    sellerState.value = 'SP';
  } catch (err) {
    console.error('Init failed:', err);
    categorySelect.innerHTML = '<option>⚠️ API not available</option>';
  }
}

// ── Start Negotiation ────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  if (isNegotiating) return;
  startBtn.disabled = true;
  startBtn.textContent = '⏳ Starting...';

  try {
    const category = categorySelect.value;
    const buyer = buyerState.value;
    const seller = sellerState.value;

    const result = await startNegotiation(category, buyer, seller);
    sessionId = result.session_id;
    isNegotiating = true;

    // Show chat UI
    welcomeScreen.style.display = 'none';
    messagesEl.style.display = 'flex';
    chatInputArea.style.display = 'block';
    resetBtn.style.display = 'block';
    dealBanner.style.display = 'none';

    // Clear messages and add product image card
    messagesEl.innerHTML = '';

    if (result.image_url) {
      messagesEl.innerHTML += `
            <div class="product-image-card animate-in">
                <img src="${result.image_url}" alt="${result.product_category}" />
                <div class="product-image-title">Negotiating: <strong>${result.product_category.replace(/_/g, ' ')}</strong></div>
            </div>
        `;
    }

    addMessage('assistant', result.opening_message);

    // Disable setup
    categorySelect.disabled = true;
    buyerState.disabled = true;
    sellerState.disabled = true;

    chatInput.focus();
  } catch (err) {
    alert(`Failed to start: ${err.message}`);
  } finally {
    startBtn.disabled = false;
    startBtn.textContent = '🚀 Start Negotiation';
  }
});

// ── Send Message ─────────────────────────────────────────────
async function handleSend() {
  const text = chatInput.value.trim();
  if (!text || !sessionId) return;

  chatInput.value = '';
  sendBtn.disabled = true;
  addMessage('user', text);

  // Show typing indicator
  const typingId = showTypingIndicator();

  try {
    const result = await sendMessage(sessionId, text);
    removeTypingIndicator(typingId);
    addMessage('assistant', result.response);

    if (result.deal_closed) {
      showDealBanner('closed', `✅ Deal closed at $${result.current_offer.toFixed(2)}! 🎉`);
      disableInput();
    } else if (result.deal_abandoned) {
      showDealBanner('abandoned', '❌ Negotiation ended. The customer walked away.');
      disableInput();
    }
  } catch (err) {
    removeTypingIndicator(typingId);
    addMessage('assistant', `⚠️ Error: ${err.message}. Please try again.`);
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', handleSend);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

// ── Reset ────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  sessionId = null;
  isNegotiating = false;

  welcomeScreen.style.display = 'flex';
  messagesEl.style.display = 'none';
  chatInputArea.style.display = 'none';
  resetBtn.style.display = 'none';
  dealBanner.style.display = 'none';
  messagesEl.innerHTML = '';

  categorySelect.disabled = false;
  buyerState.disabled = false;
  sellerState.disabled = false;
});

// ── Helpers ──────────────────────────────────────────────────
function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = `message ${role} animate-in`;

  const sender = role === 'user' ? 'You' : 'Sales Agent';
  div.innerHTML = `
    <div class="message-sender">${sender}</div>
    <div class="message-bubble">${renderMarkdown(content)}</div>
  `;

  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTypingIndicator() {
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.id = id;
  div.className = 'message assistant animate-in';
  div.innerHTML = `
    <div class="message-sender">Sales Agent</div>
    <div class="message-bubble">
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>
  `;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function showDealBanner(type, text) {
  dealBanner.className = `deal-banner ${type}`;
  dealBanner.textContent = text;
  dealBanner.style.display = 'block';
}

function disableInput() {
  chatInput.disabled = true;
  sendBtn.disabled = true;
  chatInput.placeholder = 'Negotiation ended. Click New Negotiation to start over.';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  // First escape HTML to prevent XSS
  let html = escapeHtml(text);

  // Replace ### Headers
  html = html.replace(/^###\s+(.*$)/gim, '<h3>$1</h3>');
  // Replace ## Headers
  html = html.replace(/^##\s+(.*$)/gim, '<h2>$1</h2>');
  // Replace **bold**
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Replace *italic*
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  // Replace bullet points
  html = html.replace(/^- (.*$)/gim, '<div class="bullet-item">• $1</div>');

  // Prevent double spacing: remove newlines immediately before/after block tags
  html = html.replace(/(<\/h3>|<\/h2>|<\/div>)\s*\n/gi, '$1');
  html = html.replace(/\n\s*(<h3|<h2|<div class="bullet-item")/gi, '$1');

  // Replace remaining newlines with <br>
  html = html.replace(/\n/g, '<br>');

  return html;
}

// ── Admin Modal ──────────────────────────────────────────────
const adminLoginBtn = document.getElementById('admin-login-btn');
const adminModal = document.getElementById('admin-modal');
const closeAdminModalBtn = document.getElementById('close-admin-modal-btn');
const adminIframe = document.getElementById('admin-iframe');

if (adminLoginBtn && adminModal) {
  adminLoginBtn.addEventListener('click', () => {
    // Load the login page (which redirects to admin.html if token exists)
    adminIframe.src = '/login.html';
    adminModal.style.display = 'flex';
  });

  closeAdminModalBtn.addEventListener('click', () => {
    adminModal.style.display = 'none';
    adminIframe.src = ''; // Frame cleared to stop background tasks
  });
}

// ── Boot ─────────────────────────────────────────────────────
init();
