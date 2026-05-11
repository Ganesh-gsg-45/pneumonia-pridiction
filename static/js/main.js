/* ── Theme ───────────────────────────────────────────────────────── */
(function () {
  const saved = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  // Button icon updated after DOM loads
  window.addEventListener('DOMContentLoaded', () => {
    document.getElementById('themeToggle').textContent = saved === 'dark' ? '🌙' : '☀️';
  });
})();

function toggleTheme() {
  const html    = document.documentElement;
  const current = html.getAttribute('data-theme');
  const next    = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  document.getElementById('themeToggle').textContent = next === 'dark' ? '🌙' : '☀️';
}

/* ── State ──────────────────────────────────────────────────────── */
let lastResult = null;   // { result, confidence } from last analysis
let chatHistory = [];    // for display only

/* ── Tab switching ──────────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => {
    t.classList.remove('active');
    t.classList.add('hidden');
  });
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));

  const tab = document.getElementById('tab-' + name);
  tab.classList.remove('hidden');
  tab.classList.add('active');
  document.getElementById('tab-btn-' + name).classList.add('active');
}

/* ── File upload ────────────────────────────────────────────────── */
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadPreview(fileInput.files[0]);
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadPreview(file);
});

function loadPreview(file) {
  if (!file.type.match(/image\/(png|jpeg)/)) {
    alert('Please upload a PNG or JPG image.');
    return;
  }
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('imagePreview').src = ev.target.result;
    document.getElementById('imagePreviewWrap').classList.remove('hidden');
    dropZone.classList.add('hidden');
    analyzeBtn.disabled = false;

    // store file in input for form submission
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  fileInput.value = '';
  document.getElementById('imagePreview').src = '';
  document.getElementById('imagePreviewWrap').classList.add('hidden');
  dropZone.classList.remove('hidden');
  analyzeBtn.disabled = true;
  document.getElementById('resultsContent').classList.add('hidden');
  document.getElementById('resultsPlaceholder').classList.remove('hidden');
  lastResult = null;
}

/* ── Analyze ────────────────────────────────────────────────────── */
async function analyzeXray() {
  if (!fileInput.files[0]) return;

  const btn    = document.getElementById('analyzeBtn');
  const btnTxt = document.getElementById('analyzeBtnText');
  const spin   = document.getElementById('analyzeBtnSpinner');

  btn.disabled = true;
  btnTxt.textContent = 'Analyzing…';
  spin.classList.remove('hidden');

  const form = new FormData();
  form.append('file', fileInput.files[0]);

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: form });
    const data = await res.json();

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    lastResult = { result: data.result, confidence: data.confidence };
    renderResults(data);

  } catch (err) {
    alert('Network error: ' + err.message);
  } finally {
    btn.disabled = false;
    btnTxt.textContent = '🔍 Analyze X-Ray';
    spin.classList.add('hidden');
  }
}

/* ── Render results ─────────────────────────────────────────────── */
function renderResults(data) {
  document.getElementById('resultsPlaceholder').classList.add('hidden');
  document.getElementById('resultsContent').classList.remove('hidden');

  // Badge
  const badge = document.getElementById('resultBadge');
  badge.style.background = data.color + '22';
  badge.style.borderColor = data.color + '55';

  document.getElementById('resultIcon').textContent    = data.icon;
  document.getElementById('resultLabel').textContent   = data.result;
  document.getElementById('resultLabel').style.color   = data.color;
  document.getElementById('resultSeverity').textContent = data.severity;
  document.getElementById('resultConf').textContent    = data.confidence + '%';
  document.getElementById('resultConf').style.color    = data.color;

  // Bars
  const isPN = data.result === 'PNEUMONIA';
  setBar('bar1', isPN ? 'PNEUMONIA' : 'NORMAL', data.confidence, data.color);
  setBar('bar2', isPN ? 'NORMAL' : 'PNEUMONIA', 100 - data.confidence, '#8888aa');

  // Recommendations
  const list = document.getElementById('recsList');
  list.innerHTML = data.recommendations.map((r, i) =>
    `<div class="rec-item">
       <span class="rec-num">${i + 1}</span>
       <span>${mdBold(r)}</span>
     </div>`
  ).join('');
}

function setBar(id, label, pct, color) {
  document.getElementById(id + 'Label').textContent = label;
  document.getElementById(id + 'Pct').textContent   = pct.toFixed(1) + '%';
  const fill = document.getElementById(id + 'Fill');
  fill.style.background = color;
  setTimeout(() => { fill.style.width = pct + '%'; }, 100);
}

/* ── Simple bold markdown ────────────────────────────────────────── */
function mdBold(text) {
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

/* ── Chat ────────────────────────────────────────────────────────── */
function handleEnter(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function sendQuick(question) {
  switchTab('chat');
  document.getElementById('chatInput').value = question;
  setTimeout(sendMessage, 100);
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const text  = input.value.trim();
  if (!text) return;

  input.value = '';
  autoResize(input);

  appendMsg('user', text);

  // typing indicator
  const typingId = appendTyping();

  const sendBtn  = document.getElementById('sendBtn');
  const sendTxt  = document.getElementById('sendBtnText');
  const sendSpin = document.getElementById('sendSpinner');
  sendBtn.disabled = true;
  sendTxt.classList.add('hidden');
  sendSpin.classList.remove('hidden');

  try {
    const res  = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message:     text,
        pred_result: lastResult?.result   || null,
        confidence:  lastResult?.confidence || null
      })
    });
    const data = await res.json();
    removeTyping(typingId);
    appendMsg('bot', data.response || data.error || 'No response received.');
  } catch (err) {
    removeTyping(typingId);
    appendMsg('bot', '⚠️ Network error: ' + err.message);
  } finally {
    sendBtn.disabled = false;
    sendTxt.classList.remove('hidden');
    sendSpin.classList.add('hidden');
  }
}

function appendMsg(role, text) {
  const box = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'msg-user' : 'msg-bot');

  const avatar = role === 'user' ? '👤' : '🤖';
  div.innerHTML =
    `<div class="msg-avatar">${avatar}</div>
     <div class="msg-bubble">${mdBold(escapeHtml(text)).replace(/\n/g, '<br/>')}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function appendTyping() {
  const box = document.getElementById('chatMessages');
  const id  = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.className = 'msg msg-bot';
  div.id = id;
  div.innerHTML =
    `<div class="msg-avatar">🤖</div>
     <div class="msg-bubble">
       <div class="typing-dots"><span></span><span></span><span></span></div>
     </div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/* ── Auto-resize textarea ────────────────────────────────────────── */
const chatInput = document.getElementById('chatInput');
chatInput.addEventListener('input', () => autoResize(chatInput));

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
