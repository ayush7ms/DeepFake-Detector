'use strict';
/* DeepTruth frontend.
 * Everything shown as a result comes from the backend response. When a value is
 * missing the interface says so; it never fills in a placeholder score.
 * Response text is inserted with textContent / DOM nodes, never as HTML.
 */

const TYPE_CONFIG = {
  image: { extensions: ['png', 'jpg', 'jpeg', 'webp'], maxBytes: 25 * 1024 * 1024, label: 'Image', icon: '▧' },
  video: { extensions: ['mp4', 'mov', 'webm'], maxBytes: 250 * 1024 * 1024, label: 'Video', icon: '▷' },
  audio: { extensions: ['wav', 'mp3', 'm4a'], maxBytes: 50 * 1024 * 1024, label: 'Audio', icon: '◖' }
};
const PAGE_NAMES = { dashboard: 'Dashboard', image: 'Image Investigation', video: 'Video Investigation', audio: 'Audio Investigation', history: 'Investigation History', about: 'Methodology' };
const BUTTON_LABEL = { image: 'Analyse image', video: 'Analyse video', audio: 'Analyse audio' };
const DEFAULT_LOCAL_API = 'http://127.0.0.1:8001';
const START_COMMAND = 'python -m uvicorn main:app --reload';

const state = { api: null, connected: false, health: null, stats: null, history: [] };
const selectedFiles = new Map();
const objectUrls = new Map();
const busy = new Set();
let toastTimer;

/* ---------- small helpers ---------- */
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}
function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  for (const child of children.flat()) if (child) node.append(child);
  return node;
}
const isNum = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
const clamp01 = value => Math.min(1, Math.max(0, Number(value) || 0));
const pct = (value, digits = 1) => (isNum(value) ? `${(Number(value) * 100).toFixed(digits)}%` : '—');
const humanize = text => String(text ?? '').replaceAll('_', ' ');

function formatSize(bytes) {
  if (!isNum(bytes)) return '—';
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
function formatDuration(seconds) {
  if (!isNum(seconds)) return '--:--';
  const total = Math.max(0, Math.round(Number(seconds)));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}
function formatDate(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? String(iso || '—') : date.toLocaleString();
}
function fileExtension(file) { return file.name.split('.').pop().toLowerCase(); }
/* Only JPEG data URLs produced by our backend are ever used as image sources. */
function safeImage(url) { return typeof url === 'string' && url.startsWith('data:image/jpeg;base64,') ? url : ''; }
function timeoutSignal(ms) {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3600);
}

/* ---------- navigation ---------- */
function navigate(page) {
  if (!PAGE_NAMES[page]) return;
  $$('.page-section').forEach(section => section.classList.toggle('active', section.id === `page-${page}`));
  $$('.nav-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === page);
    if (item.dataset.page === page) item.setAttribute('aria-current', 'page'); else item.removeAttribute('aria-current');
  });
  $('#pageCrumb').textContent = PAGE_NAMES[page];
  $('#sidebar').classList.remove('open');
  history.replaceState(null, '', `#${page}`);
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if ((page === 'dashboard' || page === 'history') && state.connected) refreshData();
}
$$('[data-page]').forEach(button => button.addEventListener('click', () => navigate(button.dataset.page)));
$$('[data-go]').forEach(button => button.addEventListener('click', () => navigate(button.dataset.go)));
$('#menuToggle').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
$('.brand').addEventListener('click', event => { event.preventDefault(); navigate('dashboard'); });
window.addEventListener('hashchange', () => navigate(location.hash.slice(1) || 'dashboard'));

/* ---------- backend connection ---------- */
async function probe(base) {
  try {
    const response = await fetch(`${base}/api/health`, { signal: timeoutSignal(4000) });
    if (!response.ok) return null;
    const body = await response.json();
    return body && body.service === 'DeepTruth API' ? body : null;
  } catch { return null; }
}
async function checkBackend() {
  let health = state.api !== null ? await probe(state.api) : null;
  if (!health) {
    const candidates = window.DEEPTRUTH_API_BASE_URL
      ? [String(window.DEEPTRUTH_API_BASE_URL).replace(/\/$/, '')]
      : [...(location.protocol.startsWith('http') ? [''] : []), DEFAULT_LOCAL_API];
    for (const base of candidates) {
      health = await probe(base);
      if (health) { state.api = base; break; }
    }
  }
  const wasConnected = state.connected;
  state.connected = Boolean(health);
  state.health = health;
  renderConnection();
  renderModelStatus();
  renderAudioNotice();
  updateAnalyzeButtons();
  if (state.connected && !wasConnected) await refreshData();
  if (!state.connected && wasConnected) renderStats();
}
function renderConnection() {
  const dot = $('#connDot');
  dot.classList.toggle('connected', state.connected);
  dot.classList.toggle('error', !state.connected);
  $('#connText').textContent = state.connected ? 'API connected' : 'Backend unavailable';
  $('#connChip').classList.toggle('is-online', state.connected);
  $('#connChip').classList.toggle('is-offline', !state.connected);
  $('#connChipText').textContent = state.connected ? 'API CONNECTED' : 'BACKEND OFFLINE';
}
async function apiRequest(path, options = {}) {
  const response = await fetch(`${state.api}${path}`, options);
  let body = null;
  try { body = await response.json(); } catch { /* handled below */ }
  if (!response.ok) {
    const error = new Error((body && (body.error || body.detail)) || `Request failed (HTTP ${response.status}).`);
    error.status = response.status; error.payload = body;
    throw error;
  }
  return body;
}
async function refreshData() {
  if (!state.connected) return;
  try {
    const [stats, list] = await Promise.all([apiRequest('/api/stats'), apiRequest('/api/history')]);
    state.stats = stats; state.history = list.items || [];
  } catch (error) {
    showToast(`Could not load history: ${error.message}`);
  }
  renderStats(); renderBands(); renderRecent(); renderHistory();
}

/* ---------- dashboard ---------- */
function renderStats() {
  const stats = state.connected ? state.stats : null;
  $$('[data-stat]').forEach(node => { node.textContent = stats && isNum(stats[node.dataset.stat]) ? String(stats[node.dataset.stat]) : '—'; });
  $('#statsNote').textContent = state.connected
    ? 'These are real counts from investigations stored by your local backend. Nothing is estimated or pre-filled.'
    : `The backend is not reachable, so no counts are shown. Start it with: ${START_COMMAND}`;
}
function statusRow(name, detail, tone) {
  return h('div', { class: 'status-row' }, h('i', { class: `status-dot ${tone}`, 'aria-hidden': 'true' }), h('div', {}, h('strong', { text: name }), h('small', { text: detail })));
}
function renderModelStatus() {
  const target = $('#modelStatus');
  if (!state.connected || !state.health) {
    target.replaceChildren(statusRow('Backend', `Not reachable. Start it with ${START_COMMAND}`, 'error'));
    return;
  }
  const analyzers = state.health.analyzers || {};
  const image = analyzers.image || {};
  const audio = analyzers.audio || {};
  target.replaceChildren(
    statusRow(`Image · ${image.name || 'classifier'}`, image.status === 'loaded' ? 'Model loaded in memory' : 'Loads on the first image or video (weights download once)', 'connected'),
    statusRow('Video · frame-wise', 'Uses the image model on sampled frames. No temporal analysis', 'connected'),
    statusRow('Audio · synthetic voice', audioConfigured() ? 'Detector configured' : 'No detector configured. Analysis unavailable', audioConfigured() ? 'connected' : 'warn')
  );
}
function renderBands() {
  const target = $('#bandChart');
  const counts = state.connected && state.stats ? state.stats.band_counts : null;
  const total = counts ? Object.values(counts).reduce((sum, value) => sum + value, 0) : 0;
  if (!counts || !total) {
    target.replaceChildren(h('p', { class: 'fine', text: state.connected ? 'No completed investigations yet.' : 'Backend offline.' }));
    return;
  }
  const rows = [['low', 'Low'], ['inconclusive', 'Inconclusive'], ['review', 'Elevated']];
  target.replaceChildren(...rows.map(([band, label]) => {
    const fill = h('i', { class: `band-${band}` });
    fill.style.width = `${(counts[band] / total) * 100}%`;
    return h('div', { class: 'band-row' }, h('span', { text: label }), h('div', { class: 'bar' }, fill), h('b', { text: String(counts[band]) }));
  }));
}
function emptyState(icon, title, copy) {
  return h('div', { class: 'empty-state' }, h('span', { class: 'empty-icon', 'aria-hidden': 'true', text: icon }), h('strong', { text: title }), h('span', { text: copy }));
}
function bandTag(item) {
  const band = item.band || 'none';
  return h('span', { class: `band-chip band-${band}`, text: item.band_label || 'No model score' });
}
function historyRow(item, compact) {
  const config = TYPE_CONFIG[item.media_type] || TYPE_CONFIG.image;
  const actions = h('div', { class: 'row-actions' },
    h('button', { class: 'text-button', type: 'button', text: 'Open', 'aria-label': `Open result for ${item.filename}`, onclick: () => openInvestigation(item.investigation_id) }));
  if (!compact) {
    actions.append(
      h('a', { class: 'text-button', href: `${state.api}${item.report_url}`, text: 'PDF report', 'aria-label': `Download PDF report for ${item.filename}` }),
      h('button', { class: 'text-button danger-text', type: 'button', text: 'Delete', 'aria-label': `Delete ${item.filename} from history`, onclick: () => deleteInvestigation(item) })
    );
  }
  return h('div', { class: 'history-row' },
    h('span', { class: 'history-type', 'aria-hidden': 'true', text: config.icon }),
    h('div', {}, h('strong', { text: item.filename, title: item.filename }), h('small', { text: `${config.label} · ${formatSize(item.size_bytes)} · ${formatDate(item.generated_at)}` })),
    h('div', { class: 'row-score' }, bandTag(item), isNum(item.model_score) ? h('small', { text: `score ${pct(item.model_score)}` }) : null),
    actions
  );
}
function renderRecent() {
  const target = $('#recentList');
  if (!state.connected) { target.replaceChildren(emptyState('⌕', 'Backend offline', 'History is stored by the backend. Start it to see your investigations.')); return; }
  if (!state.history.length) { target.replaceChildren(emptyState('⌕', 'No investigations yet', 'Upload an image or video to start your first investigation.')); return; }
  target.replaceChildren(...state.history.slice(0, 5).map(item => historyRow(item, true)));
}
function renderHistory() {
  const target = $('#historyList');
  const count = state.history.length;
  $('#historyCount').textContent = `${count} ITEM${count === 1 ? '' : 'S'}`;
  $('#clearHistory').disabled = !state.connected || count === 0;
  if (!state.connected) { target.replaceChildren(emptyState('◷', 'Backend offline', 'History is stored by the backend. Start it to see your investigations.')); return; }
  if (!count) { target.replaceChildren(emptyState('◷', 'No investigations yet', 'Completed analyses are listed here. Nothing is invented.')); return; }
  target.replaceChildren(...state.history.map(item => historyRow(item, false)));
}
async function openInvestigation(id) {
  try {
    const result = await apiRequest(`/api/history/${encodeURIComponent(id)}`);
    navigate(result.media_type);
    renderResult(result.media_type, result, { restored: true });
    $(`#result-${result.media_type}`).scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) { showToast(`Could not open investigation: ${error.message}`); }
}
async function deleteInvestigation(item) {
  if (!window.confirm(`Delete "${item.filename}" from history? Its report will no longer be available.`)) return;
  try {
    await apiRequest(`/api/history/${encodeURIComponent(item.investigation_id)}`, { method: 'DELETE' });
    showToast('Investigation deleted.');
    await refreshData();
  } catch (error) { showToast(`Could not delete: ${error.message}`); }
}
$('#clearHistory').addEventListener('click', async () => {
  if (!window.confirm('Delete all stored investigations? This cannot be undone.')) return;
  try {
    const { deleted_count: removed } = await apiRequest('/api/history', { method: 'DELETE' });
    showToast(`Deleted ${removed} investigation${removed === 1 ? '' : 's'}.`);
    await refreshData();
  } catch (error) { showToast(`Could not clear history: ${error.message}`); }
});

/* ---------- audio availability ---------- */
function audioConfigured() {
  const audio = state.health && state.health.analyzers && state.health.analyzers.audio;
  return Boolean(audio && audio.status === 'configured');
}
function renderAudioNotice() {
  const notice = $('#audioNotice');
  notice.className = 'notice';
  if (!state.connected) {
    notice.classList.add('notice-amber');
    notice.replaceChildren(h('strong', { text: 'Backend offline' }), h('span', { text: `Audio availability cannot be checked. Start the backend with ${START_COMMAND}.` }));
  } else if (audioConfigured()) {
    notice.classList.add('notice-blue');
    const info = state.health.analyzers.audio.model_information || {};
    notice.replaceChildren(h('strong', { text: 'A synthetic-audio detector is configured' }), h('span', { text: `Model: ${info.name || 'unnamed'}. Review its limitations in every result.` }));
  } else {
    notice.classList.add('notice-amber');
    notice.replaceChildren(h('strong', { text: 'Audio analysis is unavailable' }), h('span', { text: 'No pretrained synthetic-audio detector is configured on this server, so no score will be produced. You can still preview a recording here.' }));
  }
}

/* ---------- file selection ---------- */
function canAnalyze(kind) {
  return selectedFiles.has(kind) && state.connected && !busy.has(kind) && (kind !== 'audio' || audioConfigured());
}
function updateAnalyzeButtons() {
  Object.keys(TYPE_CONFIG).forEach(kind => {
    const button = $(`#page-${kind} .analyze-button`);
    if (busy.has(kind)) return;
    button.disabled = !canAnalyze(kind);
    button.textContent = kind === 'audio' && !audioConfigured() ? 'Analysis unavailable' : BUTTON_LABEL[kind];
    button.title = !state.connected ? 'The backend is not reachable.' : (kind === 'audio' && !audioConfigured() ? 'No synthetic-audio detector is configured.' : '');
  });
}
function renderEmptyResult(kind) {
  const copy = kind === 'audio'
    ? 'No audio score will appear unless a detector is configured. DeepTruth does not guess.'
    : 'Run an analysis and the score, evidence and limitations appear here. DeepTruth never shows placeholder results.';
  $(`#result-${kind}`).replaceChildren(h('div', { class: 'result-empty' }, h('span', { class: 'empty-icon', 'aria-hidden': 'true', text: '⌖' }), h('strong', { text: 'No analysis yet' }), h('span', { text: copy })));
}
function clearFile(kind, keepResult = false) {
  if (objectUrls.has(kind)) URL.revokeObjectURL(objectUrls.get(kind));
  objectUrls.delete(kind); selectedFiles.delete(kind);
  const section = $(`#page-${kind}`);
  const preview = section.querySelector('.file-preview');
  preview.classList.add('hidden'); preview.replaceChildren();
  section.querySelector('.dropzone').classList.remove('hidden');
  section.querySelector('.file-input').value = '';
  section.querySelector('.upload-error').textContent = '';
  section.querySelector('.analysis-feedback').classList.add('hidden');
  section.querySelector('.progress').classList.add('hidden');
  if (!keepResult) renderEmptyResult(kind);
  updateAnalyzeButtons();
}
function acceptFile(kind, file) {
  if (!file) return;
  const section = $(`#page-${kind}`);
  const config = TYPE_CONFIG[kind];
  const error = section.querySelector('.upload-error');
  if (!config.extensions.includes(fileExtension(file))) {
    error.textContent = `Unsupported file type. Choose ${config.extensions.join(', ').toUpperCase()} for this investigation.`;
    return;
  }
  if (file.size > config.maxBytes) {
    error.textContent = `This file is ${formatSize(file.size)}. The maximum size is ${formatSize(config.maxBytes)}.`;
    return;
  }
  clearFile(kind);
  selectedFiles.set(kind, file);
  const url = URL.createObjectURL(file);
  objectUrls.set(kind, url);

  const info = h('small', { text: `${config.label} · ${formatSize(file.size)} · ${(file.type || fileExtension(file)).toLowerCase()}` });
  const details = h('div', { class: 'file-details' }, h('strong', { text: file.name, title: file.name }), info);
  const remove = h('button', { class: 'remove-file', type: 'button', 'aria-label': 'Remove selected file', text: '×', onclick: () => clearFile(kind) });
  let media;
  if (kind === 'image') {
    media = h('img', { alt: 'Preview of the selected image', src: url });
    media.addEventListener('load', () => { info.textContent += ` · ${media.naturalWidth}×${media.naturalHeight}px`; });
  } else {
    media = h(kind === 'video' ? 'video' : 'audio', { src: url, controls: true, preload: 'metadata' });
    media.addEventListener('loadedmetadata', () => {
      const dims = kind === 'video' && media.videoWidth ? ` · ${media.videoWidth}×${media.videoHeight}px` : '';
      info.textContent += ` · ${formatDuration(media.duration)}${dims}`;
    });
  }
  const preview = section.querySelector('.file-preview');
  preview.className = `file-preview large kind-${kind}`;
  preview.replaceChildren(media, h('div', { class: 'preview-row' }, details, remove));
  section.querySelector('.dropzone').classList.add('hidden');
  updateAnalyzeButtons();
}
$$('.dropzone').forEach(zone => {
  const input = zone.querySelector('.file-input');
  zone.addEventListener('click', event => { if (event.target !== input) input.click(); });
  zone.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); } });
  input.addEventListener('change', () => acceptFile(zone.dataset.kind, input.files[0]));
  zone.addEventListener('dragover', event => { event.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', event => { event.preventDefault(); zone.classList.remove('dragging'); acceptFile(zone.dataset.kind, event.dataTransfer.files[0]); });
});

/* ---------- analysis request ---------- */
function uploadFile(url, file, onProgress, onSent) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', url);
    request.responseType = 'text';
    request.upload.onprogress = event => { if (event.lengthComputable) onProgress(event.loaded / event.total); };
    request.upload.onload = onSent;
    request.onload = () => {
      try { resolve({ status: request.status, body: JSON.parse(request.responseText) }); }
      catch { reject(new Error(`The server returned an unreadable response (HTTP ${request.status}).`)); }
    };
    request.onerror = () => reject(new Error('The connection to the DeepTruth backend was lost.'));
    request.onabort = () => reject(new Error('The upload was cancelled.'));
    const form = new FormData();
    form.append('file', file);
    request.send(form);
  });
}
function setBusy(kind, on) {
  const section = $(`#page-${kind}`);
  const button = section.querySelector('.analyze-button');
  if (on) busy.add(kind); else busy.delete(kind);
  button.classList.toggle('loading', on);
  if (on) {
    button.disabled = true;
    button.replaceChildren(h('span', { class: 'spinner', 'aria-hidden': 'true' }), ' Working…');
  } else {
    updateAnalyzeButtons();
  }
}
async function runAnalysis(kind) {
  const file = selectedFiles.get(kind);
  if (!file) return;
  const section = $(`#page-${kind}`);
  const feedback = section.querySelector('.analysis-feedback');
  const progress = section.querySelector('.progress');
  const bar = progress.querySelector('i');
  if (!state.connected) {
    feedback.textContent = `The backend is not reachable. Start it with: ${START_COMMAND}`;
    feedback.classList.remove('hidden');
    return;
  }
  setBusy(kind, true);
  renderEmptyResult(kind);
  progress.classList.remove('hidden', 'indeterminate');
  bar.style.width = '0%';
  feedback.classList.remove('hidden');
  feedback.textContent = 'Uploading to your local backend…';
  const started = Date.now();
  let phase = 'upload';
  const ticker = setInterval(() => {
    if (phase !== 'analyse') return;
    const seconds = Math.round((Date.now() - started) / 1000);
    feedback.textContent = `Analysing on this machine… ${seconds}s. The first run loads the model, and downloads it once, so it can take several minutes.`;
  }, 1000);
  try {
    const { status, body } = await uploadFile(
      `${state.api}/api/analyze/${kind}`, file,
      fraction => { bar.style.width = `${Math.round(fraction * 100)}%`; progress.setAttribute('aria-valuenow', String(Math.round(fraction * 100))); feedback.textContent = `Uploading… ${Math.round(fraction * 100)}%`; },
      () => { phase = 'analyse'; progress.classList.add('indeterminate'); feedback.textContent = 'Analysing on this machine…'; }
    );
    renderResult(kind, body, { restored: false, httpStatus: status });
    if (body.analysis_status === 'completed') {
      showToast('Analysis complete. Read the limitations before drawing conclusions.');
      await refreshData();
    } else {
      showToast('No model score was produced. This is not a finding about the media.');
    }
    feedback.textContent = body.analysis_status === 'completed' ? 'Analysis complete.' : `No score produced (${humanize(body.analysis_status)}).`;
  } catch (error) {
    renderResult(kind, {
      analysis_status: 'request_failed', model_score: null, evidence: [], suspicious_timestamps: [],
      error: error.message, limitations: [error.message, 'No prediction was produced; a failed request is not evidence that the media is fake.']
    }, { restored: false });
    feedback.textContent = `Request failed: ${error.message}`;
    checkBackend();
  } finally {
    clearInterval(ticker);
    progress.classList.add('hidden'); progress.classList.remove('indeterminate');
    setBusy(kind, false);
  }
}
$$('.analyze-button').forEach(button => button.addEventListener('click', () => runAnalysis(button.closest('.page-section').id.replace('page-', ''))));

/* ---------- result rendering ---------- */
function gauge(score, thresholds) {
  const low = isNum(thresholds && thresholds.low) ? Number(thresholds.low) : 0.25;
  const review = isNum(thresholds && thresholds.review) ? Number(thresholds.review) : 0.75;
  const track = h('div', { class: 'gauge', role: 'img', 'aria-label': `Model score ${pct(score)} on a scale from 0 to 100 percent. Descriptive bands change at ${pct(low, 0)} and ${pct(review, 0)}.` });
  track.style.setProperty('--low', `${low * 100}%`);
  track.style.setProperty('--review', `${review * 100}%`);
  const marker = h('i', { class: 'gauge-marker' });
  marker.style.left = `${clamp01(score) * 100}%`;
  track.append(marker);
  const tick = (value, label) => { const node = h('span', { text: label }); node.style.left = `${value * 100}%`; return node; };
  return h('div', { class: 'gauge-wrap' }, track, h('div', { class: 'gauge-scale', 'aria-hidden': 'true' }, tick(0, '0%'), tick(low, pct(low, 0)), tick(review, pct(review, 0)), tick(1, '100%')));
}
function scoreSection(result) {
  const interpretation = result.interpretation || {};
  const band = interpretation.band || 'none';
  const isDemo = (result.evidence || []).some((item) => item.type === 'demo_rule');
  if (!isNum(result.model_score) && isDemo) {
    const fake = band === 'review';
    return h('div', { class: `score-block band-${band}` },
      h('div', { class: `demo-verdict ${fake ? 'is-fake' : 'is-real'}` }, h('strong', { text: fake ? 'FAKE' : 'REAL' }),
        h('span', { text: 'Simulated demo verdict from a file-name rule (WhatsApp = fake, otherwise real). The image was not judged by a model.' })),
      h('div', { class: 'score-detail' }, h('p', { class: 'score-summary', text: interpretation.summary || '' })));
  }
  if (!isNum(result.model_score)) {
    return h('div', { class: 'score-block band-none' }, h('div', { class: 'score-main' }, h('span', { class: 'score-label', text: 'Fake-class model score' }), h('strong', { class: 'score-value', text: '—' }), h('span', { class: 'band-chip band-none', text: 'No model score' })), h('div', { class: 'score-detail' }, h('p', { class: 'score-summary', text: 'No score was produced for this file.' })));
  }
  const demo = (result.evidence || []).some((item) => item.type === 'demo_rule');
  const demoFake = demo && band === 'review';
  const banner = demo ? h('div', { class: `demo-verdict ${demoFake ? 'is-fake' : 'is-real'}` },
    h('strong', { text: demoFake ? 'FAKE' : 'REAL' }),
    h('span', { text: 'Simulated demo verdict from a file-name rule (WhatsApp = fake, otherwise real). The image was not judged by the model.' })) : null;
  return h('div', { class: `score-block band-${band}` },
    banner,
    h('div', { class: 'score-main' },
      h('span', { class: 'score-label', text: demo ? 'Underlying model score (not the verdict)' : 'Fake-class model score' }),
      h('strong', { class: 'score-value', text: pct(result.model_score) }),
      h('span', { class: `band-chip band-${band}`, text: interpretation.label || 'Model score' })),
    h('div', { class: 'score-detail' },
      gauge(result.model_score, interpretation.thresholds),
      h('p', { class: 'score-summary', text: interpretation.summary || '' }),
      h('p', { class: 'fine', text: `${interpretation.note || ''} The score is the checkpoint’s own output, not a calibrated probability.` })));
}
function warningsBox(list) {
  if (!Array.isArray(list) || !list.length) return null;
  return h('div', { class: 'notice notice-amber' }, h('strong', { text: list.length === 1 ? 'Scope warning' : 'Scope warnings' }), h('ul', { class: 'plain-list' }, list.map(text => h('li', { text }))));
}
function classBars(evidence) {
  const item = (evidence || []).find(entry => entry.type === 'image_classification');
  if (!item) return null;
  const row = (label, value, tone) => {
    const fill = h('i', { class: tone }); fill.style.width = `${clamp01(value) * 100}%`;
    return h('div', { class: 'band-row' }, h('span', { text: label }), h('div', { class: 'bar' }, fill), h('b', { text: pct(value) }));
  };
  return h('div', { class: 'class-scores' }, h('h4', { text: 'Class scores' }), row('Fake class', item.fake_class_score, 'band-review'), row('Real class', item.real_class_score, 'band-low'),
    h('p', { class: 'fine', text: `Highest class: ${item.model_output}. ${item.interpretation || ''}` }));
}
function imageVisuals(result) {
  const visuals = result.visuals || {};
  const explain = (result.evidence || []).find(entry => entry.type === 'explainability');
  const figure = (title, source, caption, fallback) => h('figure', { class: 'evidence-figure' },
    source ? h('img', { src: source, alt: title }) : h('div', { class: 'figure-missing', text: fallback }),
    h('figcaption', {}, h('strong', { text: title }), h('span', { text: caption })));
  const modelInput = visuals.model_input ? safeImage(visuals.model_input.image) : '';
  const heat = visuals.heatmap ? safeImage(visuals.heatmap.image) : '';
  const retained = result.visuals_retained !== false;
  const heatFallback = explain && explain.status === 'unavailable'
    ? `Grad-CAM unavailable: ${explain.reason || 'no reason returned'}`
    : (retained ? 'No overlay was generated.' : 'Overlay not stored. DeepTruth does not keep image content after analysis. Run the analysis again to regenerate it.');
  return h('section', { class: 'result-section' },
    h('h3', { text: 'Visual evidence' }),
    h('div', { class: 'figure-grid' },
      figure('What the model received', modelInput, (visuals.model_input && visuals.model_input.caption) || 'The resized, cropped picture the network scored.', retained ? 'Not available.' : 'Not stored after analysis.'),
      figure('Grad-CAM overlay', heat, (visuals.heatmap && visuals.heatmap.caption) || 'Where the model’s fake-class score came from.', heatFallback)),
    heat ? h('div', { class: 'heat-legend' }, h('span', { text: 'less influence' }), h('i', { 'aria-hidden': 'true' }), h('span', { text: 'more influence' })) : null,
    h('p', { class: 'fine', text: 'Grad-CAM explains the model, not the image: bright areas raised this checkpoint’s fake-class score. They do not show which pixels were edited and are not proof of manipulation.' }));
}
function evidenceTable(result) {
  const rows = [];
  (result.evidence || []).forEach(item => {
    if (item.type === 'image_classification') rows.push(['Classifier output', `${item.model_output} (fake ${pct(item.fake_class_score)} · real ${pct(item.real_class_score)})`]);
    else if (item.type === 'face_detection') rows.push(['Face check', item.faces_detected === null || item.faces_detected === undefined ? 'Unavailable' : `${item.faces_detected} frontal face${item.faces_detected === 1 ? '' : 's'} found (${item.method})`]);
    else if (item.type === 'explainability') rows.push(['Grad-CAM', item.status === 'generated' ? `Generated on a ${item.patch_grid} patch grid for the “${item.target_class}” class` : `Unavailable: ${item.reason || 'no reason returned'}`]);
    else if (item.type !== 'sampled_video_frames') rows.push([humanize(item.type || 'Evidence'), Object.entries(item).filter(([key]) => key !== 'type').map(([key, value]) => `${humanize(key)}: ${typeof value === 'object' ? JSON.stringify(value) : value}`).join(' · ')]);
  });
  if (!rows.length) return null;
  return h('section', { class: 'result-section' }, h('h3', { text: 'Evidence returned by the backend' }),
    h('dl', { class: 'kv-list' }, rows.map(([key, value]) => [h('dt', { text: key }), h('dd', { text: value })])));
}
function limitationsBox(list) {
  const items = Array.isArray(list) && list.length ? list : ['The backend returned no limitations.'];
  return h('details', { class: 'limits', open: true }, h('summary', { text: `Limitations and uncertainty (${items.length})` }), h('ul', { class: 'plain-list' }, items.map(text => h('li', { text: String(text) }))));
}
function metaFooter(result) {
  const upload = result.upload_information || {};
  const model = result.model_information || {};
  const entries = [
    ['Investigation ID', result.investigation_id], ['File', upload.filename || result.filename], ['File size', isNum(upload.size_bytes) ? formatSize(upload.size_bytes) : null],
    ['SHA-256', upload.sha256], ['Generated', result.generated_at ? formatDate(result.generated_at) : null],
    ['Model', model.name ? `${model.name} (${model.repository || 'repository unknown'})` : null], ['Device', model.device]
  ].filter(([, value]) => value);
  if (!entries.length) return null;
  return h('details', { class: 'limits meta' }, h('summary', { text: 'Technical details' }), h('dl', { class: 'kv-list mono' }, entries.map(([key, value]) => [h('dt', { text: key }), h('dd', { text: String(value) })])));
}
function timelineChart(frames, duration, threshold, candidates) {
  const width = 720, height = 190, left = 40, right = 12, top = 12, bottom = 30;
  const plotW = width - left - right, plotH = height - top - bottom;
  const svg = svgEl('svg', { viewBox: `0 0 ${width} ${height}`, class: 'timeline-chart', role: 'img', 'aria-label': `Bar chart of fake-class model scores for ${frames.length} sampled frames across the video. Frames at or above ${pct(threshold, 0)} are highlighted for human review.` });
  [0, 0.5, 1].forEach(tick => {
    const y = top + plotH * (1 - tick);
    svg.append(svgEl('line', { x1: left, x2: width - right, y1: y, y2: y, class: 'grid-line' }));
    const label = svgEl('text', { x: left - 6, y: y + 3, class: 'axis-label', 'text-anchor': 'end' }); label.textContent = `${tick * 100}%`; svg.append(label);
  });
  const thresholdY = top + plotH * (1 - threshold);
  svg.append(svgEl('line', { x1: left, x2: width - right, y1: thresholdY, y2: thresholdY, class: 'threshold-line' }));
  const thresholdLabel = svgEl('text', { x: width - right - 2, y: thresholdY - 4, class: 'axis-label review', 'text-anchor': 'end' }); thresholdLabel.textContent = `review band ${pct(threshold, 0)}`; svg.append(thresholdLabel);
  const barW = Math.min(26, Math.max(8, plotW / Math.max(frames.length, 1) * 0.6));
  frames.forEach((frame, index) => {
    const fraction = duration > 0 ? clamp01(frame.timestamp_seconds / duration) : (frames.length > 1 ? index / (frames.length - 1) : 0.5);
    const x = left + barW / 2 + fraction * (plotW - barW);
    const barH = Math.max(1.5, clamp01(frame.model_score) * plotH);
    const flagged = candidates.has(frame.frame_index);
    const bar = svgEl('rect', { x: x - barW / 2, y: top + plotH - barH, width: barW, height: barH, rx: 2, class: flagged ? 'bar-review' : 'bar-normal' });
    const title = svgEl('title'); title.textContent = `${frame.timestamp_seconds.toFixed(2)} s · frame ${frame.frame_index} · model score ${pct(frame.model_score)}${flagged ? ' · review candidate' : ''}`;
    bar.append(title); svg.append(bar);
    const axis = svgEl('text', { x, y: height - 12, class: 'axis-label', 'text-anchor': 'middle' }); axis.textContent = formatDuration(frame.timestamp_seconds); svg.append(axis);
  });
  return svg;
}
function videoSections(result) {
  const evidence = (result.evidence || []).find(entry => entry.type === 'sampled_video_frames');
  if (!evidence || !Array.isArray(evidence.sampled_frames)) return [];
  const frames = evidence.sampled_frames;
  const meta = (result.model_information && result.model_information.video_metadata) || {};
  const thumbs = (result.visuals && result.visuals.frame_thumbnails) || {};
  const candidates = new Set((result.suspicious_timestamps || []).map(item => item.frame_index));
  const threshold = isNum(evidence.review_threshold) ? Number(evidence.review_threshold) : 0.75;
  const facts = [
    `${evidence.frames_analyzed} of ${evidence.frames_requested} frames analysed`,
    isNum(evidence.frames_with_detected_face) ? `${evidence.frames_with_detected_face} with a face` : null,
    meta.width ? `${meta.width}×${meta.height}px` : null, isNum(meta.fps) ? `${Number(meta.fps).toFixed(2)} fps` : null, isNum(meta.duration_seconds) ? formatDuration(meta.duration_seconds) : null,
    evidence.aggregate_basis ? `aggregate over ${humanize(evidence.aggregate_basis)}` : null
  ].filter(Boolean);
  const cards = frames.map(frame => {
    const flagged = candidates.has(frame.frame_index);
    const source = safeImage(thumbs[String(frame.frame_index)]);
    const fill = h('i', { class: flagged ? 'band-review' : 'band-inconclusive' }); fill.style.width = `${clamp01(frame.model_score) * 100}%`;
    return h('article', { class: `frame-card${flagged ? ' review-candidate' : ''}` },
      source ? h('img', { src: source, alt: `Sampled frame at ${frame.timestamp_seconds.toFixed(2)} seconds` }) : h('div', { class: 'figure-missing', text: result.visuals_retained === false ? 'Frame preview not stored' : 'No preview' }),
      h('div', { class: 'frame-meta' }, h('strong', { text: `${frame.timestamp_seconds.toFixed(2)} s` }), h('small', { text: `frame ${frame.frame_index}` })),
      h('div', { class: 'bar' }, fill), h('div', { class: 'frame-meta' }, h('span', { text: pct(frame.model_score) }), h('small', { text: frame.faces_detected === null || frame.faces_detected === undefined ? 'face check n/a' : `${frame.faces_detected} face${frame.faces_detected === 1 ? '' : 's'}` })),
      flagged ? h('span', { class: 'band-chip band-review', text: 'review candidate' }) : null);
  });
  return [
    h('section', { class: 'result-section' }, h('h3', { text: 'Frame extraction' }), h('p', { class: 'chips' }, facts.map(text => h('span', { class: 'chip', text })))),
    h('section', { class: 'result-section' }, h('h3', { text: 'Score timeline' }),
      timelineChart(frames, Number(meta.duration_seconds) || 0, threshold, candidates),
      h('p', { class: 'fine', text: 'Each bar is one sampled frame scored on its own. Gaps between bars were not analysed, and the chart shows no motion or lip-sync analysis.' })),
    h('section', { class: 'result-section' }, h('h3', { text: 'Timestamps for human review' }),
      (result.suspicious_timestamps || []).length
        ? h('ul', { class: 'plain-list' }, result.suspicious_timestamps.map(item => h('li', { text: `${Number(item.timestamp_seconds).toFixed(2)} s (frame ${item.frame_index}): model score ${pct(item.model_score)}. ${item.reason || ''}` })))
        : h('p', { class: 'muted', text: 'No sampled frame reached the review band. That does not mean the video is unmodified.' })),
    h('section', { class: 'result-section' }, h('h3', { text: `Frame-by-frame results (${frames.length})` }), h('div', { class: 'frame-grid' }, cards))
  ];
}
function problemSection(result) {
  const hints = {
    model_unavailable: 'The model could not be loaded. The first run must download the weights once, so check your internet connection, then run “python verify_model.py” to diagnose.',
    not_configured: 'No detector is configured for this media type, so no score exists.',
    rejected: 'The file was rejected before analysis.',
    invalid_media: 'The file content does not match a supported, readable format.',
    request_failed: `Make sure the backend is running (${START_COMMAND}).`
  };
  return h('div', { class: 'notice notice-amber' }, h('strong', { text: `No score produced · ${humanize(result.analysis_status)}` }), h('span', { text: result.error || hints[result.analysis_status] || 'The analysis did not complete.' }),
    hints[result.analysis_status] && result.error ? h('span', { text: hints[result.analysis_status] }) : null,
    h('span', { class: 'fine', text: 'A failed or unavailable analysis is not evidence that the media is fake or genuine.' }));
}
function renderResult(kind, result, options = {}) {
  const panel = $(`#result-${kind}`);
  const completed = result.analysis_status === 'completed';
  const reportLink = result.report_url && state.api !== null
    ? h('a', { class: 'button button-secondary', href: `${state.api}${result.report_url}`, text: 'Download PDF report' }) : null;
  const nodes = [
    h('div', { class: 'result-head' },
      h('div', {}, h('h2', { text: completed ? 'Investigation result' : 'No result produced' }), h('p', { class: 'muted', text: (result.upload_information && result.upload_information.filename) || result.filename || 'Uploaded file' })),
      h('div', { class: 'head-actions' }, h('span', { class: `tag${completed ? ' tag-ok' : ' tag-warn'}`, text: humanize(result.analysis_status).toUpperCase() }), reportLink))
  ];
  if (options.restored) nodes.push(h('div', { class: 'notice notice-blue' }, h('strong', { text: 'Restored from history' }), h('span', { text: result.visuals_retained === false && kind !== 'audio' ? 'Scores, evidence and limitations were saved. The original file and any visual overlays were not stored, so they are not shown.' : 'Opened from your stored investigations.' })));
  if (!completed) nodes.push(problemSection(result));
  if (completed || isNum(result.model_score)) nodes.push(scoreSection(result));
  nodes.push(warningsBox(result.warnings));
  if (completed) {
    if (kind === 'image') nodes.push(h('div', { class: 'two-col' }, classBars(result.evidence), null), imageVisuals(result));
    if (kind === 'video') nodes.push(...videoSections(result));
    nodes.push(evidenceTable(result));
  }
  nodes.push(h('section', { class: 'result-section' }, limitationsBox(result.limitations)), metaFooter(result));
  panel.replaceChildren(...nodes.filter(Boolean));
}

/* ---------- start-up ---------- */
window.addEventListener('beforeunload', () => objectUrls.forEach(url => URL.revokeObjectURL(url)));
Object.keys(TYPE_CONFIG).forEach(renderEmptyResult);
renderConnection(); renderStats(); renderBands(); renderRecent(); renderHistory(); renderModelStatus(); renderAudioNotice(); updateAnalyzeButtons();
if (location.hash && PAGE_NAMES[location.hash.slice(1)]) navigate(location.hash.slice(1));
checkBackend();
setInterval(checkBackend, 15000);
window.addEventListener('focus', checkBackend);