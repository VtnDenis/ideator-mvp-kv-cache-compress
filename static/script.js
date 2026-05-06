const STATE = {
  model: 'llama-7b',
  seqLen: 2048,
  ratio: 4.0,
  dtype: 'fp16',
  modelConfigs: null,
};

function $(s) { return document.querySelector(s); }
function $$(s) { return document.querySelectorAll(s); }

async function showToast(msg) {
  const toast = $('#toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2200);
}

function formatNumber(n) {
  return n.toLocaleString('en-US');
}

function formatMB(n) {
  if (n >= 1024) return `${(n / 1024).toFixed(2)} GB`;
  return `${n.toFixed(1)} MB`;
}

async function loadModelConfigs() {
  try {
    const res = await fetch('/api/configs');
    STATE.modelConfigs = (await res.json()).models;
  } catch (e) {
    console.warn('Could not load configs, using defaults');
  }
}

function applyModelPreset(modelKey) {
  if (!STATE.modelConfigs || !STATE.modelConfigs[modelKey]) return;
  const cfg = STATE.modelConfigs[modelKey];
  $('#seqLenSlider').setAttribute('data-layers', cfg.layers);
  $('#seqLenSlider').setAttribute('data-heads', cfg.heads);
  $('#seqLenSlider').setAttribute('data-headdim', cfg.head_dim);
  const dtypeBtn = $(`.radio-label[data-value="${cfg.dtype}"]`);
  if (dtypeBtn) dtypeBtn.click();
}

$('#modelSelect').addEventListener('change', (e) => {
  STATE.model = e.target.value;
  applyModelPreset(STATE.model);
});

$('#seqLenSlider').addEventListener('input', (e) => {
  STATE.seqLen = parseInt(e.target.value);
  $('#seqLenValue').textContent = formatNumber(STATE.seqLen);
});

$('#ratioSlider').addEventListener('input', (e) => {
  STATE.ratio = parseFloat(e.target.value);
  $('#ratioValue').textContent = STATE.ratio.toFixed(1) + 'x';
});

$$('.radio-label').forEach(label => {
  label.addEventListener('click', function() {
    $$('.radio-label').forEach(l => l.classList.remove('active'));
    this.classList.add('active');
    const input = this.querySelector('input');
    input.checked = true;
    STATE.dtype = input.value;
  });
});

function getDtypeBytes() {
  switch (STATE.dtype) {
    case 'fp32': return 4;
    case 'bf16': return 2;
    case 'fp16': default: return 2;
  }
}

async function calculate() {
  const el = $('#seqLenSlider');
  const layers = parseInt(el.getAttribute('data-layers') || '32');
  const heads = parseInt(el.getAttribute('data-heads') || '32');
  const headDim = parseInt(el.getAttribute('data-headdim') || '128');

  const params = new URLSearchParams({
    layers: layers,
    heads: heads,
    head_dim: headDim,
    seq_len: STATE.seqLen,
    ratio: STATE.ratio,
    dtype: getDtypeBytes(),
  });

  try {
    const res = await fetch(`/api/estimate?${params.toString()}`);
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    showToast('Error calculating. Is the server running?');
  }
}

function renderResults(data) {
  const container = $('#demoResults');
  container.innerHTML = `
    <div class="results-grid">
      <div class="result-item">
        <div class="result-item-label">Original Cache</div>
        <div class="result-item-value">${formatMB(data.original_cache_MB)}</div>
        <div class="result-item-sub">${data.num_layers} layers x ${data.num_heads} heads x ${data.head_dim}d</div>
      </div>
      <div class="result-item">
        <div class="result-item-label">Compressed Cache</div>
        <div class="result-item-value" style="color: var(--accent)">${formatMB(data.compressed_cache_MB)}</div>
        <div class="result-item-sub">${data.sequence_length.toLocaleString()} tokens</div>
      </div>
      <div class="result-item">
        <div class="result-item-label">Memory Saved</div>
        <div class="result-item-value">${formatMB(data.memory_saved_MB)}</div>
        <div class="result-item-sub">${data.reduction_percent}% reduction</div>
      </div>
      <div class="result-item">
        <div class="result-item-label">Compression Ratio</div>
        <div class="result-item-value">${data.compression_ratio}x</div>
        <div class="result-item-sub">eOptShrinkQ method</div>
      </div>
      <div class="result-item wide">
        <div class="result-item-label">Memory Usage</div>
        <div class="result-bar-wrap">
          <div class="result-bar" style="width: ${100 - data.reduction_percent}%"></div>
        </div>
        <div class="result-item-sub" style="margin-top: 8px; display: flex; justify-content: space-between;">
          <span>Compressed: ${formatMB(data.compressed_cache_MB)}</span>
          <span>Original: ${formatMB(data.original_cache_MB)}</span>
        </div>
      </div>
    </div>
  `;
}

$('#calculateBtn').addEventListener('click', calculate);

function initHeroCanvas() {
  const canvas = $('#heroCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  function draw() {
    ctx.clearRect(0, 0, W, H);

    const time = Date.now() * 0.001;
    const bars = 14;
    const barW = 18;
    const gap = (W - bars * barW) / (bars + 1);

    for (let i = 0; i < bars; i++) {
      const x = gap + i * (barW + gap);
      const baseH = 120;
      const amp = 60 + Math.sin(time * 0.8 + i * 0.7) * 40 + Math.sin(time * 1.3 + i * 1.1) * 30;
      const h = baseH + amp;
      const y = H - h - 60;

      const compH = h * 0.25 + Math.sin(time * 0.6 + i * 0.5) * 15;
      const compY = H - compH - 60;

      ctx.fillStyle = 'rgba(37, 44, 58, 0.6)';
      roundRect(ctx, x, y, barW, h, 4);
      ctx.fill();

      const gradient = ctx.createLinearGradient(x, compY, x, compY + compH);
      gradient.addColorStop(0, '#00e5a0');
      gradient.addColorStop(1, '#00b884');
      ctx.fillStyle = gradient;
      roundRect(ctx, x, compY, barW, compH, 4);
      ctx.fill();

      ctx.fillStyle = 'rgba(0, 229, 160, 0.15)';
      roundRect(ctx, x - 2, compY - 2, barW + 4, compH + 4, 4);
      ctx.fill();

      ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
      roundRect(ctx, x - 1, compY - 1, barW + 2, 2, 1);
      ctx.fill();
    }

    for (let i = 0; i < 6; i++) {
      const y = 60 + i * (H - 120) / 5;
      ctx.strokeStyle = 'rgba(37, 44, 58, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(20, y);
      ctx.lineTo(W - 20, y);
      ctx.stroke();
    }

    requestAnimationFrame(draw);
  }

  draw();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

$('#installBtn').addEventListener('click', (e) => {
  e.preventDefault();
  const cmd = 'pip install kv-cache-compress';
  navigator.clipboard.writeText(cmd).then(() => {
    showToast('Copied: ' + cmd);
  }).catch(() => {
    showToast(cmd);
  });
});

function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.feature-card, .metric-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'all 500ms cubic-bezier(0.4, 0, 0.2, 1)';
    observer.observe(el);
  });

  setTimeout(() => {
    document.querySelectorAll('.feature-card, .metric-card').forEach(el => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    });
  }, 200);
}

loadModelConfigs();
initHeroCanvas();
initScrollReveal();

document.addEventListener('DOMContentLoaded', () => {
  $$('.metric-card .metric-value').forEach(el => {
    const target = el.getAttribute('data-count');
    if (!target) return;
    el.style.opacity = '0';
    el.style.transform = 'scale(0.5)';
    el.style.transition = 'all 600ms cubic-bezier(0.34, 1.56, 0.64, 1)';
    setTimeout(() => {
      el.style.opacity = '1';
      el.style.transform = 'scale(1)';
    }, 400);
  });
});
