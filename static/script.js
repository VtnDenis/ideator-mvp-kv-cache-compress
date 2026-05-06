const STATE = {
  model: 'llama-7b',
  seqLen: 2048,
  ratio: 4.0,
  dtype: 'fp16',
};

const MODEL_CONFIGS = {
  "gpt2": { layers: 12, heads: 12, head_dim: 64, dtype: "fp16" },
  "gpt2-medium": { layers: 24, heads: 16, head_dim: 64, dtype: "fp16" },
  "gpt2-large": { layers: 36, heads: 20, head_dim: 64, dtype: "fp16" },
  "gpt2-xl": { layers: 48, heads: 25, head_dim: 64, dtype: "fp16" },
  "llama-7b": { layers: 32, heads: 32, head_dim: 128, dtype: "fp16" },
  "llama-13b": { layers: 40, heads: 40, head_dim: 128, dtype: "fp16" },
  "llama-70b": { layers: 80, heads: 64, head_dim: 128, dtype: "fp16" },
};

function $(s) { return document.querySelector(s); }
function $$(s) { return document.querySelectorAll(s); }

function showToast(msg) {
  const toast = $('#toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(function() { toast.classList.remove('show'); }, 2200);
}

function formatNumber(n) {
  return n.toLocaleString('en-US');
}

function formatMB(n) {
  if (n >= 1024) return (n / 1024).toFixed(2) + ' GB';
  return n.toFixed(1) + ' MB';
}

function getDtypeBytes() {
  switch (STATE.dtype) {
    case 'fp32': return 4;
    case 'bf16': return 2;
    case 'fp16': default: return 2;
  }
}

function estimateMemory(config) {
  var cacheSize = config.layers * 2 * config.heads * config.seqLen * config.headDim * config.dtypeBytes;
  var compressedSize = cacheSize / config.ratio;
  var saved = cacheSize - compressedSize;
  return {
    layers: config.layers,
    heads: config.heads,
    headDim: config.headDim,
    seqLen: config.seqLen,
    originalMB: cacheSize / (1024 * 1024),
    compressedMB: compressedSize / (1024 * 1024),
    savedMB: saved / (1024 * 1024),
    ratio: config.ratio,
    reductionPercent: (1 - 1 / config.ratio) * 100,
  };
}

function applyModelPreset(modelKey) {
  var cfg = MODEL_CONFIGS[modelKey];
  if (!cfg) return;
  $('#seqLenSlider').setAttribute('data-layers', cfg.layers);
  $('#seqLenSlider').setAttribute('data-heads', cfg.heads);
  $('#seqLenSlider').setAttribute('data-headdim', cfg.head_dim);

  $$('.radio-label').forEach(function(l) {
    l.classList.remove('active');
  });

  var dtypeBtn = document.querySelector('.radio-label[data-value="' + cfg.dtype + '"]');
  if (dtypeBtn) {
    dtypeBtn.classList.add('active');
    var radio = dtypeBtn.querySelector('input');
    if (radio) radio.checked = true;
    STATE.dtype = cfg.dtype;
  }
}

function calculate() {
  var el = $('#seqLenSlider');
  var layers = parseInt(el.getAttribute('data-layers') || '32');
  var heads = parseInt(el.getAttribute('data-heads') || '32');
  var headDim = parseInt(el.getAttribute('data-headdim') || '128');

  var result = estimateMemory({
    layers: layers,
    heads: heads,
    headDim: headDim,
    seqLen: STATE.seqLen,
    ratio: STATE.ratio,
    dtypeBytes: getDtypeBytes(),
  });

  renderResults(result);
}

function renderResults(data) {
  var container = $('#demoResults');
  if (!container) return;

  container.innerHTML =
    '<div class="results-grid">' +
      '<div class="result-item">' +
        '<div class="result-item-label">Original Cache</div>' +
        '<div class="result-item-value">' + formatMB(data.originalMB) + '</div>' +
        '<div class="result-item-sub">' + data.layers + ' layers x ' + data.heads + ' heads x ' + data.headDim + 'd</div>' +
      '</div>' +
      '<div class="result-item">' +
        '<div class="result-item-label">Compressed Cache</div>' +
        '<div class="result-item-value" style="color: var(--accent)">' + formatMB(data.compressedMB) + '</div>' +
        '<div class="result-item-sub">' + formatNumber(data.seqLen) + ' tokens</div>' +
      '</div>' +
      '<div class="result-item">' +
        '<div class="result-item-label">Memory Saved</div>' +
        '<div class="result-item-value">' + formatMB(data.savedMB) + '</div>' +
        '<div class="result-item-sub">' + Math.round(data.reductionPercent) + '% reduction</div>' +
      '</div>' +
      '<div class="result-item">' +
        '<div class="result-item-label">Compression Ratio</div>' +
        '<div class="result-item-value">' + data.ratio.toFixed(1) + 'x</div>' +
        '<div class="result-item-sub">eOptShrinkQ method</div>' +
      '</div>' +
      '<div class="result-item wide">' +
        '<div class="result-item-label">Memory Usage</div>' +
        '<div class="result-bar-wrap">' +
          '<div class="result-bar" style="width: ' + (100 - data.reductionPercent) + '%"></div>' +
        '</div>' +
        '<div class="result-item-sub" style="margin-top: 8px; display: flex; justify-content: space-between;">' +
          '<span>Compressed: ' + formatMB(data.compressedMB) + '</span>' +
          '<span>Original: ' + formatMB(data.originalMB) + '</span>' +
        '</div>' +
      '</div>' +
    '</div>';
}

$('#modelSelect').addEventListener('change', function(e) {
  STATE.model = e.target.value;
  applyModelPreset(STATE.model);
});

$('#seqLenSlider').addEventListener('input', function(e) {
  STATE.seqLen = parseInt(e.target.value);
  $('#seqLenValue').textContent = formatNumber(STATE.seqLen);
});

$('#ratioSlider').addEventListener('input', function(e) {
  STATE.ratio = parseFloat(e.target.value);
  $('#ratioValue').textContent = STATE.ratio.toFixed(1) + 'x';
});

$$('.radio-label').forEach(function(label) {
  label.addEventListener('click', function() {
    $$('.radio-label').forEach(function(l) { l.classList.remove('active'); });
    this.classList.add('active');
    var input = this.querySelector('input');
    input.checked = true;
    STATE.dtype = input.value;
  });
});

$('#calculateBtn').addEventListener('click', calculate);

function initHeroCanvas() {
  var canvas = $('#heroCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width;
  var H = canvas.height;

  function draw() {
    ctx.clearRect(0, 0, W, H);

    var time = Date.now() * 0.001;
    var bars = 14;
    var barW = 18;
    var gap = (W - bars * barW) / (bars + 1);

    for (var i = 0; i < bars; i++) {
      var x = gap + i * (barW + gap);
      var baseH = 120;
      var amp = 60 + Math.sin(time * 0.8 + i * 0.7) * 40 + Math.sin(time * 1.3 + i * 1.1) * 30;
      var h = baseH + amp;
      var y = H - h - 60;

      var compH = h * 0.25 + Math.sin(time * 0.6 + i * 0.5) * 15;
      var compY = H - compH - 60;

      ctx.fillStyle = 'rgba(37, 44, 58, 0.6)';
      roundRect(ctx, x, y, barW, h, 4);
      ctx.fill();

      var gradient = ctx.createLinearGradient(x, compY, x, compY + compH);
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

    for (var j = 0; j < 6; j++) {
      var gy = 60 + j * (H - 120) / 5;
      ctx.strokeStyle = 'rgba(37, 44, 58, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(20, gy);
      ctx.lineTo(W - 20, gy);
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
  ctx.arcTo(x, y, x + r, y);
  ctx.closePath();
}

$('#installBtn').addEventListener('click', function(e) {
  e.preventDefault();
  var cmd = 'pip install kv-cache-compress';
  if (navigator.clipboard) {
    navigator.clipboard.writeText(cmd).then(function() {
      showToast('Copied: ' + cmd);
    }).catch(function() {
      showToast(cmd);
    });
  } else {
    showToast(cmd);
  }
});

function initScrollReveal() {
  try {
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.feature-card, .metric-card').forEach(function(el) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      el.style.transition = 'all 500ms cubic-bezier(0.4, 0, 0.2, 1)';
      observer.observe(el);
    });

    setTimeout(function() {
      document.querySelectorAll('.feature-card, .metric-card').forEach(function(el) {
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
      });
    }, 200);
  } catch(e) {}
}

initHeroCanvas();
initScrollReveal();

document.addEventListener('DOMContentLoaded', function() {
  $$('.metric-card .metric-value').forEach(function(el) {
    var target = el.getAttribute('data-count');
    if (!target) return;
    el.style.opacity = '0';
    el.style.transform = 'scale(0.5)';
    el.style.transition = 'all 600ms cubic-bezier(0.34, 1.56, 0.64, 1)';
    setTimeout(function() {
      el.style.opacity = '1';
      el.style.transform = 'scale(1)';
    }, 400);
  });
});
