"""
Semantic Image Search — local, private, fast
Uses CLIP (via transformers + torch) to index images and search by text.
Run: python app.py
Then open http://localhost:5000
"""

import os, sys, json, base64, time, hashlib, pickle, threading
from pathlib import Path
from io import BytesIO

# ── auto-install dependencies ────────────────────────────────────────────────
def install(pkg):
    import subprocess
    print(f"[setup] Installing {pkg}…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    install("flask"); from flask import Flask, request, jsonify, send_from_directory

try:
    from PIL import Image
except ImportError:
    install("Pillow"); from PIL import Image

try:
    import numpy as np
except ImportError:
    install("numpy"); import numpy as np

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    print("[setup] Installing torch + transformers (this may take a minute)…")
    install("torch"); install("transformers")
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        CLIP_AVAILABLE = True
    except Exception as e:
        print(f"[warn] CLIP unavailable: {e}. Falling back to colour histogram search.")
        CLIP_AVAILABLE = False

# ── config ───────────────────────────────────────────────────────────────────
INDEX_FILE   = Path("image_index.pkl")
IMAGE_FOLDER = Path(os.environ.get("IMAGE_FOLDER", "./images"))
SUPPORTED    = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
THUMB_SIZE   = (300, 300)
MODEL_NAME   = "openai/clip-vit-base-patch32"

# ── model loader (lazy, thread-safe) ─────────────────────────────────────────
_model_lock  = threading.Lock()
_model       = None
_processor   = None

def get_model():
    global _model, _processor
    if not CLIP_AVAILABLE:
        return None, None
    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"[model] Loading {MODEL_NAME} …")
                _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
                _model     = CLIPModel.from_pretrained(MODEL_NAME)
                _model.eval()
                print("[model] Ready ✓")
    return _model, _processor

# ── index ─────────────────────────────────────────────────────────────────────
# { path_str: {"embedding": np.array, "mtime": float, "thumb": b64_str} }
_index: dict = {}
_index_lock  = threading.Lock()

def _file_hash(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()

def _to_vec(feat) -> "np.ndarray":
    """Safely extract a flat float32 vector from model output.
    Handles both plain tensors and dataclass wrappers like
    BaseModelOutputWithPooling that some transformers versions return."""
    import torch as _t
    if isinstance(feat, _t.Tensor):
        t = feat
    elif hasattr(feat, 'pooler_output') and feat.pooler_output is not None:
        t = feat.pooler_output          # BERT-style pooled CLS
    elif hasattr(feat, 'last_hidden_state'):
        t = feat.last_hidden_state[:, 0, :]  # CLS token
    elif hasattr(feat, '__getitem__'):
        t = feat[0]                     # tuple/list fallback
    else:
        raise TypeError(f'Cannot extract tensor from {type(feat)}')
    v = t.reshape(-1).float().detach().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)

def _embed_image_clip(img: Image.Image) -> np.ndarray:
    model, processor = get_model()
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
    return _to_vec(feat)

def _embed_text_clip(text: str) -> np.ndarray:
    model, processor = get_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        feat = model.get_text_features(**inputs)
    return _to_vec(feat)

# Fallback: colour histogram embedding
def _embed_image_histogram(img: Image.Image) -> np.ndarray:
    img_rgb = img.convert("RGB").resize((64, 64))
    arr = np.array(img_rgb, dtype=np.float32)
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=32, range=(0, 256))
        hist.append(h)
    v = np.concatenate(hist).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def _embed_text_histogram(text: str) -> np.ndarray:
    """Keyword colour matching fallback."""
    colour_map = {
        "red": [1,0,0], "green": [0,1,0], "blue": [0,0,1],
        "yellow": [1,1,0], "purple": [0.5,0,0.5], "orange": [1,0.5,0],
        "pink": [1,0.75,0.8], "white": [1,1,1], "black": [0,0,0],
        "grey": [0.5,0.5,0.5], "gray": [0.5,0.5,0.5], "brown": [0.6,0.3,0],
        "cyan": [0,1,1], "dark": [0.1,0.1,0.1], "bright": [0.9,0.9,0.9],
    }
    words = text.lower().split()
    rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    for w in words:
        if w in colour_map:
            rgb = np.array(colour_map[w], dtype=np.float32)
            break
    # Tile to match histogram size (3 channels × 32 bins = 96)
    v = np.tile(rgb, 32).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def embed_image(img):
    return _embed_image_clip(img) if CLIP_AVAILABLE else _embed_image_histogram(img)

def embed_text(text):
    return _embed_text_clip(text) if CLIP_AVAILABLE else _embed_text_histogram(text)

def _thumb_b64(img: Image.Image) -> str:
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

def load_index():
    global _index
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "rb") as f:
            _index = pickle.load(f)
        # Check for stale shape entries and drop them
        bad = [k for k, v in _index.items() if v.get("embedding") is None or v["embedding"].ndim != 1]
        if bad:
            print(f"[index] Dropping {len(bad)} stale entries with wrong embedding shape. Re-index to rebuild.")
            for k in bad:
                del _index[k]
        print(f"[index] Loaded {len(_index)} entries from cache.")

def save_index():
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(_index, f)

def index_folder(folder: Path, progress_cb=None):
    """Index all images in folder (and subdirs), skipping unchanged files."""
    import traceback as _tb
    folder = folder.resolve()
    if not folder.exists():
        return {"error": f"Folder not found: {folder}"}

    files = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED]
    total, indexed, skipped, errors = len(files), 0, 0, 0
    last_error = ""

    print(f"[index] Found {total} image files in {folder}")

    for i, path in enumerate(files):
        key = str(path)
        mtime = path.stat().st_mtime

        if key in _index and _index[key].get("mtime") == mtime:
            skipped += 1
            if progress_cb:
                progress_cb(i + 1, total, str(path.name), "skip")
            continue
        try:
            img = Image.open(path).convert("RGB")
            img.load()                          # force decode now, catch corrupt files early
            emb = embed_image(img.copy())       # copy so thumbnail() doesn't destroy it
            thumb = _thumb_b64(img)
            with _index_lock:
                _index[key] = {"embedding": emb, "mtime": mtime, "thumb": thumb,
                                "name": path.name, "folder": str(path.parent)}
            indexed += 1
            if progress_cb:
                progress_cb(i + 1, total, str(path.name), "ok")
        except Exception as e:
            errors += 1
            last_error = f"{path.name}: {e}"
            print(f"[index] ERROR {path.name}: {e}")
            _tb.print_exc()
            if progress_cb:
                progress_cb(i + 1, total, str(path.name), f"err:{e}")

    save_index()
    result = {"total": total, "indexed": indexed, "skipped": skipped, "errors": errors}
    if last_error:
        result["last_error"] = last_error
    print(f"[index] Done: {result}")
    return result

# ── indexing thread ───────────────────────────────────────────────────────────
_index_status = {"running": False, "progress": 0, "total": 0, "current": "", "done": False, "result": None}

def _run_index(folder):
    _index_status.update({"running": True, "done": False, "progress": 0, "result": None, "last_error": ""})
    def cb(done, total, name, status):
        _index_status["progress"] = done
        _index_status["total"]    = total
        _index_status["current"]  = name
        if status.startswith("err:"):
            _index_status["last_error"] = f"{name}: {status[4:]}"
    result = index_folder(folder, progress_cb=cb)
    _index_status.update({"running": False, "done": True, "result": result})

# ── search ────────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = 10):
    if not query.strip():
        return []
    if not _index:
        return []
    q_emb = embed_text(query)
    scores = []
    with _index_lock:
        items = list(_index.items())
    for path, data in items:
        emb = data["embedding"].flatten()
        if emb.shape != q_emb.shape:
            continue  # skip stale entries with wrong shape
        sim = float(np.dot(q_emb, emb))
        scores.append((sim, path, data))
    scores.sort(reverse=True)
    results = []
    for sim, path, data in scores[:top_k]:
        results.append({
            "path":   path,
            "name":   data.get("name", Path(path).name),
            "folder": data.get("folder", ""),
            "score":  round(sim, 4),
            "thumb":  data.get("thumb", ""),
        })
    return results

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lens — Semantic Image Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:      #0c0c0e;
  --surface: #14141a;
  --border:  #2a2a36;
  --accent:  #7c6af7;
  --accent2: #c8b8ff;
  --text:    #e8e6f0;
  --muted:   #6b6880;
  --green:   #4ade80;
  --red:     #f87171;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; }

/* ── Layout ── */
#app { display: grid; grid-template-rows: auto 1fr; height: 100vh; }
header { padding: 1.5rem 2rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }
h1 { font-family: 'DM Serif Display', serif; font-size: 1.6rem; letter-spacing: -0.02em; color: var(--accent2); white-space: nowrap; }
h1 span { font-style: italic; color: var(--muted); }

/* ── Search bar ── */
#search-wrap { flex: 1; min-width: 200px; position: relative; }
#search { width: 100%; background: var(--surface); border: 1px solid var(--border); color: var(--text); font-family: 'DM Mono', monospace; font-size: 0.95rem; padding: 0.6rem 2.5rem 0.6rem 1rem; border-radius: 8px; outline: none; transition: border-color .2s; }
#search:focus { border-color: var(--accent); }
#search::placeholder { color: var(--muted); }
#search-clear { position: absolute; right: 0.7rem; top: 50%; transform: translateY(-50%); background: none; border: none; color: var(--muted); cursor: pointer; font-size: 1.1rem; display: none; }
#search-clear.visible { display: block; }

/* ── Folder controls ── */
#folder-row { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
#folder-input { background: var(--surface); border: 1px solid var(--border); color: var(--text); font-family: 'DM Mono', monospace; font-size: 0.8rem; padding: .45rem .8rem; border-radius: 6px; width: 260px; outline: none; }
#folder-input:focus { border-color: var(--accent); }
#btn-index { background: var(--accent); color: #fff; border: none; font-family: 'DM Mono', monospace; font-size: 0.8rem; padding: .45rem 1rem; border-radius: 6px; cursor: pointer; transition: opacity .2s; }
#btn-index:hover { opacity: .85; }
#btn-index:disabled { opacity: .4; cursor: default; }

/* ── Progress / status ── */
#status-bar { font-size: 0.75rem; color: var(--muted); padding: 0.4rem 2rem; border-bottom: 1px solid var(--border); min-height: 28px; display: flex; align-items: center; gap: .8rem; }
#progress-fill-wrap { flex: 1; max-width: 200px; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; display: none; }
#progress-fill { height: 100%; background: var(--accent); width: 0%; transition: width .3s; border-radius: 2px; }

/* ── Results grid ── */
#main { overflow-y: auto; padding: 1.5rem 2rem; }
#results-meta { font-size: 0.75rem; color: var(--muted); margin-bottom: 1rem; }
#grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 1rem; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; cursor: pointer; transition: transform .15s, border-color .15s; position: relative; }
.card:hover { transform: translateY(-3px); border-color: var(--accent); }
.card img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #1a1a24; }
.card-info { padding: .5rem .65rem .6rem; }
.card-name { font-size: 0.72rem; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card-score { font-size: 0.68rem; color: var(--muted); margin-top: .15rem; }
.score-bar { height: 2px; background: var(--border); border-radius: 1px; margin-top: .35rem; overflow: hidden; }
.score-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 1px; }

/* ── Empty states ── */
#empty { text-align: center; padding: 4rem 2rem; color: var(--muted); display: none; }
#empty .icon { font-size: 3rem; margin-bottom: 1rem; }
#empty p { font-size: 0.85rem; line-height: 1.7; }

/* ── Lightbox ── */
#lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.88); z-index: 100; justify-content: center; align-items: center; }
#lightbox.open { display: flex; }
#lightbox-inner { max-width: 90vw; max-height: 90vh; position: relative; }
#lightbox img { max-width: 90vw; max-height: 80vh; border-radius: 8px; display: block; }
#lightbox-caption { color: var(--muted); font-size: 0.75rem; margin-top: .6rem; text-align: center; }
#lightbox-close { position: absolute; top: -2rem; right: 0; background: none; border: none; color: var(--text); font-size: 1.4rem; cursor: pointer; }

/* ── Chip indicator ── */
.chip { display: inline-block; padding: .1rem .55rem; border-radius: 99px; font-size: 0.68rem; font-weight: 500; }
.chip-green { background: rgba(74,222,128,.15); color: var(--green); border: 1px solid rgba(74,222,128,.25); }
.chip-red   { background: rgba(248,113,113,.15); color: var(--red);   border: 1px solid rgba(248,113,113,.25); }
.chip-accent { background: rgba(124,106,247,.2); color: var(--accent2); border: 1px solid rgba(124,106,247,.3); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg); } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Animations ── */
@keyframes fadeUp { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform: translateY(0); } }
.card { animation: fadeUp .2s ease both; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>Lens <span>/ image search</span></h1>

    <div id="search-wrap">
      <input id="search" type="text" placeholder="Describe what you're looking for…" autocomplete="off">
      <button id="search-clear" title="Clear">✕</button>
    </div>

    <div id="folder-row">
      <input id="folder-input" type="text" placeholder="./images" value="./images">
      <button id="btn-index">⊕ Index folder</button>
    </div>
  </header>

  <div id="status-bar">
    <span id="status-text">No index loaded. Add a folder path and click Index.</span>
    <div id="progress-fill-wrap"><div id="progress-fill"></div></div>
  </div>

  <div id="main">
    <div id="results-meta"></div>
    <div id="grid"></div>
    <div id="empty">
      <div class="icon">🔍</div>
      <p id="empty-msg">Start typing to search your images.<br>Results update as you type.</p>
    </div>
  </div>
</div>

<!-- Lightbox -->
<div id="lightbox">
  <div id="lightbox-inner">
    <button id="lightbox-close">✕</button>
    <img id="lightbox-img" src="" alt="">
    <div id="lightbox-caption"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let debounceTimer = null;
let lastQuery = "";

// ── Status helpers ────────────────────────────────────────────────────
function setStatus(msg, chip="") {
  $("status-text").innerHTML = msg + (chip ? " " + chip : "");
}
function chipHtml(label, type="accent") {
  return `<span class="chip chip-${type}">${label}</span>`;
}

// ── Search ────────────────────────────────────────────────────────────
$("search").addEventListener("input", e => {
  const q = e.target.value.trim();
  $("search-clear").classList.toggle("visible", q.length > 0);
  clearTimeout(debounceTimer);
  if (!q) { clearResults(); return; }
  debounceTimer = setTimeout(() => doSearch(q), 120);
});

$("search-clear").addEventListener("click", () => {
  $("search").value = "";
  $("search-clear").classList.remove("visible");
  clearResults();
  $("search").focus();
});

async function doSearch(query) {
  if (query === lastQuery) return;
  lastQuery = query;
  try {
    const t0 = performance.now();
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&k=40`);
    const data = await res.json();
    const ms = (performance.now() - t0).toFixed(0);
    renderResults(data.results, query, ms);
  } catch(e) {
    setStatus("Search error: " + e.message, chipHtml("error","red"));
  }
}

function clearResults() {
  lastQuery = "";
  $("grid").innerHTML = "";
  $("results-meta").textContent = "";
  $("empty").style.display = "block";
  $("empty-msg").textContent = "Start typing to search your images.\nResults update as you type.";
}

function renderResults(results, query, ms) {
  const grid = $("grid");
  grid.innerHTML = "";

  if (!results.length) {
    $("empty").style.display = "block";
    $("empty-msg").textContent = `No results for "${query}". Try different keywords.`;
    $("results-meta").textContent = "";
    return;
  }

  $("empty").style.display = "none";
  $("results-meta").textContent = `${results.length} results · ${ms}ms`;

  const maxScore = results[0]?.score || 1;

  results.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = "card";
    card.style.animationDelay = `${Math.min(i * 18, 300)}ms`;
    const pct = Math.round((r.score / maxScore) * 100);
    card.innerHTML = `
      <img src="data:image/jpeg;base64,${r.thumb}" alt="${r.name}" loading="lazy">
      <div class="card-info">
        <div class="card-name" title="${r.path}">${r.name}</div>
        <div class="card-score">${(r.score*100).toFixed(1)}% match</div>
        <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
      </div>
    `;
    card.addEventListener("click", () => openLightbox(r));
    grid.appendChild(card);
  });
}

// ── Lightbox ──────────────────────────────────────────────────────────
function openLightbox(r) {
  $("lightbox-img").src = `data:image/jpeg;base64,${r.thumb}`;
  $("lightbox-caption").textContent = `${r.name}  ·  ${r.folder}`;
  $("lightbox").classList.add("open");
}
$("lightbox-close").addEventListener("click", () => $("lightbox").classList.remove("open"));
$("lightbox").addEventListener("click", e => { if (e.target === $("lightbox")) $("lightbox").classList.remove("open"); });
document.addEventListener("keydown", e => { if (e.key === "Escape") $("lightbox").classList.remove("open"); });

// ── Indexing ──────────────────────────────────────────────────────────
$("btn-index").addEventListener("click", startIndexing);

async function startIndexing() {
  const folder = $("folder-input").value.trim() || "./images";
  $("btn-index").disabled = true;
  $("progress-fill-wrap").style.display = "block";
  $("progress-fill").style.width = "0%";
  setStatus("Starting indexer…");

  try {
    await fetch("/api/index", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({folder})
    });
    pollProgress();
  } catch(e) {
    setStatus("Failed to start indexer: " + e.message, chipHtml("error","red"));
    $("btn-index").disabled = false;
  }
}

function pollProgress() {
  const iv = setInterval(async () => {
    try {
      const res = await fetch("/api/index/status");
      const s = await res.json();
      if (s.total > 0) {
        const pct = Math.round((s.progress / s.total) * 100);
        $("progress-fill").style.width = pct + "%";
        setStatus(`Indexing… ${s.progress}/${s.total} — ${s.current}`);
      }
      if (s.done) {
        clearInterval(iv);
        $("btn-index").disabled = false;
        $("progress-fill").style.width = "100%";
        const r = s.result || {};
        const hasErrors = (r.errors || 0) > 0;
        const errDetail = s.last_error ? ` — last: ${s.last_error}` : "";
        if (r.indexed === 0 && hasErrors) {
          setStatus(
            `All ${r.errors} files failed${errDetail}. Check terminal for full traceback.`,
            chipHtml("✗ errors", "red")
          );
        } else {
          setStatus(
            `Index ready — ${r.indexed||0} new, ${r.skipped||0} cached` +
            (hasErrors ? `, ${r.errors} errors${errDetail}` : ""),
            chipHtml(hasErrors ? "⚠ partial" : "✓ ready", hasErrors ? "accent" : "green")
          );
        }
        setTimeout(() => { $("progress-fill-wrap").style.display = "none"; }, 2000);
        // refresh search if active
        if (lastQuery) { lastQuery=""; doSearch($("search").value.trim()); }
      }
    } catch(e) { clearInterval(iv); }
  }, 600);
}

// ── On load: get index stats ──────────────────────────────────────────
(async () => {
  try {
    const res = await fetch("/api/stats");
    const s = await res.json();
    if (s.count > 0) {
      setStatus(`${s.count} images indexed`, chipHtml("ready","green"));
      $("empty").style.display = "block";
    }
  } catch(e) {}
})();
</script>
</body>
</html>
"""

# ── JSON encoder that handles numpy scalars ───────────────────────────────────
import json as _json

class _SafeEncoder(_json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as _np
            if isinstance(obj, _np.floating):
                return float(obj)
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

app.json_encoder = _SafeEncoder  # type: ignore

# ── Global error handler — always return JSON, never HTML ─────────────────────
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    print(f"[error] {e}\n{tb}")
    return jsonify({"error": str(e), "results": [], "count": 0}), 500

@app.route("/")
def index():
    return HTML, 200, {"Content-Type": "text/html"}

@app.route("/api/search")
def api_search():
    try:
        q = request.args.get("q", "").strip()
        k = min(int(request.args.get("k", 10)), 100)
        results = search(q, top_k=k)
        # Ensure all score values are plain Python floats
        for r in results:
            r["score"] = float(r["score"])
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "results": [], "count": 0}), 500

@app.route("/api/index", methods=["POST"])
def api_index():
    try:
        data   = request.get_json(force=True) or {}
        folder = Path(data.get("folder", "./images"))
        if _index_status["running"]:
            return jsonify({"error": "Already indexing"}), 409
        t = threading.Thread(target=_run_index, args=(folder,), daemon=True)
        t.start()
        return jsonify({"status": "started", "folder": str(folder)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/index/status")
def api_index_status():
    # Sanitise _index_status so it's always JSON-safe
    safe = {k: v for k, v in _index_status.items() if k != "result"}
    safe["result"] = _index_status.get("result") or {}
    return jsonify(safe)

@app.route("/api/stats")
def api_stats():
    return jsonify({"count": len(_index), "clip": CLIP_AVAILABLE})

if __name__ == "__main__":
    IMAGE_FOLDER.mkdir(exist_ok=True)
    load_index()
    engine = "CLIP (openai/clip-vit-base-patch32)" if CLIP_AVAILABLE else "colour histogram (fallback)"
    print(f"""
╔══════════════════════════════════════════════╗
║   Lens — Semantic Image Search               ║
║   Engine : {engine:<34}║
║   Open   : http://localhost:5000             ║
╚══════════════════════════════════════════════╝
""")
    app.run(debug=True, port=5000, threaded=True)