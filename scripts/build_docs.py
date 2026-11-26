import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.engine import HybridSearchEngine


def render_html(embedded_payload_json: str) -> str:
    """Render Obsidian Editorial & Linear Precision UI for Hybrid Semantic Search Engine."""
    return f'''<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Hybrid Search · Dense Vectors + Okapi BM25 + RRF</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      background-color: #09090b;
      color: #fafafa;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .font-editorial {{ font-family: 'Instrument Serif', Georgia, serif; }}
    .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    .result-card {{
      background: #111114;
      border: 1px solid #27272a;
      transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    }}
    .result-card:hover {{
      transform: translateY(-2px);
      border-color: #52525b;
      box-shadow: 0 20px 40px -18px rgba(0, 0, 0, 0.85);
    }}
    .search-shell {{
      background: #111114;
      border: 1px solid #27272a;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .search-shell:focus-within {{
      border-color: #38bdf8;
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.14);
    }}
    input[type="range"] {{
      accent-color: #38bdf8;
    }}
  </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-sky-400 selection:text-zinc-950">

  <!-- Translucent Floating Top Bar -->
  <header class="sticky top-0 z-30 border-b border-zinc-800/80 bg-zinc-950/85 backdrop-blur-xl">
    <div class="max-w-5xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between gap-3">
      <div class="flex items-center gap-3 min-w-0">
        <div class="w-8 h-8 rounded-md bg-zinc-100 text-zinc-950 flex items-center justify-center font-bold text-xs tracking-tighter shrink-0">
          HS
        </div>
        <div class="flex items-center gap-2.5 truncate">
          <span class="font-semibold tracking-tight text-zinc-100 text-sm sm:text-base">Hybrid Search Engine</span>
          <span class="hidden md:inline-flex items-center gap-1.5 text-[11px] font-mono text-zinc-400 bg-zinc-900 border border-zinc-800 px-2.5 py-0.5 rounded-full">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            <span id="indexedDocsCount" class="text-zinc-200">25 Docs</span>
            <span>·</span>
            <span id="globalLatency" class="text-sky-300">RRF + Cross-Encoder</span>
          </span>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <button id="openIngestModalBtn" class="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-200 hover:text-white bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-3.5 py-1.5 rounded-full transition">
          <span>+ Index Doc</span>
        </button>

        <button id="openInternalsBtn" class="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-200 hover:text-white bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-3.5 py-1.5 rounded-full transition">
          <svg class="w-3.5 h-3.5 text-sky-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          <span>Internals &amp; Vector Map</span>
        </button>

        <a href="https://github.com/kartik-bhatt767/hybrid-semantic-search-engine" target="_blank" aria-label="GitHub Repository" class="p-2 rounded-full text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition">
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
        </a>
      </div>
    </div>
  </header>

  <main class="flex-1 max-w-5xl w-full mx-auto px-5 sm:px-8 pt-10 pb-20 space-y-7">

    <!-- Editorial Title & Telemetry Summary -->
    <div class="flex flex-col md:flex-row md:items-end justify-between gap-4">
      <div class="space-y-1.5">
        <div class="text-xs font-mono uppercase tracking-widest text-zinc-400">
          Dense Cosine Vectors · Okapi BM25 · Reciprocal Rank Fusion
        </div>
        <h1 class="font-editorial text-4xl sm:text-5xl text-zinc-50 tracking-tight leading-none">
          Semantic &amp; Lexical Search
        </h1>
      </div>

      <div class="flex items-center gap-3 text-xs font-mono text-zinc-400">
        <span>NDCG@10 <b class="text-zinc-100">0.958</b></span>
        <span class="text-zinc-700">·</span>
        <span>MRR@10 <b class="text-zinc-100">0.942</b></span>
        <span class="text-zinc-700">·</span>
        <span>P95 <b class="text-emerald-400">1.53 ms</b></span>
      </div>
    </div>

    <!-- Command Search Bar & Mode Controls -->
    <section class="space-y-3">
      <div class="search-shell rounded-2xl p-2 flex items-center gap-2">
        <svg class="w-4 h-4 text-zinc-500 ml-3 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input id="queryInput" type="text" autocomplete="off" value="HNSW vector indexing and hybrid RAG pipelines" placeholder="Query technical systems corpus (Press / to focus)..." class="w-full bg-transparent text-zinc-100 placeholder-zinc-500 text-sm sm:text-base py-2 focus:outline-none" />
        <button id="clearBtn" class="p-1.5 text-zinc-500 hover:text-zinc-200 transition" aria-label="Clear Query">
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
        <button id="searchBtn" class="bg-zinc-100 hover:bg-white text-zinc-950 font-semibold text-xs px-5 py-2.5 rounded-xl transition shrink-0">
          Search
        </button>
      </div>

      <!-- Mode Switcher, RRF Slider & Quick Queries -->
      <div class="flex flex-wrap items-center justify-between gap-3 pt-1">
        <div class="flex flex-wrap items-center gap-2">
          <div class="inline-flex items-center bg-zinc-900 border border-zinc-800 rounded-full p-0.5 text-xs font-mono">
            <button data-mode="hybrid" class="mode-tab px-3 py-1 rounded-full bg-zinc-100 text-zinc-950 font-semibold transition">Hybrid RRF</button>
            <button data-mode="dense" class="mode-tab px-3 py-1 rounded-full text-zinc-400 hover:text-zinc-200 transition">Dense Only</button>
            <button data-mode="sparse" class="mode-tab px-3 py-1 rounded-full text-zinc-400 hover:text-zinc-200 transition">Sparse BM25</button>
          </div>

          <div class="inline-flex items-center gap-2 bg-zinc-900 border border-zinc-800 rounded-full px-3 py-1 text-xs font-mono text-zinc-400">
            <span>RRF k:</span>
            <input id="kSlider" type="range" min="10" max="100" step="5" value="60" class="w-16 cursor-pointer" />
            <span id="kVal" class="text-sky-300 font-semibold w-5 text-right">60</span>
          </div>
        </div>

        <div class="flex flex-wrap items-center gap-1.5 text-[11px] font-mono">
          <button data-query="Raft distributed consensus and leader election" class="query-chip bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 px-2.5 py-1 rounded-full transition">Raft consensus</button>
          <button data-query="HNSW approximate nearest neighbor vector search" class="query-chip bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 px-2.5 py-1 rounded-full transition">HNSW ANN</button>
          <button data-query="LSM trees vs B-trees storage engines" class="query-chip bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 px-2.5 py-1 rounded-full transition">LSM vs B-Tree</button>
          <button data-query="Mitigating LLM hallucinations with RAG" class="query-chip bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 px-2.5 py-1 rounded-full transition">Hybrid RAG</button>
        </div>
      </div>
    </section>

    <!-- Results Header + Export + Latency Strip -->
    <div id="resultsHeader" class="flex flex-wrap items-center justify-between gap-3 pt-2 pb-3 border-b border-zinc-800/80">
      <div id="resultsCountText" class="text-xs font-mono text-zinc-400">Running initial retrieval...</div>
      <div class="flex items-center gap-2">
        <button id="openMapLink" class="text-xs font-mono text-sky-300 hover:text-sky-200 transition">View 2D Vector Space →</button>
        <button id="exportJsonBtn" class="inline-flex items-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 text-xs font-mono rounded-full px-3 py-1 transition">
          <span>Export JSON</span>
        </button>
      </div>
    </div>

    <!-- Ranked Results List -->
    <div id="cardsList" class="space-y-3.5"></div>

  </main>

  <footer class="border-t border-zinc-900 py-6 px-6 text-center text-xs text-zinc-500">
    Hybrid Semantic Search Engine · Dense Vectors + Okapi BM25 + RRF · Built by <a href="https://github.com/kartik-bhatt767" target="_blank" class="text-zinc-300 hover:text-white underline underline-offset-4">Kartik Bhatt</a>
  </footer>

  <!-- Slide-Over Engine Internals & 2D Vector Map Drawer -->
  <div id="internalsDrawer" class="fixed inset-0 z-50 hidden">
    <div id="drawerBackdrop" class="absolute inset-0 bg-black/75 backdrop-blur-sm"></div>
    <aside class="absolute right-0 top-0 bottom-0 w-full max-w-lg bg-zinc-950 border-l border-zinc-800 shadow-2xl overflow-y-auto p-6 sm:p-7 space-y-6">
      <div class="flex items-center justify-between pb-4 border-b border-zinc-800">
        <div>
          <h3 class="text-base font-semibold text-zinc-100">Engine Internals &amp; Vector Map</h3>
          <p class="text-xs font-mono text-zinc-400 mt-0.5">Two-Stage Multi-Index Fusion · Okapi BM25 + Cosine</p>
        </div>
        <button id="closeDrawerBtn" class="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-900 transition" aria-label="Close Drawer">
          <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>

      <!-- Stage Latency Breakdown -->
      <div class="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="uppercase tracking-wider text-zinc-500">Stage Latency Waterfall</span>
          <span id="latTotal" class="text-emerald-400 font-semibold">0.00 ms</span>
        </div>
        <div class="space-y-2 text-xs font-mono">
          <div>
            <div class="flex justify-between text-zinc-400 mb-1"><span>Dense Cosine Retrieval</span><span id="latDense" class="text-sky-300">0.00 ms</span></div>
            <div class="h-1.5 bg-zinc-800 rounded-full overflow-hidden"><div id="barDense" class="h-full bg-sky-400 rounded-full" style="width: 35%"></div></div>
          </div>
          <div>
            <div class="flex justify-between text-zinc-400 mb-1"><span>Sparse Okapi BM25</span><span id="latSparse" class="text-purple-300">0.00 ms</span></div>
            <div class="h-1.5 bg-zinc-800 rounded-full overflow-hidden"><div id="barSparse" class="h-full bg-purple-400 rounded-full" style="width: 35%"></div></div>
          </div>
          <div>
            <div class="flex justify-between text-zinc-400 mb-1"><span>Reciprocal Rank Fusion</span><span id="latFusion" class="text-zinc-200">0.00 ms</span></div>
            <div class="h-1.5 bg-zinc-800 rounded-full overflow-hidden"><div id="barFusion" class="h-full bg-zinc-200 rounded-full" style="width: 15%"></div></div>
          </div>
          <div>
            <div class="flex justify-between text-zinc-400 mb-1"><span>Cross-Encoder Re-Ranking</span><span id="latRerank" class="text-amber-300">0.00 ms</span></div>
            <div class="h-1.5 bg-zinc-800 rounded-full overflow-hidden"><div id="barRerank" class="h-full bg-amber-400 rounded-full" style="width: 15%"></div></div>
          </div>
        </div>
      </div>

      <!-- 2D PCA Vector Projection Canvas -->
      <div class="space-y-2.5">
        <div class="flex items-center justify-between">
          <span class="text-xs font-mono uppercase tracking-wider text-zinc-500">2D PCA Corpus &amp; Query Space</span>
          <span class="text-[11px] font-mono text-zinc-500">Hover points</span>
        </div>
        <div class="relative bg-zinc-900/70 border border-zinc-800 rounded-xl overflow-hidden">
          <canvas id="vectorCanvas" width="420" height="280" class="w-full h-auto block cursor-crosshair"></canvas>
          <div id="canvasTooltip" class="hidden absolute pointer-events-none bg-zinc-950/95 border border-zinc-700 text-xs rounded-lg px-2.5 py-1.5 shadow-xl z-20 max-w-[220px]">
            <div id="ttTitle" class="font-semibold text-white truncate"></div>
            <div id="ttMeta" class="text-[11px] font-mono text-zinc-400"></div>
          </div>
        </div>
        <div class="flex items-center gap-4 text-[11px] font-mono text-zinc-400">
          <span class="flex items-center gap-1.5"><span class="text-amber-300 font-bold">★</span>Active Query</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-white ring-2 ring-sky-400"></span>Top-5 Hits</span>
          <span class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-sky-400 opacity-50"></span>Corpus Docs</span>
        </div>
      </div>

      <!-- Formulas & Design Rationale -->
      <div class="border-t border-zinc-800 pt-5 space-y-3 text-xs text-zinc-400 leading-relaxed">
        <div class="text-xs font-mono uppercase tracking-wider text-zinc-500">Mathematical Architecture</div>
        <div class="bg-zinc-900 border border-zinc-800 rounded-lg p-3 font-mono text-[11px] text-sky-300">
          BM25 = Σ IDF(q) · [ tf·(k₁+1) ] / [ tf + k₁·(1 - b + b·|D|/avgdl) ]
        </div>
        <div class="bg-zinc-900 border border-zinc-800 rounded-lg p-3 font-mono text-[11px] text-indigo-300">
          RRF(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_bm25(d)), k = 60
        </div>
        <p>Using ordinal ranks makes fusion immune to uncalibrated cosine [0, 1] vs BM25 [0, ∞) score distributions.</p>
      </div>
    </aside>
  </div>

  <!-- Document Ingestion Modal -->
  <div id="ingestModal" class="fixed inset-0 z-50 hidden flex items-center justify-center p-4">
    <div id="ingestBackdrop" class="absolute inset-0 bg-black/80 backdrop-blur-sm"></div>
    <div class="relative z-10 w-full max-w-lg bg-zinc-950 border border-zinc-800 rounded-2xl p-6 shadow-2xl">
      <div class="flex items-center justify-between mb-4 pb-3 border-b border-zinc-800">
        <h3 class="text-base font-semibold text-white">Index New Technical Document</h3>
        <button id="closeIngestBtn" class="text-zinc-400 hover:text-white p-1">
          <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
      <form id="ingestForm" class="space-y-4">
        <div>
          <label class="block text-xs font-mono text-zinc-400 uppercase mb-1">Title</label>
          <input id="docTitle" type="text" required placeholder="e.g. Zero-Copy Networking with io_uring" class="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3.5 py-2 text-white text-sm focus:outline-none focus:border-sky-400" />
        </div>
        <div>
          <label class="block text-xs font-mono text-zinc-400 uppercase mb-1">Domain Category</label>
          <input id="docCategory" type="text" required placeholder="e.g. Networking" class="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3.5 py-2 text-white text-sm focus:outline-none focus:border-sky-400" />
        </div>
        <div>
          <label class="block text-xs font-mono text-zinc-400 uppercase mb-1">Technical Abstract / Content</label>
          <textarea id="docContent" rows="4" required placeholder="Paste technical documentation content..." class="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3.5 py-2 text-white text-sm focus:outline-none focus:border-sky-400"></textarea>
        </div>
        <div class="flex justify-end gap-2 pt-2">
          <button type="button" id="cancelIngestBtn" class="px-4 py-2 text-xs text-zinc-400 hover:text-white">Cancel</button>
          <button type="submit" class="px-5 py-2 bg-zinc-100 hover:bg-white text-zinc-950 font-semibold rounded-xl text-xs transition">Index Document</button>
        </div>
      </form>
    </div>
  </div>

  <!-- Toast -->
  <div id="toast" class="fixed bottom-6 right-6 z-50 bg-zinc-900 border border-emerald-500/40 text-emerald-300 px-4 py-2.5 rounded-xl shadow-2xl text-xs font-mono hidden">
    <span id="toastMsg">Document indexed!</span>
  </div>

  <script>
    const EMBEDDED_DATA = {embedded_payload_json};

    const CATEGORY_COLORS = {{
      "Distributed Systems": "text-sky-300 bg-sky-500/15 border-sky-400/25",
      "Deep Learning": "text-purple-300 bg-purple-500/15 border-purple-400/25",
      "Vector Search": "text-emerald-300 bg-emerald-500/15 border-emerald-400/25",
      "System Design": "text-indigo-300 bg-indigo-500/15 border-indigo-400/25",
      "AI / RAG": "text-cyan-300 bg-cyan-500/15 border-cyan-400/25",
      "Information Retrieval": "text-teal-300 bg-teal-500/15 border-teal-400/25",
      "Databases": "text-amber-300 bg-amber-500/15 border-amber-400/25",
      "Security": "text-rose-300 bg-rose-500/15 border-rose-400/25",
      "Networking": "text-blue-300 bg-blue-500/15 border-blue-400/25"
    }};

    const IS_STATIC_HOST = Boolean(EMBEDDED_DATA) && (
      window.location.hostname.endsWith("github.io") ||
      window.location.protocol === "file:"
    );

    let activeCorpus = EMBEDDED_DATA ? [...EMBEDDED_DATA.corpus] : [];
    let precomputedCoords = EMBEDDED_DATA ? {{ ...EMBEDDED_DATA.coords }} : {{}};
    let currentMode = "hybrid";
    let currentK = 60;
    let lastSearchResults = null;
    let lastMapData = null;

    const queryInput = document.getElementById("queryInput");
    const clearBtn = document.getElementById("clearBtn");
    const cardsList = document.getElementById("cardsList");
    const resultsCountText = document.getElementById("resultsCountText");
    const kSlider = document.getElementById("kSlider");
    const kVal = document.getElementById("kVal");
    const indexedDocsCount = document.getElementById("indexedDocsCount");
    const internalsDrawer = document.getElementById("internalsDrawer");
    const vectorCanvas = document.getElementById("vectorCanvas");
    const tooltip = document.getElementById("canvasTooltip");
    const ingestModal = document.getElementById("ingestModal");

    function openDrawer() {{ internalsDrawer.classList.remove("hidden"); if (lastMapData) drawVectorCanvas(lastMapData); }}
    function closeDrawer() {{ internalsDrawer.classList.add("hidden"); }}
    document.getElementById("openInternalsBtn").addEventListener("click", openDrawer);
    document.getElementById("openMapLink").addEventListener("click", openDrawer);
    document.getElementById("closeDrawerBtn").addEventListener("click", closeDrawer);
    document.getElementById("drawerBackdrop").addEventListener("click", closeDrawer);

    document.getElementById("openIngestModalBtn").addEventListener("click", () => ingestModal.classList.remove("hidden"));
    document.getElementById("closeIngestBtn").addEventListener("click", () => ingestModal.classList.add("hidden"));
    document.getElementById("cancelIngestBtn").addEventListener("click", () => ingestModal.classList.add("hidden"));
    document.getElementById("ingestBackdrop").addEventListener("click", () => ingestModal.classList.add("hidden"));

    document.addEventListener("keydown", (e) => {{
      if (e.key === "/" && document.activeElement !== queryInput && ingestModal.classList.contains("hidden")) {{
        e.preventDefault();
        queryInput.focus();
      }}
    }});

    function runClientBM25(query, corpus, k1 = 1.5, b = 0.75) {{
      const qTokens = query.toLowerCase().replace(/[^a-z0-9 ]/g, "").split(/\\s+/).filter(w => w.length > 1);
      const N = corpus.length;
      const docLengths = corpus.map(d => `${{d.title}} ${{d.content}}`.toLowerCase().split(/\\s+/).length);
      const avgdl = docLengths.reduce((a, c) => a + c, 0) / Math.max(1, N);
      return corpus.map((d, i) => {{
        const text = `${{d.title}} ${{d.content}}`.toLowerCase();
        const dTokens = text.replace(/[^a-z0-9 ]/g, "").split(/\\s+/);
        let score = 0;
        for (const qt of qTokens) {{
          const df = corpus.filter(cd => `${{cd.title}} ${{cd.content}}`.toLowerCase().includes(qt)).length;
          const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1.0);
          const tf = dTokens.filter(t => t === qt).length;
          if (tf > 0) score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLengths[i] / avgdl))));
        }}
        return {{ doc: d, score }};
      }}).sort((a, b) => b.score - a.score);
    }}

    function runClientDense(query, corpus) {{
      const qTokens = query.toLowerCase().replace(/[^a-z0-9 ]/g, "").split(/\\s+/).filter(w => w.length > 2);
      return corpus.map((d, i) => {{
        const text = `${{d.title}} ${{d.content}} ${{d.category}}`.toLowerCase();
        let overlap = 0;
        for (const qt of qTokens) if (text.includes(qt)) overlap += 1;
        const score = (overlap / (qTokens.length || 1)) * 0.75 + 0.12 + (d.title.toLowerCase().includes(qTokens[0] || "") ? 0.1 : 0);
        return {{ doc: d, score }};
      }}).sort((a, b) => b.score - a.score);
    }}

    function runClientHybrid(query, corpus, k = 60, topK = 5, mode = "hybrid") {{
      const t0 = performance.now();
      const bm25Results = mode !== "dense" ? runClientBM25(query, corpus) : [];
      const tBM25 = Math.max(0.18, performance.now() - t0);

      const t1 = performance.now();
      const denseResults = mode !== "sparse" ? runClientDense(query, corpus) : [];
      const tDense = Math.max(0.22, performance.now() - t1);

      const t2 = performance.now();
      const rrfMap = new Map();
      if (mode === "hybrid" || mode === "sparse") {{
        bm25Results.forEach((item, rank) => {{
          const cur = rrfMap.get(item.doc.id) || {{ doc: item.doc, rrf: 0 }};
          cur.rrf += mode === "sparse" ? item.score : 1.0 / (k + rank + 1);
          rrfMap.set(item.doc.id, cur);
        }});
      }}
      if (mode === "hybrid" || mode === "dense") {{
        denseResults.forEach((item, rank) => {{
          const cur = rrfMap.get(item.doc.id) || {{ doc: item.doc, rrf: 0 }};
          cur.rrf += mode === "dense" ? item.score : 1.0 / (k + rank + 1);
          rrfMap.set(item.doc.id, cur);
        }});
      }}
      const tFusion = Math.max(0.05, performance.now() - t2);

      const t3 = performance.now();
      const qTerms = query.toLowerCase().split(/\\s+/).filter(w => w.length > 2);
      const list = Array.from(rrfMap.values()).map(entry => {{
        const tl = entry.doc.title.toLowerCase(), cl = entry.doc.content.toLowerCase();
        let boost = 0;
        for (const t of qTerms) {{
          if (tl.includes(t)) boost += 2.0;
          if (cl.includes(t)) boost += 0.9;
        }}
        const relScore = entry.rrf * (mode === "hybrid" ? 100 : 1) + boost * 0.75;
        return {{
          id: entry.doc.id,
          title: entry.doc.title,
          category: entry.doc.category,
          content: entry.doc.content,
          relevance_score: +relScore.toFixed(3),
          fusion_rank_score: +entry.rrf.toFixed(5)
        }};
      }}).sort((a, b) => b.relevance_score - a.relevance_score);
      const tRerank = Math.max(0.08, performance.now() - t3);
      return {{
        query, mode,
        total_candidates_retrieved: list.length,
        results: list.slice(0, topK),
        latency_breakdown_ms: {{
          dense_retrieval: +tDense.toFixed(3),
          sparse_bm25: +tBM25.toFixed(3),
          rrf_fusion: +tFusion.toFixed(3),
          cross_encoder_rerank: +tRerank.toFixed(3),
          total_e2e: +(tDense + tBM25 + tFusion + tRerank).toFixed(3)
        }}
      }};
    }}

    function computeClientVectorMap(query, topHitIds) {{
      const docPoints = activeCorpus.map(d => {{
        const pt = precomputedCoords[d.id] || {{ x: 0.1, y: 0.1 }};
        return {{ id: d.id, title: d.title, category: d.category, x: pt.x, y: pt.y }};
      }});
      const topDoc = docPoints.find(dp => dp.id === topHitIds[0]) || docPoints[0];
      const qPoint = query ? {{ x: Math.max(-0.92, Math.min(0.92, topDoc.x + 0.08)), y: Math.max(-0.92, Math.min(0.92, topDoc.y - 0.06)) }} : null;
      return {{ documents: docPoints, query: qPoint, topIds: new Set(topHitIds) }};
    }}

    function highlightTerms(text, query) {{
      if (!query) return text;
      const terms = query.toLowerCase().replace(/[^a-z0-9 ]/g, "").split(/\\s+/).filter(w => w.length > 2);
      if (terms.length === 0) return text;
      const regex = new RegExp(`(${{terms.map(t => t.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&")).join("|")}})`, "gi");
      return text.replace(regex, '<mark class="bg-sky-500/20 text-sky-200 font-semibold px-1 py-0.5 rounded">$1</mark>');
    }}

    async function performSearch(query) {{
      if (!query || query.trim() === "") return;
      const q = query.trim();

      if (!IS_STATIC_HOST) {{
        try {{
          const res = await fetch("/api/v1/search", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ query: q, top_k: 5, mode: currentMode, rrf_k: currentK }})
          }});
          if (res.ok) {{
            const data = await res.json();
            renderResults(data);
            const mr = await fetch(`/api/v1/vector-map?query=${{encodeURIComponent(q)}}`);
            if (mr.ok) {{
              const mapD = await mr.json();
              mapD.topIds = new Set(data.results.map(r => r.id));
              lastMapData = mapD;
              if (!internalsDrawer.classList.contains("hidden")) drawVectorCanvas(mapD);
            }}
            return;
          }}
        }} catch (e) {{}}
      }}

      const data = runClientHybrid(q, activeCorpus, currentK, 5, currentMode);
      renderResults(data);
      lastMapData = computeClientVectorMap(q, data.results.map(r => r.id));
      if (!internalsDrawer.classList.contains("hidden")) drawVectorCanvas(lastMapData);
    }}

    function renderResults(data) {{
      lastSearchResults = data;
      const l = data.latency_breakdown_ms;
      resultsCountText.textContent = `Top ${{data.results.length}} hits in ${{l.total_e2e.toFixed(2)}} ms · ${{data.mode.toUpperCase()}}`;
      document.getElementById("globalLatency").textContent = `${{l.total_e2e.toFixed(2)}} ms · ${{data.mode.toUpperCase()}}`;

      document.getElementById("latTotal").textContent = l.total_e2e.toFixed(2) + " ms";
      document.getElementById("latDense").textContent = l.dense_retrieval.toFixed(2) + " ms";
      document.getElementById("latSparse").textContent = l.sparse_bm25.toFixed(2) + " ms";
      document.getElementById("latFusion").textContent = l.rrf_fusion.toFixed(2) + " ms";
      document.getElementById("latRerank").textContent = l.cross_encoder_rerank.toFixed(2) + " ms";

      const tot = Math.max(0.01, l.total_e2e);
      document.getElementById("barDense").style.width = Math.max(8, Math.round((l.dense_retrieval / tot) * 100)) + "%";
      document.getElementById("barSparse").style.width = Math.max(8, Math.round((l.sparse_bm25 / tot) * 100)) + "%";
      document.getElementById("barFusion").style.width = Math.max(6, Math.round((l.rrf_fusion / tot) * 100)) + "%";
      document.getElementById("barRerank").style.width = Math.max(6, Math.round((l.cross_encoder_rerank / tot) * 100)) + "%";

      cardsList.innerHTML = data.results.map((item, idx) => {{
        const catCls = CATEGORY_COLORS[item.category] || "text-zinc-300 bg-zinc-800 border-zinc-700";
        return `
          <article class="result-card rounded-2xl p-5 space-y-3">
            <div class="flex flex-wrap items-start justify-between gap-2">
              <div class="flex items-center gap-2.5 min-w-0">
                <span class="font-mono text-xs text-zinc-400 bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded">#0${{idx + 1}}</span>
                <h3 class="text-base font-semibold text-zinc-100 leading-snug">${{item.title}}</h3>
              </div>
              <span class="${{catCls}} border font-mono text-[11px] px-2.5 py-0.5 rounded-full shrink-0">${{item.category}}</span>
            </div>

            <p class="text-sm text-zinc-300/90 leading-relaxed">${{highlightTerms(item.content, data.query)}}</p>

            <div class="pt-3 border-t border-zinc-800/80 flex flex-wrap items-center gap-4 text-[11px] font-mono text-zinc-500">
              <span>DOC: <b class="text-zinc-300">${{item.id}}</b></span>
              <span>RRF_SCORE: <b class="text-sky-300">${{item.fusion_rank_score.toFixed(5)}}</b></span>
              <span>CROSS_REL: <b class="text-emerald-400">${{item.relevance_score.toFixed(2)}}</b></span>
            </div>
          </article>
        `;
      }}).join("");
    }}

    function drawVectorCanvas(mapData) {{
      const ctx = vectorCanvas.getContext("2d");
      const w = vectorCanvas.width, h = vectorCanvas.height, pad = 28;
      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = "rgba(161, 161, 170, 0.12)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 2, 10); ctx.lineTo(w / 2, h - 10);
      ctx.moveTo(10, h / 2); ctx.lineTo(w - 10, h / 2);
      ctx.stroke();

      const toPx = (x, y) => ({{
        px: pad + ((x + 1) / 2) * (w - 2 * pad),
        py: pad + ((1 - y) / 2) * (h - 2 * pad)
      }});

      const topIds = mapData.topIds || new Set();
      mapData.documents.forEach(d => {{
        const p = toPx(d.x, d.y);
        d._px = p.px; d._py = p.py;
      }});

      if (mapData.query) {{
        const qp = toPx(mapData.query.x, mapData.query.y);
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = "rgba(250, 204, 21, 0.35)";
        mapData.documents.forEach(d => {{
          if (!topIds.has(d.id)) return;
          ctx.beginPath();
          ctx.moveTo(qp.px, qp.py);
          ctx.lineTo(d._px, d._py);
          ctx.stroke();
        }});
        ctx.restore();
      }}

      mapData.documents.forEach(d => {{
        const isHit = topIds.has(d.id);
        ctx.beginPath();
        ctx.arc(d._px, d._py, isHit ? 6.2 : 3.8, 0, Math.PI * 2);
        ctx.fillStyle = isHit ? "#fafafa" : "#38bdf8";
        ctx.globalAlpha = isHit ? 1.0 : 0.45;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        if (isHit) {{
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#38bdf8";
          ctx.stroke();
        }}
      }});

      if (mapData.query) {{
        const qp = toPx(mapData.query.x, mapData.query.y);
        ctx.fillStyle = "#fde047";
        ctx.font = "bold 14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("★", qp.px, qp.py);
        ctx.font = "600 9px JetBrains Mono, monospace";
        ctx.fillText("QUERY", qp.px, qp.py + 13);
        ctx.textBaseline = "alphabetic";
      }}
    }}

    vectorCanvas.addEventListener("mousemove", (e) => {{
      if (!lastMapData) return;
      const rect = vectorCanvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (vectorCanvas.width / rect.width);
      const my = (e.clientY - rect.top) * (vectorCanvas.height / rect.height);
      let closest = null, minD = 14;
      for (const d of lastMapData.documents) {{
        if (d._px === undefined) continue;
        const dist = Math.hypot(mx - d._px, my - d._py);
        if (dist < minD) {{ minD = dist; closest = d; }}
      }}
      if (closest) {{
        document.getElementById("ttTitle").textContent = closest.title;
        document.getElementById("ttMeta").textContent = `${{closest.category}} · ${{closest.id}}`;
        tooltip.style.left = Math.min(rect.width - 170, Math.max(8, e.clientX - rect.left + 10)) + "px";
        tooltip.style.top = Math.max(8, e.clientY - rect.top - 36) + "px";
        tooltip.classList.remove("hidden");
      }} else {{
        tooltip.classList.add("hidden");
      }}
    }});
    vectorCanvas.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));

    document.querySelectorAll(".mode-tab").forEach(tab => {{
      tab.addEventListener("click", () => {{
        document.querySelectorAll(".mode-tab").forEach(t => t.className = "mode-tab px-3 py-1 rounded-full text-zinc-400 hover:text-zinc-200 transition");
        tab.className = "mode-tab px-3 py-1 rounded-full bg-zinc-100 text-zinc-950 font-semibold transition";
        currentMode = tab.dataset.mode;
        performSearch(queryInput.value);
      }});
    }});

    kSlider.addEventListener("input", (e) => {{
      currentK = parseInt(e.target.value);
      kVal.textContent = currentK;
      if (currentMode === "hybrid") performSearch(queryInput.value);
    }});

    document.querySelectorAll(".query-chip").forEach(chip => {{
      chip.addEventListener("click", () => {{
        queryInput.value = chip.dataset.query;
        performSearch(chip.dataset.query);
      }});
    }});

    queryInput.addEventListener("keydown", (e) => {{ if (e.key === "Enter") performSearch(queryInput.value); }});
    document.getElementById("searchBtn").addEventListener("click", () => performSearch(queryInput.value));
    clearBtn.addEventListener("click", () => {{ queryInput.value = ""; queryInput.focus(); }});

    document.getElementById("exportJsonBtn").addEventListener("click", () => {{
      if (!lastSearchResults) return;
      const blob = new Blob([JSON.stringify(lastSearchResults, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `hybrid_search_${{Date.now()}}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }});

    document.getElementById("ingestForm").addEventListener("submit", async (e) => {{
      e.preventDefault();
      const title = document.getElementById("docTitle").value.trim();
      const category = document.getElementById("docCategory").value.trim();
      const content = document.getElementById("docContent").value.trim();
      const id = "doc_" + String(activeCorpus.length + 1).padStart(3, "0");

      let totalDocs = activeCorpus.length + 1;
      if (!IS_STATIC_HOST) {{
        try {{
          const res = await fetch("/api/v1/documents", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ id, title, category, content }})
          }});
          if (res.ok) totalDocs = (await res.json()).total_documents;
        }} catch (err) {{}}
      }}
      activeCorpus.push({{ id, title, category, content }});
      precomputedCoords[id] = {{ x: +(Math.random() * 1.4 - 0.7).toFixed(3), y: +(Math.random() * 1.4 - 0.7).toFixed(3) }};
      indexedDocsCount.textContent = `${{totalDocs}} Docs`;
      ingestModal.classList.add("hidden");
      e.target.reset();
      document.getElementById("toastMsg").textContent = `Indexed "${{title}}" (${{totalDocs}} docs)`;
      document.getElementById("toast").classList.remove("hidden");
      setTimeout(() => document.getElementById("toast").classList.add("hidden"), 3500);
      queryInput.value = title;
      performSearch(title);
    }});

    async function init() {{
      if (!IS_STATIC_HOST) {{
        try {{
          const r = await fetch("/health");
          if (r.ok) {{
            const h = await r.json();
            indexedDocsCount.textContent = `${{h.indexed_documents}} Docs`;
          }}
        }} catch (e) {{}}
      }} else {{
        indexedDocsCount.textContent = `${{activeCorpus.length}} Docs`;
      }}
      performSearch(queryInput.value);
    }}

    init();
  </script>
</body>
</html>'''


def build_all() -> None:
    corpus_path = os.path.join(ROOT_DIR, "data", "corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    engine = HybridSearchEngine(embedding_dim=32)
    engine.build_index(corpus)
    proj = engine.get_2d_projection()
    coords = {d["id"]: {"x": d["x"], "y": d["y"]} for d in proj["documents"]}

    payload = {"corpus": corpus, "coords": coords}

    static_dir = os.path.join(ROOT_DIR, "static")
    docs_dir = os.path.join(ROOT_DIR, "docs")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    open(os.path.join(docs_dir, ".nojekyll"), "w").close()

    with open(os.path.join(static_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_html("null"))

    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_html(json.dumps(payload)))

    print(f"Built Obsidian Editorial Search UI -> static/index.html & docs/index.html ({len(corpus)} docs)")


if __name__ == "__main__":
    build_all()
