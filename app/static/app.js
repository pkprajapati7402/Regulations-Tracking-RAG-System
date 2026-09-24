/* Regulation-Tracking RAG — chat UI (vanilla JS, no build step). */
const $ = (sel) => document.querySelector(sel);
const api = async (path, opts = {}) => {
  const res = await fetch(path, opts);
  const isJson = (res.headers.get("content-type") || "").includes("json");
  const body = isJson ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.detail || res.statusText);
  return body;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { sessionId: null, busy: false, lastCitations: [] };

/* ------------------------------ rendering ------------------------------ */
function renderAnswer(text) {
  // escape, then linkify [n], 【n】 or 【n†...】 citations and render "- " bullets
  let html = esc(text).replace(/(?:\[|\u3010)(\d{1,2})(?:\u2020[^\u3011]+)?(?:\]|\u3011)/g, '<span class="cite-ref" data-n="$1">$1</span>');
  const lines = html.split("\n");
  let out = "", inList = false;
  for (const line of lines) {
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) { out += "<ul>"; inList = true; }
      out += `<li>${line.replace(/^\s*[-*]\s+/, "")}</li>`;
    } else {
      if (inList) { out += "</ul>"; inList = false; }
      out += line + "\n";
    }
  }
  if (inList) out += "</ul>";
  return out.trim();
}

function metricClass(rate) { return rate >= 0.8 ? "good" : rate >= 0.5 ? "warn" : "bad"; }

function sourcesHtml(citations) {
  if (!citations.length) return "";
  const items = citations.map((c) => {
    const comps = Object.entries(c.components || {})
      .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(4) : v}`).join("  ");
    const status = c.status === "active"
      ? '<span class="badge active">active</span>'
      : '<span class="badge deprecated">superseded</span>';
    const type = c.chunk_type === "table" ? '<span class="badge table">table</span>' : "";
    return `<div class="source" id="src-${c.n}">
      <div class="s-head">
        <span class="s-n">[${c.n}]</span>
        <span class="s-doc">${esc(c.doc_number)}</span>
        ${status}${type}
        ${c.published_date ? `<span class="badge">${esc(c.published_date)}</span>` : ""}
      </div>
      <div class="s-title">${esc(c.doc_title)}${c.section_ref ? ` — <i class="muted">${esc(c.section_ref)}</i>` : ""}</div>
      <div class="s-body">${esc(c.text)}</div>
      <div class="s-scores">score=${Number(c.score).toFixed(4)} ${esc(comps)}</div>
    </div>`;
  }).join("");
  return `<details class="sources"><summary>📎 ${citations.length} cited source${citations.length > 1 ? "s" : ""} — click to inspect</summary>${items}</details>`;
}

function addMessage(role, html) {
  const welcome = $(".welcome");
  if (welcome) welcome.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="who">${role === "user" ? "You" : "Assistant"}</div><div class="bubble">${html}</div>`;
  $("#messages").appendChild(el);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return el;
}

function renderAssistant(el, data) {
  const f = data.faithfulness || {};
  const r = data.retrieval || {};
  const u = data.usage || {};
  const rate = f.checked ? f.rate : 1;
  const metrics = `
    <div class="meta-row">
      <span class="metric ${metricClass(rate)}">citation faithfulness ${(rate * 100).toFixed(0)}% (${f.supported || 0}/${f.checked || 0})</span>
      <span class="metric">${esc(r.mode || "")} · ${r.candidates || 0} passages · ${Math.round(r.latency_ms || 0)} ms</span>
      <span class="metric">${esc(u.provider || "")}${u.total_tokens ? ` · ${u.total_tokens} tok` : ""}</span>
      ${r.include_deprecated ? '<span class="metric warn">audit mode: superseded text included</span>' : ""}
    </div>`;
  const notices = (data.notices || []).map((n) => `<div class="notice">⚠ ${esc(n)}</div>`).join("");
  el.querySelector(".bubble").innerHTML =
    renderAnswer(data.answer) + metrics + notices + sourcesHtml(data.citations || []);
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

/* -------------------------------- chat -------------------------------- */
async function send(text) {
  if (!text.trim() || state.busy) return;
  state.busy = true;
  $("#send").disabled = true;
  addMessage("user", esc(text));
  const pending = addMessage("assistant", '<span class="typing"><i></i><i></i><i></i></span>');

    const provider = $("#llm-provider") ? $("#llm-provider").value : undefined;
    const model = $("#llm-model") && $("#model-field")?.style.display !== "none" ? $("#llm-model").value : undefined;
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: state.sessionId,
        mode: $("#mode").value,
        top_k: Number($("#topk").value),
        include_deprecated: $("#include-deprecated").checked,
        provider: provider,
        model: model,
      }),
    });
    state.sessionId = data.session_id;
    state.lastCitations = data.citations;
    renderAssistant(pending, data);
    $("#chat-title").textContent = text.length > 70 ? text.slice(0, 70) + "…" : text;
  } catch (err) {
    pending.querySelector(".bubble").innerHTML =
      `<span style="color:var(--bad)">Request failed: ${esc(err.message)}</span>`;
  } finally {
    state.busy = false;
    $("#send").disabled = false;
    $("#input").focus();
  }
}

/* ------------------------------ sidebar ------------------------------- */
const providerModels = {
  groq: [
    { id: "openai/gpt-oss-120b", name: "openai/gpt-oss-120b (Deep Reasoning)" },
    { id: "qwen/qwen3.8-27b", name: "qwen/qwen3.8-27b (Fast & Accurate)" },
    { id: "openai/gpt-oss-20b", name: "openai/gpt-oss-20b (Lightweight)" },
  ],
  gemini: [
    { id: "gemini-3.6-flash", name: "gemini-3.6-flash (Complex tasks)" },
  ],
  extractive: []
};

function updateModels(prov) {
  const modelField = $("#model-field");
  const modelSelect = $("#llm-model");
  if (!modelSelect || !modelField) return;
  const list = providerModels[prov] || [];
  if (!list.length) {
    modelField.style.display = "none";
    return;
  }
  modelField.style.display = "";
  modelSelect.innerHTML = list.map(m => `<option value="${m.id}">${m.name}</option>`).join("");
}

async function loadHealth() {
  try {
    const h = await api("/api/health");
    $("#health").textContent = `${h.llm.split(":")[0]} · ${h.embedding.split(":")[0]} · ${h.reranker}`;
    $("#health").title = `embedding=${h.embedding}\nreranker=${h.reranker}\nllm=${h.llm}\nchunker=${h.chunker}`;
    if (h.llm && $("#llm-provider")) {
      const parts = h.llm.split(":");
      const prov = parts[0];
      const mod = parts.slice(1).join(":");
      if (["groq", "gemini", "extractive"].includes(prov)) {
        $("#llm-provider").value = prov;
        updateModels(prov);
        if (mod && $("#llm-model")) $("#llm-model").value = mod;
      }
    }
  } catch { $("#health").textContent = "offline"; }
}

async function loadDocs() {
  try {
    const [docs, health] = await Promise.all([api("/api/documents"), api("/api/health")]);
    const s = health.index || {};
    $("#index-stats").innerHTML = `
      <span class="stat"><b>${s.documents ?? 0}</b> docs</span>
      <span class="stat"><b>${s.chunks_active ?? 0}</b> active chunks</span>
      <span class="stat"><b>${s.chunks_deprecated ?? 0}</b> superseded</span>
      <span class="stat"><b>${s.tables_indexed ?? 0}</b> tables</span>`;
    $("#doc-list").innerHTML = docs.length
      ? docs.map((d) => {
          const latest = d.versions[d.versions.length - 1] || {};
          const superseded = d.versions.filter((v) => v.status !== "active").length;
          return `<div class="doc">
            <div class="num">${esc(d.doc_number)}</div>
            <div class="title">${esc(d.title)}</div>
            <span class="badge ${latest.status === "active" ? "active" : "deprecated"}">v${latest.version_number || 1} ${esc(latest.status || "")}</span>
            <span class="badge">${latest.chunk_count || 0} chunks</span>
            ${superseded ? `<span class="badge deprecated">${superseded} old version${superseded > 1 ? "s" : ""}</span>` : ""}
          </div>`;
        }).join("")
      : '<p class="muted small">No documents indexed yet. Run <code>python -m ingestion.reindex --seed</code> or upload one.</p>';
  } catch (err) {
    $("#doc-list").innerHTML = `<p class="muted small">Could not load documents: ${esc(err.message)}</p>`;
  }
}

/* ------------------------------- upload ------------------------------- */
function openUpload(open) {
  const modal = $("#upload-modal");
  if (!modal) return;
  modal.hidden = !open;
  const status = $("#upload-status");
  if (!open) {
    $("#upload-form").reset();
    if (status) {
      status.hidden = true;
      status.textContent = "";
      status.className = "status";
    }
    $("#do-upload").disabled = false;
  } else {
    if (status) {
      status.hidden = true;
      status.textContent = "";
      status.className = "status";
    }
  }
}

async function submitUpload(ev) {
  ev.preventDefault();
  const activeTab = $(".tab.active")?.dataset?.tab || "file";
  const status = $("#upload-status");
  status.hidden = false;
  status.className = "status";
  status.textContent = "Parsing, chunking, embedding and indexing…";
  $("#do-upload").disabled = true;

  try {
    let data;
    if (activeTab === "file") {
      const file = $("#file").files[0];
      if (!file) throw new Error("Choose a file first");
      const fd = new FormData();
      fd.append("file", file);
      if ($("#up-title").value) fd.append("title", $("#up-title").value);
      fd.append("category", $("#up-category").value);
      if ($("#up-chunker").value) fd.append("chunker", $("#up-chunker").value);
      if ($("#up-supersedes").value) fd.append("supersedes", $("#up-supersedes").value);
      if ($("#up-url").value) fd.append("source_url", $("#up-url").value);
      data = await api("/api/documents/upload", { method: "POST", body: fd });
    } else {
      const text = ($("#paste-text").value || "").trim();
      if (!text) throw new Error("Please paste the circular text first");
      data = await api("/api/documents/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          title: $("#up-title").value || null,
          category: $("#up-category").value,
          chunker: $("#up-chunker").value || null,
          supersedes: $("#up-supersedes").value || null,
          source_url: $("#up-url").value || null,
        }),
      });
    }
    const d = data.document;
    status.className = `status ${data.status === "indexed" ? "ok" : ""}`;
    status.textContent =
      `✓ ${data.status.toUpperCase()}: ${d.doc_number} (v${d.version_number}) — ${data.detail}. ` +
      `Index now holds ${data.index.chunks_active} active chunks across ${data.index.documents} documents.`;
    await loadDocs();
    await loadHealth();
    // Auto-close modal after brief delay so user can see confirmation
    setTimeout(() => {
      openUpload(false);
    }, 1500);
  } catch (err) {
    status.className = "status err";
    status.textContent = `Failed: ${err.message}`;
  } finally {
    $("#do-upload").disabled = false;
  }
}

/* -------------------------------- wiring ------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#input").value; $("#input").value = ""; $("#input").style.height = "auto"; send(v); });
  $("#input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#input").addEventListener("input", (e) => {
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 180) + "px";
  });
  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("chip")) send(e.target.textContent);
    if (e.target.classList.contains("cite-ref")) {
      const details = e.target.closest(".bubble").querySelector(".sources");
      if (details) {
        details.open = true;
        const src = details.querySelector(`#src-${e.target.dataset.n}`);
        if (src) {
          src.scrollIntoView({ behavior: "smooth", block: "center" });
          src.classList.add("highlight");
          setTimeout(() => src.classList.remove("highlight"), 1600);
        }
      }
    }
  });
  $("#topk").addEventListener("input", (e) => ($("#topk-val").textContent = e.target.value));
  $("#mode").addEventListener("change", (e) => {
    $("#mode-pill").textContent = e.target.selectedOptions[0].textContent.toLowerCase();
  });
  $("#new-chat").addEventListener("click", () => {
    state.sessionId = null;
    $("#chat-title").textContent = "Chat with the circulars";
    location.reload();
  });
  $("#refresh-docs").addEventListener("click", loadDocs);
  $("#open-upload").addEventListener("click", () => openUpload(true));
  $("#close-upload").addEventListener("click", () => openUpload(false));
  $("#cancel-upload").addEventListener("click", () => openUpload(false));
  $("#upload-modal").addEventListener("click", (e) => { if (e.target.id === "upload-modal") openUpload(false); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#upload-modal").hidden) openUpload(false);
  });
  $("#upload-form").addEventListener("submit", submitUpload);
  document.querySelectorAll(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      document.querySelectorAll(".tab-pane").forEach((p) => (p.hidden = p.dataset.pane !== tab.dataset.tab));
    })
  );

  $("#llm-provider")?.addEventListener("change", (e) => updateModels(e.target.value));

  loadHealth();
  loadDocs();
  $("#input").focus();
});
