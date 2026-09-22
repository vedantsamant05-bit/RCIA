/* ==========================================================================
   RCIA Dashboard — app.js
   Vanilla JS, no build step. Talks to the FastAPI backend at API_BASE.
   ========================================================================== */

const API_BASE = window.RCIA_CONFIG?.API_BASE || window.location.origin;

// ---------------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------------
const railLinks = document.querySelectorAll(".rail-link");
const views = document.querySelectorAll(".view");

railLinks.forEach((link) => {
  link.addEventListener("click", () => {
    railLinks.forEach((l) => l.classList.remove("active"));
    link.classList.add("active");
    const target = link.dataset.view;
    views.forEach((v) => v.classList.toggle("active", v.id === `view-${target}`));

    if (target === "queue") loadQueue();
    if (target === "corpus") loadCorpus();
    if (target === "audit") loadAudit();
  });
});

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
async function checkHealth() {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    dot.className = "dot ok";
    text.textContent = data.llm_enabled ? "Connected · LLM reasoning on" : "Connected · heuristic mode";
  } catch (e) {
    dot.className = "dot err";
    text.textContent = "Backend unreachable — start the API server";
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function esc(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function tagFor(kind, value) {
  return `<span class="tag tag-${kind}-${value}">${esc(value).replace(/_/g, " ")}</span>`;
}

function impactTag(impact) {
  return `<span class="tag tag-${impact}">${esc(impact)}</span>`;
}

// ---------------------------------------------------------------------------
// INGEST view
// ---------------------------------------------------------------------------
const ingestForm = document.getElementById("ingest-form");
const ingestResultEl = document.getElementById("ingest-result");
const ingestSubmitBtn = document.getElementById("ingest-submit");

ingestForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const title = document.getElementById("reg-title").value.trim();
  const source = document.getElementById("reg-source").value.trim();
  const text = document.getElementById("reg-text").value.trim();
  if (!title || !text) return;

  ingestSubmitBtn.disabled = true;
  ingestSubmitBtn.textContent = "Running pipeline…";
  ingestResultEl.innerHTML = `<p class="muted">Segmenting clauses, retrieving candidates, running the agent state machine…</p>`;

  try {
    const res = await fetch(`${API_BASE}/api/regulations/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, source: source || null, text }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    ingestResultEl.innerHTML = `
      <h3>Pipeline complete</h3>
      <div class="stat-line"><span>Regulation ID</span><span>${esc(data.regulation_id)}</span></div>
      <div class="stat-line"><span>Clauses analyzed</span><strong>${data.clauses_extracted}</strong></div>
      <div class="stat-line"><span>Candidates retrieved</span><strong>${data.candidates_retrieved}</strong></div>
      <div class="stat-line"><span>Material impacts</span><strong>${data.material_impacts}</strong></div>
      <div class="stat-line"><span>Irrelevant candidates auto-dismissed</span><strong>${data.irrelevant_dismissed}</strong></div>
      <div class="stat-line"><span>Human-review findings</span><strong>${data.human_review_findings}</strong></div>
      <p class="muted" style="margin-top:16px; font-size:12.5px;">Open the Review queue tab to see flagged clauses, drafted redlines, and reasoning traces.</p>
    `;
    ingestForm.reset();
    loadRegulations();
    refreshQueueCount();
  } catch (err) {
    ingestResultEl.innerHTML = `<p style="color:var(--risk-high)">Pipeline failed: ${esc(err.message)}</p>`;
  } finally {
    ingestSubmitBtn.disabled = false;
    ingestSubmitBtn.textContent = "Run pipeline";
  }
});

document.getElementById("load-sample").addEventListener("click", async () => {
  try {
    const res = await fetch(`${API_BASE}/api/samples`);
    const samples = await res.json();
    const pick = samples[Math.floor(Math.random() * samples.length)];
    document.getElementById("reg-title").value = pick.title;
    document.getElementById("reg-source").value = pick.source || "";
    document.getElementById("reg-text").value = pick.text;
  } catch (e) {
    ingestResultEl.innerHTML = `<p style="color:var(--risk-high)">Could not load sample regulations.</p>`;
  }
});

async function loadRegulations() {
  const el = document.getElementById("regulations-list");
  try {
    const res = await fetch(`${API_BASE}/api/regulations`);
    const regs = await res.json();
    if (!regs.length) {
      el.innerHTML = `<p class="muted pad">No regulations ingested yet.</p>`;
      return;
    }
    el.innerHTML = regs
      .map(
        (r) => `
      <div class="ledger-row" data-reg-id="${esc(r.id)}">
        <div class="ledger-row-main">
          <div class="ledger-row-title">${esc(r.title)}</div>
          <div class="ledger-row-sub">${esc(r.source || "no source noted")} · ${esc(r.id)}</div>
        </div>
        <div class="ledger-row-meta">${fmtDate(r.ingested_at)}</div>
      </div>`
      )
      .join("");

    el.querySelectorAll(".ledger-row").forEach((row) => {
      row.addEventListener("click", () => openRegulationDrawer(row.dataset.regId));
    });
  } catch (e) {
    el.innerHTML = `<p class="muted pad">Could not load regulations — is the backend running?</p>`;
  }
}

async function openRegulationDrawer(regId) {
  const res = await fetch(`${API_BASE}/api/regulations/${regId}/results`);
  const data = await res.json();
  const content = `
    <h2 style="font-family:var(--font-display); font-size:20px; margin:0 0 4px 0;">${esc(data.regulation.title)}</h2>
    <p class="muted" style="font-size:12.5px; margin:0 0 20px 0;">${esc(data.regulation.source || "")}</p>
    <div class="ledger">
      ${data.items
        .map(
          (item) => `
        <div class="ledger-row" style="cursor:default;">
          <div class="ledger-row-main">
            <div class="ledger-row-title">${esc(item.doc_title)} [${esc(item.section)}]</div>
            <div class="ledger-row-sub">${impactTag(item.impact_type)} &nbsp; conf ${item.confidence.toFixed(2)}</div>
          </div>
          <div class="ledger-row-meta">${tagFor("status", item.status)}</div>
        </div>`
        )
        .join("")}
    </div>
  `;
  showDrawer(content);
}

// ---------------------------------------------------------------------------
// REVIEW QUEUE view
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// REVIEW QUEUE view
// ---------------------------------------------------------------------------
let currentStatusFilter = "pending_material";

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    currentStatusFilter = chip.dataset.status;
    loadQueue();
  });
});

async function loadQueue() {
  const el = document.getElementById("queue-list");
  el.innerHTML = `<p class="muted pad">Loading review queue…</p>`;
  try {
    // Fetch all items to populate summary stats
    const allRes = await fetch(`${API_BASE}/api/review`);
    const allItems = await allRes.json();

    const uniqueClauses = new Set(allItems.map((i) => i.regulation_clause_id)).size;
    const totalCandidates = allItems.length;
    const materialImpacts = allItems.filter((i) => i.impact_type !== "no-op").length;
    const autoDismissed = allItems.filter((i) => i.routing === "auto_dismiss").length;

    const elClauses = document.getElementById("stat-clauses");
    const elCandidates = document.getElementById("stat-candidates");
    const elImpacts = document.getElementById("stat-impacts");
    const elDismissed = document.getElementById("stat-dismissed");

    if (elClauses) elClauses.textContent = uniqueClauses;
    if (elCandidates) elCandidates.textContent = totalCandidates;
    if (elImpacts) elImpacts.textContent = materialImpacts;
    if (elDismissed) elDismissed.textContent = autoDismissed;

    const url = currentStatusFilter
      ? `${API_BASE}/api/review?status=${encodeURIComponent(currentStatusFilter)}`
      : `${API_BASE}/api/review`;
    const res = await fetch(url);
    const items = await res.json();

    if (!items.length) {
      el.innerHTML = `<p class="muted pad">No items in this view.</p>`;
      return;
    }

    el.innerHTML = items.map((item) => queueItemHtml(item)).join("");

    el.querySelectorAll(".queue-item-head").forEach((head) => {
      head.addEventListener("click", () => {
        head.closest(".queue-item").classList.toggle("open");
      });
    });

    el.querySelectorAll("[data-action]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const id = btn.dataset.id;
        const action = btn.dataset.action;
        await fetch(`${API_BASE}/api/review/${id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        loadQueue();
        refreshQueueCount();
      });
    });
  } catch (e) {
    el.innerHTML = `<p class="muted pad">Could not load review queue — is the backend running?</p>`;
  }
}

function formatRedlineHtml(redlineText) {
  if (!redlineText) return `<p class="muted">No redline drafted.</p>`;
  if (redlineText.includes("CURRENT:") && redlineText.includes("PROPOSED:")) {
    const parts = redlineText.split("PROPOSED:");
    const currentText = parts[0].replace("CURRENT:", "").trim();
    const proposedText = parts[1].trim();
    return `
      <div class="redline-split">
        <div class="redline-box redline-current">
          <div class="redline-label">CURRENT POLICY WORDING</div>
          <div class="redline-text">${esc(currentText)}</div>
        </div>
        <div class="redline-box redline-proposed">
          <div class="redline-label">PROPOSED REDLINE AMENDMENT</div>
          <div class="redline-text">${esc(proposedText)}</div>
        </div>
      </div>
    `;
  }
  return `<p>${esc(redlineText)}</p>`;
}

function formatBusinessActionHtml(actionText) {
  if (!actionText || actionText.includes("No business action required")) return "";
  const lines = actionText.split("\n").filter((l) => l.trim().length > 0);
  const formattedLines = lines
    .map((line) => {
      const l = esc(line.trim());
      if (l.startsWith("ACTION REQUIRED:")) {
        return `<div class="action-highlight">${l.replace("ACTION REQUIRED:", "<strong>⚡ PRIMARY ACTION:</strong>")}</div>`;
      }
      if (l.startsWith("TARGET POLICY:")) {
        return `<div class="action-target">🎯 ${l}</div>`;
      }
      if (l.startsWith("DEADLINE:")) {
        return `<div class="action-deadline">📅 ${l}</div>`;
      }
      if (l.startsWith("STEPS:")) {
        return `<div class="action-steps-title">Implementation Steps:</div>`;
      }
      return `<div class="action-step-item">${l}</div>`;
    })
    .join("");

  return `
    <div class="business-action-box">
      <div class="business-action-header">⚡ BUSINESS ACTION REQUIRED</div>
      <div class="business-action-body">${formattedLines}</div>
    </div>
  `;
}

function queueItemHtml(item) {
  const trace = item.reasoning_trace
    .map(
      (step, i) => `
      <li class="trace-step">
        <span class="trace-num">${i + 1}</span>
        <span class="trace-detail">${esc(step.detail)}</span>
      </li>`
    )
    .join("");

  const needsAction = item.status === "pending";

  return `
    <div class="queue-item">
      <div class="queue-item-head">
        ${impactTag(item.impact_type)}
        <div class="queue-item-titles">
          <div class="queue-item-reg">${esc(item.regulation_title)}</div>
          <div class="queue-item-clause">${esc(item.doc_title)} — Section ${esc(item.section)}</div>
        </div>
        ${tagFor("risk", item.risk)}
        ${tagFor("status", item.status)}
        <div class="queue-item-conf">${item.confidence.toFixed(2)} / 1.00 <span class="conf-mode">(Heuristic)</span></div>
      </div>
      <div class="queue-item-body">
        <div class="compare-cols">
          <div class="compare-col">
            <h4>New regulatory clause</h4>
            <p>${esc(item.regulation_clause_text)}</p>
          </div>
          <div class="compare-col">
            <h4>Existing internal clause</h4>
            <p>${esc(item.internal_clause_text)}</p>
          </div>
        </div>

        ${formatBusinessActionHtml(item.business_action)}

        <div class="draft-block">
          <h4>Drafted redline ${item.citation_verified ? `<span class="tag" style="color:var(--verified); border-color:var(--verified); margin-left:8px;">citation verified</span>` : `<span class="tag" style="color:var(--unverified); border-color:var(--unverified); margin-left:8px;">citation unverified</span>`}</h4>
          ${formatRedlineHtml(item.draft_redline)}
        </div>

        <div class="trace">
          <h4>Reasoning trace</h4>
          <ul class="trace-steps">${trace}</ul>
        </div>

        ${
          needsAction
            ? `<div class="actions-row">
                 <button class="btn btn-approve" data-action="approve" data-id="${esc(item.id)}">Approve redline</button>
                 <button class="btn btn-reject" data-action="reject" data-id="${esc(item.id)}">Reject</button>
                 ${item.routing === "escalate" ? `<button class="btn btn-ghost" data-action="escalate_confirmed" data-id="${esc(item.id)}">Confirm escalation reviewed</button>` : ""}
               </div>`
            : `<div class="reviewer-note">Reviewer action: ${esc(item.reviewer_action || item.status)}${item.reviewed_at ? " · " + fmtDate(item.reviewed_at) : ""}</div>`
        }
      </div>
    </div>
  `;
}

async function refreshQueueCount() {
  try {
    const res = await fetch(`${API_BASE}/api/review?status=pending`);
    const items = await res.json();
    document.getElementById("count-queue").textContent = items.length || "";
  } catch (e) {
    /* silent */
  }
}

// ---------------------------------------------------------------------------
// CORPUS view
// ---------------------------------------------------------------------------
async function loadCorpus() {
  const el = document.getElementById("corpus-list");
  el.innerHTML = `<p class="muted pad">Loading corpus…</p>`;
  try {
    const res = await fetch(`${API_BASE}/api/corpus`);
    const docs = await res.json();
    el.innerHTML = Object.entries(docs)
      .map(
        ([title, clauses]) => `
      <div class="corpus-doc">
        <div class="corpus-doc-title">${esc(title)}</div>
        ${clauses
          .map(
            (c) => `
          <div class="corpus-clause">
            <div class="corpus-clause-section">${esc(c.section)}</div>
            <div class="corpus-clause-text">${esc(c.text)}</div>
          </div>`
          )
          .join("")}
      </div>`
      )
      .join("");
  } catch (e) {
    el.innerHTML = `<p class="muted pad">Could not load corpus — is the backend running?</p>`;
  }
}

// ---------------------------------------------------------------------------
// AUDIT LOG view
// ---------------------------------------------------------------------------
async function loadAudit() {
  const tbody = document.getElementById("audit-tbody");
  tbody.innerHTML = `<tr><td colspan="9" class="muted pad">Loading…</td></tr>`;
  try {
    const res = await fetch(`${API_BASE}/api/audit`);
    const rows = await res.json();
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="muted pad">No audit entries yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows
      .map(
        (r) => `
      <tr data-id="${esc(r.id)}">
        <td>${fmtDate(r.created_at)}</td>
        <td class="wrap">${esc(r.regulation_title)}</td>
        <td class="wrap">${esc(r.doc_title)} [${esc(r.section)}]</td>
        <td>${impactTag(r.impact_type)}</td>
        <td>${r.confidence.toFixed(2)}</td>
        <td>${tagFor("risk", r.risk)}</td>
        <td>${esc(r.routing)}</td>
        <td>${tagFor("status", r.status)}</td>
        <td>${esc(r.reviewer_action || "—")}</td>
      </tr>`
      )
      .join("");

    tbody.querySelectorAll("tr").forEach((tr) => {
      tr.addEventListener("click", () => {
        const row = rows.find((r) => r.id === tr.dataset.id);
        if (row) openAuditDrawer(row);
      });
    });
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="9" class="muted pad">Could not load audit log — is the backend running?</td></tr>`;
  }
}

function openAuditDrawer(row) {
  const trace = row.reasoning_trace
    .map((step, i) => `<li class="trace-step"><span class="trace-num">${i + 1}</span><span class="trace-detail">${esc(step.detail)}</span></li>`)
    .join("");
  showDrawer(`
    <h2 style="font-family:var(--font-display); font-size:20px; margin:0 0 4px 0;">${esc(row.doc_title)} — ${esc(row.section)}</h2>
    <p class="muted" style="font-size:12.5px; margin:0 0 20px 0;">${esc(row.regulation_title)} · logged ${fmtDate(row.created_at)}</p>
    <div class="stat-line"><span>Impact</span><span>${esc(row.impact_type)}</span></div>
    <div class="stat-line"><span>Confidence</span><span>${row.confidence.toFixed(2)}</span></div>
    <div class="stat-line"><span>Risk</span><span>${esc(row.risk)}</span></div>
    <div class="stat-line"><span>Routing</span><span>${esc(row.routing)}</span></div>
    <div class="stat-line"><span>Citation verified</span><span>${row.citation_verified ? "yes" : "no"}</span></div>
    <div class="stat-line"><span>Status</span><span>${esc(row.status)}</span></div>
    <div class="stat-line"><span>Reviewer note</span><span>${esc(row.reviewer_notes || "—")}</span></div>
    <div class="trace" style="margin-top:20px;">
      <h4>Reasoning trace</h4>
      <ul class="trace-steps">${trace}</ul>
    </div>
  `);
}

// ---------------------------------------------------------------------------
// EVAL view
// ---------------------------------------------------------------------------
document.getElementById("run-eval").addEventListener("click", async () => {
  const btn = document.getElementById("run-eval");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const res = await fetch(`${API_BASE}/api/eval`);
    const data = await res.json();
    renderEval(data);
  } catch (e) {
    document.getElementById("eval-metrics").innerHTML = `<p style="color:var(--risk-high)">Evaluation failed — is the backend running?</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run evaluation";
  }
});

function renderEval(data) {
  const stat = (label, value) => `
    <div class="eval-stat">
      <div class="eval-stat-value">${value}</div>
      <div class="eval-stat-label">${label}</div>
    </div>`;

  document.getElementById("eval-metrics").innerHTML = [
    stat("Labeled pairs", data.n_labels),
    stat("Retrieval recall@5", data.retrieval_recall_at_5),
    stat("Classification precision", data.classification_precision),
    stat("Classification recall", data.classification_recall),
    stat("Classification F1", data.classification_f1),
    stat("Impact-type accuracy", data.impact_type_accuracy ?? "—"),
    stat("Citation hallucination rate", data.citation_hallucination_rate),
  ].join("");

  document.getElementById("eval-detail").innerHTML = data.detail
    .map((d) => {
      const correct = d.actual_impacted === d.predicted_impacted;
      return `
      <div class="eval-detail-row">
        <div class="row1">${esc(d.regulation_clause_text)} → <strong>${esc(d.target)}</strong></div>
        <div class="row2 ${correct ? "correct" : "incorrect"}">
          retrieved: ${d.target_retrieved} · actual: ${d.actual_impacted ? "impacted" : "not impacted"}
          (${esc(d.true_impact_type)}) · predicted: ${d.predicted_impacted ? "impacted" : "not impacted"}
          ${d.predicted_impact_type ? "(" + esc(d.predicted_impact_type) + ")" : ""}
        </div>
      </div>`;
    })
    .join("");
}

// ---------------------------------------------------------------------------
// Drawer
// ---------------------------------------------------------------------------
const drawer = document.getElementById("drawer");
function showDrawer(html) {
  document.getElementById("drawer-content").innerHTML = html;
  drawer.classList.add("open");
}
document.getElementById("drawer-close").addEventListener("click", () => drawer.classList.remove("open"));
document.getElementById("drawer-backdrop").addEventListener("click", () => drawer.classList.remove("open"));

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
checkHealth();
loadRegulations();
refreshQueueCount();
