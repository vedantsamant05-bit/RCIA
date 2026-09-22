/* ==========================================================================
   RCIA Dashboard — app.js
   Vanilla JS, no build step. Talks to the FastAPI backend at API_BASE.

   Safe version:
   - Waits for the DOM before wiring event handlers.
   - Guards optional/missing UI elements.
   - Keeps existing API behavior unchanged.
   ========================================================================== */

(function () {
  "use strict";

  function initRCIA() {
    const API_BASE =
      (window.RCIA_CONFIG && window.RCIA_CONFIG.API_BASE) ||
      window.location.origin;

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

        views.forEach((v) => {
          v.classList.toggle("active", v.id === `view-${target}`);
        });

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

      if (!dot || !text) return;

      try {
        const res = await fetch(`${API_BASE}/api/health`);

        if (!res.ok) {
          throw new Error("Health check failed");
        }

        const data = await res.json();

        dot.className = "dot ok";

        text.textContent = data.llm_enabled
          ? "Connected · LLM reasoning on"
          : "Connected · heuristic mode";
      } catch (e) {
        dot.className = "dot err";
        text.textContent =
          "Backend unreachable — start the API server";
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

      if (Number.isNaN(d.getTime())) return "—";

      return d.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    }

    function tagFor(kind, value) {
      return `
        <span class="tag tag-${kind}-${value}">
          ${esc(value).replace(/_/g, " ")}
        </span>
      `;
    }

    function impactTag(impact) {
      return `
        <span class="tag tag-${impact}">
          ${esc(impact)}
        </span>
      `;
    }

    // ---------------------------------------------------------------------------
    // INGEST view
    // ---------------------------------------------------------------------------
    const ingestForm = document.getElementById("ingest-form");
    const ingestResultEl =
      document.getElementById("ingest-result");
    const ingestSubmitBtn =
      document.getElementById("ingest-submit");

    if (
      ingestForm &&
      ingestResultEl &&
      ingestSubmitBtn
    ) {
      ingestForm.addEventListener(
        "submit",
        async (e) => {
          e.preventDefault();

          const titleEl =
            document.getElementById("reg-title");
          const sourceEl =
            document.getElementById("reg-source");
          const textEl =
            document.getElementById("reg-text");

          if (!titleEl || !sourceEl || !textEl) {
            ingestResultEl.innerHTML = `
              <p style="color:var(--risk-high)">
                Required form fields are missing from the page.
              </p>
            `;
            return;
          }

          const title = titleEl.value.trim();
          const source = sourceEl.value.trim();
          const text = textEl.value.trim();

          if (!title || !text) {
            return;
          }

          ingestSubmitBtn.disabled = true;
          ingestSubmitBtn.textContent =
            "Running pipeline…";

          ingestResultEl.innerHTML = `
            <p class="muted">
              Segmenting clauses, retrieving candidates,
              running the agent state machine…
            </p>
          `;

          try {
            const res = await fetch(
              `${API_BASE}/api/regulations/ingest`,
              {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  title,
                  source: source || null,
                  text,
                }),
              }
            );

            if (!res.ok) {
              throw new Error(await res.text());
            }

            const data = await res.json();

            ingestResultEl.innerHTML = `
              <h3>Pipeline complete</h3>

              <div class="stat-line">
                <span>Regulation ID</span>
                <span>${esc(data.regulation_id)}</span>
              </div>

              <div class="stat-line">
                <span>Clauses analyzed</span>
                <strong>${data.clauses_extracted}</strong>
              </div>

              <div class="stat-line">
                <span>Candidates retrieved</span>
                <strong>${data.candidates_retrieved}</strong>
              </div>

              <div class="stat-line">
                <span>Material impacts</span>
                <strong>${data.material_impacts}</strong>
              </div>

              <div class="stat-line">
                <span>Irrelevant candidates auto-dismissed</span>
                <strong>${data.irrelevant_dismissed}</strong>
              </div>

              <div class="stat-line">
                <span>Human-review findings</span>
                <strong>${data.human_review_findings}</strong>
              </div>

              <p
                class="muted"
                style="margin-top:16px; font-size:12.5px;"
              >
                Open the Review queue tab to see flagged
                clauses, drafted redlines, and reasoning traces.
              </p>
            `;

            ingestForm.reset();

            loadRegulations();
            refreshQueueCount();
          } catch (err) {
            ingestResultEl.innerHTML = `
              <p style="color:var(--risk-high)">
                Pipeline failed: ${esc(err.message)}
              </p>
            `;
          } finally {
            ingestSubmitBtn.disabled = false;
            ingestSubmitBtn.textContent =
              "Run pipeline";
          }
        }
      );
    }

    // ---------------------------------------------------------------------------
    // Load sample
    // ---------------------------------------------------------------------------
    const loadSampleBtn =
      document.getElementById("load-sample");

    if (loadSampleBtn) {
      loadSampleBtn.addEventListener(
        "click",
        async () => {
          try {
            const res = await fetch(
              `${API_BASE}/api/samples`
            );

            if (!res.ok) {
              throw new Error("Unable to load samples");
            }

            const samples = await res.json();

            if (!Array.isArray(samples) || samples.length === 0) {
              throw new Error("No sample regulations available");
            }

            const pick =
              samples[
              Math.floor(
                Math.random() * samples.length
              )
              ];

            const titleEl =
              document.getElementById("reg-title");
            const sourceEl =
              document.getElementById("reg-source");
            const textEl =
              document.getElementById("reg-text");

            if (titleEl) {
              titleEl.value = pick.title || "";
            }

            if (sourceEl) {
              sourceEl.value =
                pick.source || "";
            }

            if (textEl) {
              textEl.value =
                pick.text || "";
            }
          } catch (e) {
            if (ingestResultEl) {
              ingestResultEl.innerHTML = `
                <p style="color:var(--risk-high)">
                  Could not load sample regulations.
                </p>
              `;
            }
          }
        }
      );
    }

    // ---------------------------------------------------------------------------
    // Regulations list
    // ---------------------------------------------------------------------------
    async function loadRegulations() {
      const el =
        document.getElementById("regulations-list");

      if (!el) return;

      try {
        const res = await fetch(
          `${API_BASE}/api/regulations`
        );

        if (!res.ok) {
          throw new Error("Could not load regulations");
        }

        const regs = await res.json();

        if (!Array.isArray(regs) || regs.length === 0) {
          el.innerHTML = `
            <p class="muted pad">
              No regulations ingested yet.
            </p>
          `;
          return;
        }

        el.innerHTML = regs
          .map(
            (r) => `
              <div
                class="ledger-row"
                data-reg-id="${esc(r.id)}"
              >
                <div class="ledger-row-main">
                  <div class="ledger-row-title">
                    ${esc(r.title)}
                  </div>

                  <div class="ledger-row-sub">
                    ${esc(
              r.source ||
              "no source noted"
            )}
                    · ${esc(r.id)}
                  </div>
                </div>

                <div class="ledger-row-meta">
                  ${fmtDate(r.ingested_at)}
                </div>
              </div>
            `
          )
          .join("");

        el.querySelectorAll(".ledger-row").forEach(
          (row) => {
            row.addEventListener(
              "click",
              () =>
                openRegulationDrawer(
                  row.dataset.regId
                )
            );
          }
        );
      } catch (e) {
        el.innerHTML = `
          <p class="muted pad">
            Could not load regulations —
            is the backend running?
          </p>
        `;
      }
    }

    // ---------------------------------------------------------------------------
    // Regulation drawer
    // ---------------------------------------------------------------------------
    async function openRegulationDrawer(regId) {
      try {
        const res = await fetch(
          `${API_BASE}/api/regulations/${regId}/results`
        );

        if (!res.ok) {
          throw new Error(
            "Could not load regulation results"
          );
        }

        const data = await res.json();

        const items =
          Array.isArray(data.items)
            ? data.items
            : [];

        const content = `
          <h2
            style="
              font-family:var(--font-display);
              font-size:20px;
              margin:0 0 4px 0;
            "
          >
            ${esc(data.regulation.title)}
          </h2>

          <p
            class="muted"
            style="
              font-size:12.5px;
              margin:0 0 20px 0;
            "
          >
            ${esc(
          data.regulation.source || ""
        )}
          </p>

          <div class="ledger">
            ${items
            .map(
              (item) => `
                  <div
                    class="ledger-row"
                    style="cursor:default;"
                  >
                    <div class="ledger-row-main">
                      <div class="ledger-row-title">
                        ${esc(item.doc_title)}
                        [${esc(item.section)}]
                      </div>

                      <div class="ledger-row-sub">
                        ${impactTag(
                item.impact_type
              )}
                        &nbsp; conf
                        ${typeof item.confidence ===
                  "number"
                  ? item.confidence.toFixed(
                    2
                  )
                  : "—"
                }
                      </div>
                    </div>

                    <div class="ledger-row-meta">
                      ${tagFor(
                  "status",
                  item.status
                )}
                    </div>
                  </div>
                `
            )
            .join("")}
          </div>
        `;

        showDrawer(content);
      } catch (e) {
        showDrawer(`
          <p class="muted">
            Could not load regulation results.
          </p>
        `);
      }
    }

    // ---------------------------------------------------------------------------
    // REVIEW QUEUE
    // ---------------------------------------------------------------------------
    let currentStatusFilter =
      "pending_material";

    document
      .querySelectorAll(".chip")
      .forEach((chip) => {
        chip.addEventListener("click", () => {
          document
            .querySelectorAll(".chip")
            .forEach((c) =>
              c.classList.remove("active")
            );

          chip.classList.add("active");

          currentStatusFilter =
            chip.dataset.status || "";

          loadQueue();
        });
      });

    async function loadQueue() {
      const el =
        document.getElementById("queue-list");

      if (!el) return;

      el.innerHTML = `
        <p class="muted pad">
          Loading review queue…
        </p>
      `;

      try {
        // Fetch all items for summary statistics.
        const allRes = await fetch(
          `${API_BASE}/api/review`
        );

        if (!allRes.ok) {
          throw new Error(
            "Could not load review queue"
          );
        }

        const allItems = await allRes.json();

        const safeItems =
          Array.isArray(allItems)
            ? allItems
            : [];

        const uniqueClauses =
          new Set(
            safeItems.map(
              (i) => i.regulation_clause_id
            )
          ).size;

        const totalCandidates =
          safeItems.length;

        const materialImpacts =
          safeItems.filter(
            (i) =>
              i.impact_type !== "no-op"
          ).length;

        const autoDismissed =
          safeItems.filter(
            (i) =>
              i.routing === "auto_dismiss"
          ).length;

        const elClauses =
          document.getElementById(
            "stat-clauses"
          );

        const elCandidates =
          document.getElementById(
            "stat-candidates"
          );

        const elImpacts =
          document.getElementById(
            "stat-impacts"
          );

        const elDismissed =
          document.getElementById(
            "stat-dismissed"
          );

        if (elClauses) {
          elClauses.textContent =
            uniqueClauses;
        }

        if (elCandidates) {
          elCandidates.textContent =
            totalCandidates;
        }

        if (elImpacts) {
          elImpacts.textContent =
            materialImpacts;
        }

        if (elDismissed) {
          elDismissed.textContent =
            autoDismissed;
        }

        const url = currentStatusFilter
          ? `${API_BASE}/api/review?status=${encodeURIComponent(
            currentStatusFilter
          )}`
          : `${API_BASE}/api/review`;

        const res = await fetch(url);

        if (!res.ok) {
          throw new Error(
            "Could not load filtered queue"
          );
        }

        const items = await res.json();

        if (
          !Array.isArray(items) ||
          items.length === 0
        ) {
          el.innerHTML = `
            <p class="muted pad">
              No items in this view.
            </p>
          `;
          return;
        }

        el.innerHTML = items
          .map((item) =>
            queueItemHtml(item)
          )
          .join("");

        el
          .querySelectorAll(
            ".queue-item-head"
          )
          .forEach((head) => {
            head.addEventListener(
              "click",
              () => {
                const parent =
                  head.closest(
                    ".queue-item"
                  );

                if (parent) {
                  parent.classList.toggle(
                    "open"
                  );
                }
              }
            );
          });

        el
          .querySelectorAll(
            "[data-action]"
          )
          .forEach((btn) => {
            btn.addEventListener(
              "click",
              async (e) => {
                e.stopPropagation();

                const id =
                  btn.dataset.id;

                const action =
                  btn.dataset.action;

                if (!id || !action) {
                  return;
                }

                btn.disabled = true;

                try {
                  const response =
                    await fetch(
                      `${API_BASE}/api/review/${id}`,
                      {
                        method: "POST",
                        headers: {
                          "Content-Type":
                            "application/json",
                        },
                        body: JSON.stringify({
                          action,
                        }),
                      }
                    );

                  if (!response.ok) {
                    throw new Error(
                      await response.text()
                    );
                  }

                  await loadQueue();
                  await refreshQueueCount();
                } catch (error) {
                  alert(
                    "Review action failed: " +
                    error.message
                  );
                  btn.disabled = false;
                }
              }
            );
          });
      } catch (e) {
        el.innerHTML = `
          <p class="muted pad">
            Could not load review queue —
            is the backend running?
          </p>
        `;
      }
    }

    // ---------------------------------------------------------------------------
    // Redline formatting
    // ---------------------------------------------------------------------------
    function formatRedlineHtml(redlineText) {
      if (!redlineText) {
        return `
          <p class="muted">
            No redline drafted.
          </p>
        `;
      }

      if (
        redlineText.includes("CURRENT:") &&
        redlineText.includes("PROPOSED:")
      ) {
        const parts =
          redlineText.split(
            "PROPOSED:"
          );

        let currentText =
          parts[0]
            .replace("CURRENT:", "")
            .replace(
              "DRAFT ONLY - HUMAN REVIEW REQUIRED",
              ""
            )
            .trim();

        let proposedText =
          parts
            .slice(1)
            .join("PROPOSED:")
            .trim();

        return `
          <div class="redline-split">
            <div
              class="redline-box redline-current"
            >
              <div class="redline-label">
                CURRENT POLICY WORDING
              </div>

              <div class="redline-text">
                ${esc(currentText)}
              </div>
            </div>

            <div
              class="redline-box redline-proposed"
            >
              <div class="redline-label">
                PROPOSED REDLINE AMENDMENT
              </div>

              <div class="redline-text">
                ${esc(proposedText)}
              </div>
            </div>
          </div>
        `;
      }

      return `<p>${esc(redlineText)}</p>`;
    }

    // ---------------------------------------------------------------------------
    // Business action formatting
    // ---------------------------------------------------------------------------
    function formatBusinessActionHtml(
      actionText
    ) {
      if (
        !actionText ||
        actionText.includes(
          "No business action required"
        )
      ) {
        return "";
      }

      const lines = actionText
        .split("\n")
        .filter(
          (l) => l.trim().length > 0
        )
        .filter(
          (l) =>
            l.trim() !==
            "DRAFT ONLY - HUMAN REVIEW REQUIRED"
        );

      const formattedLines =
        lines
          .map((line) => {
            const l =
              esc(line.trim());

            if (
              l.startsWith(
                "ACTION REQUIRED:"
              )
            ) {
              return `
                <div class="action-highlight">
                  ${l.replace(
                "ACTION REQUIRED:",
                "<strong>⚡ PRIMARY ACTION:</strong>"
              )}
                </div>
              `;
            }

            if (
              l.startsWith(
                "TARGET POLICY:"
              )
            ) {
              return `
                <div class="action-target">
                  🎯 ${l}
                </div>
              `;
            }

            if (
              l.startsWith(
                "DEADLINE:"
              )
            ) {
              return `
                <div class="action-deadline">
                  📅 ${l}
                </div>
              `;
            }

            if (
              l.startsWith(
                "STEPS:"
              )
            ) {
              return `
                <div class="action-steps-title">
                  Implementation Steps:
                </div>
              `;
            }

            return `
              <div class="action-step-item">
                ${l}
              </div>
            `;
          })
          .join("");

      return `
        <div class="business-action-box">
          <div class="business-action-header">
            ⚡ BUSINESS ACTION REQUIRED
          </div>

          <div class="business-action-body">
            ${formattedLines}
          </div>
        </div>
      `;
    }

    // ---------------------------------------------------------------------------
    // Queue item
    // ---------------------------------------------------------------------------
    function queueItemHtml(item) {
      const trace =
        Array.isArray(
          item.reasoning_trace
        )
          ? item.reasoning_trace
            .map(
              (step, i) => `
                  <li class="trace-step">
                    <span class="trace-num">
                      ${i + 1}
                    </span>

                    <span class="trace-detail">
                      ${esc(
                step.detail
              )}
                    </span>
                  </li>
                `
            )
            .join("")
          : "";

      const needsAction =
        item.status === "pending";

      const confidence =
        typeof item.confidence ===
          "number"
          ? item.confidence.toFixed(
            2
          )
          : "—";

      return `
        <div class="queue-item">

          <div class="queue-item-head">

            ${impactTag(
        item.impact_type
      )}

            <div class="queue-item-titles">

              <div class="queue-item-reg">
                ${esc(
        item.regulation_title
      )}
              </div>

              <div class="queue-item-clause">
                ${esc(
        item.doc_title
      )}
                —
                Section
                ${esc(item.section)}
              </div>

            </div>

            ${tagFor(
        "risk",
        item.risk
      )}

            ${tagFor(
        "status",
        item.status
      )}

            <div class="queue-item-conf">
              ${confidence} / 1.00
              <span class="conf-mode">
                (Heuristic)
              </span>
            </div>

          </div>

          <div class="queue-item-body">

            <div class="compare-cols">

              <div class="compare-col">

                <h4>
                  New regulatory clause
                </h4>

                <p>
                  ${esc(
        item.regulation_clause_text
      )}
                </p>

              </div>

              <div class="compare-col">

                <h4>
                  Existing internal clause
                </h4>

                <p>
                  ${esc(
        item.internal_clause_text
      )}
                </p>

              </div>

            </div>

            ${formatBusinessActionHtml(
        item.business_action
      )}

            <div class="draft-block">

              <h4>

                Drafted redline

                ${item.citation_verified
          ? `
                      <span
                        class="tag"
                        style="
                          color:var(--verified);
                          border-color:var(--verified);
                          margin-left:8px;
                        "
                      >
                        source grounded
                      </span>
                    `
          : `
                      <span
                        class="tag"
                        style="
                          color:var(--unverified);
                          border-color:var(--unverified);
                          margin-left:8px;
                        "
                      >
                        source not grounded
                      </span>
                    `
        }

              </h4>

              ${formatRedlineHtml(
          item.draft_redline
        )}

            </div>

            <div class="trace">

              <h4>
                Reasoning trace
              </h4>

              <ul class="trace-steps">
                ${trace}
              </ul>

            </div>

            ${needsAction
          ? `
                  <div class="actions-row">

                    <button
                      class="btn btn-approve"
                      data-action="approve"
                      data-id="${esc(
            item.id
          )}"
                    >
                      Approve redline
                    </button>

                    <button
                      class="btn btn-reject"
                      data-action="reject"
                      data-id="${esc(
            item.id
          )}"
                    >
                      Reject
                    </button>

                    ${item.routing ===
            "escalate"
            ? `
                          <button
                            class="btn btn-ghost"
                            data-action="escalate_confirmed"
                            data-id="${esc(
              item.id
            )}"
                          >
                            Confirm escalation reviewed
                          </button>
                        `
            : ""
          }

                  </div>
                `
          : `
                  <div class="reviewer-note">
                    Reviewer action:
                    ${esc(
            item.reviewer_action ||
            item.status
          )}

                    ${item.reviewed_at
            ? " · " +
            fmtDate(
              item.reviewed_at
            )
            : ""
          }
                  </div>
                `
        }

          </div>
        </div>
      `;
    }

    // ---------------------------------------------------------------------------
    // Queue count
    // ---------------------------------------------------------------------------
    async function refreshQueueCount() {
      try {
        const res =
          await fetch(
            `${API_BASE}/api/review?status=pending`
          );

        if (!res.ok) {
          throw new Error(
            "Could not load queue count"
          );
        }

        const items =
          await res.json();

        const countQueue =
          document.getElementById(
            "count-queue"
          );

        if (countQueue) {
          countQueue.textContent =
            Array.isArray(items) &&
              items.length
              ? items.length
              : "";
        }
      } catch (e) {
        // Silent by design.
      }
    }

    // ---------------------------------------------------------------------------
    // CORPUS view
    // ---------------------------------------------------------------------------
    async function loadCorpus() {
      const el =
        document.getElementById(
          "corpus-list"
        );

      if (!el) return;

      el.innerHTML = `
        <p class="muted pad">
          Loading corpus…
        </p>
      `;

      try {
        const res =
          await fetch(
            `${API_BASE}/api/corpus`
          );

        if (!res.ok) {
          throw new Error(
            "Could not load corpus"
          );
        }

        const docs =
          await res.json();

        el.innerHTML =
          Object.entries(
            docs || {}
          )
            .map(
              ([title, clauses]) => `
                <div class="corpus-doc">

                  <div class="corpus-doc-title">
                    ${esc(title)}
                  </div>

                  ${Array.isArray(clauses)
                  ? clauses
                    .map(
                      (c) => `
                              <div class="corpus-clause">

                                <div
                                  class="corpus-clause-section"
                                >
                                  ${esc(
                        c.section
                      )}
                                </div>

                                <div
                                  class="corpus-clause-text"
                                >
                                  ${esc(
                        c.text
                      )}
                                </div>

                              </div>
                            `
                    )
                    .join("")
                  : ""
                }

                </div>
              `
            )
            .join("");
      } catch (e) {
        el.innerHTML = `
          <p class="muted pad">
            Could not load corpus —
            is the backend running?
          </p>
        `;
      }
    }

    // ---------------------------------------------------------------------------
    // AUDIT LOG
    // ---------------------------------------------------------------------------
    async function loadAudit() {
      const tbody =
        document.getElementById(
          "audit-tbody"
        );

      if (!tbody) return;

      tbody.innerHTML = `
        <tr>
          <td
            colspan="9"
            class="muted pad"
          >
            Loading…
          </td>
        </tr>
      `;

      try {
        const res =
          await fetch(
            `${API_BASE}/api/audit`
          );

        if (!res.ok) {
          throw new Error(
            "Could not load audit log"
          );
        }

        const rows =
          await res.json();

        if (
          !Array.isArray(rows) ||
          rows.length === 0
        ) {
          tbody.innerHTML = `
            <tr>
              <td
                colspan="9"
                class="muted pad"
              >
                No audit entries yet.
              </td>
            </tr>
          `;
          return;
        }

        tbody.innerHTML =
          rows
            .map(
              (r) => `
                <tr
                  data-id="${esc(
                r.id
              )}"
                >

                  <td>
                    ${fmtDate(
                r.created_at
              )}
                  </td>

                  <td class="wrap">
                    ${esc(
                r.regulation_title
              )}
                  </td>

                  <td class="wrap">
                    ${esc(
                r.doc_title
              )}
                    [
                    ${esc(
                r.section
              )}
                    ]
                  </td>

                  <td>
                    ${impactTag(
                r.impact_type
              )}
                  </td>

                  <td>
                    ${typeof r.confidence ===
                  "number"
                  ? r.confidence.toFixed(
                    2
                  )
                  : "—"
                }
                  </td>

                  <td>
                    ${tagFor(
                  "risk",
                  r.risk
                )}
                  </td>

                  <td>
                    ${esc(
                  r.routing
                )}
                  </td>

                  <td>
                    ${tagFor(
                  "status",
                  r.status
                )}
                  </td>

                  <td>
                    ${esc(
                  r.reviewer_action ||
                  "—"
                )}
                  </td>

                </tr>
              `
            )
            .join("");

        tbody
          .querySelectorAll("tr")
          .forEach((tr) => {
            tr.addEventListener(
              "click",
              () => {
                const row =
                  rows.find(
                    (r) =>
                      r.id ===
                      tr.dataset.id
                  );

                if (row) {
                  openAuditDrawer(
                    row
                  );
                }
              }
            );
          });
      } catch (e) {
        tbody.innerHTML = `
          <tr>
            <td
              colspan="9"
              class="muted pad"
            >
              Could not load audit log —
              is the backend running?
            </td>
          </tr>
        `;
      }
    }

    // ---------------------------------------------------------------------------
    // Audit drawer
    // ---------------------------------------------------------------------------
    function openAuditDrawer(row) {
      const trace =
        Array.isArray(
          row.reasoning_trace
        )
          ? row.reasoning_trace
            .map(
              (step, i) => `
                  <li class="trace-step">
                    <span class="trace-num">
                      ${i + 1}
                    </span>

                    <span class="trace-detail">
                      ${esc(
                step.detail
              )}
                    </span>
                  </li>
                `
            )
            .join("")
          : "";

      showDrawer(`
        <h2
          style="
            font-family:var(--font-display);
            font-size:20px;
            margin:0 0 4px 0;
          "
        >
          ${esc(
        row.doc_title
      )}
          —
          ${esc(row.section)}
        </h2>

        <p
          class="muted"
          style="
            font-size:12.5px;
            margin:0 0 20px 0;
          "
        >
          ${esc(
        row.regulation_title
      )}
          · logged
          ${fmtDate(
        row.created_at
      )}
        </p>

        <div class="stat-line">
          <span>Impact</span>
          <span>
            ${esc(
        row.impact_type
      )}
          </span>
        </div>

        <div class="stat-line">
          <span>Confidence</span>
          <span>
            ${typeof row.confidence ===
          "number"
          ? row.confidence.toFixed(
            2
          )
          : "—"
        }
          </span>
        </div>

        <div class="stat-line">
          <span>Risk</span>
          <span>
            ${esc(row.risk)}
          </span>
        </div>

        <div class="stat-line">
          <span>Routing</span>
          <span>
            ${esc(
          row.routing
        )}
          </span>
        </div>

        <div class="stat-line">
          <span>Source grounded</span>
          <span>
            ${row.citation_verified
          ? "yes"
          : "no"
        }
          </span>
        </div>

        <div class="stat-line">
          <span>Status</span>
          <span>
            ${esc(
          row.status
        )}
          </span>
        </div>

        <div class="stat-line">
          <span>Reviewer note</span>
          <span>
            ${esc(
          row.reviewer_notes ||
          "—"
        )}
          </span>
        </div>

        <div
          class="trace"
          style="margin-top:20px;"
        >
          <h4>
            Reasoning trace
          </h4>

          <ul class="trace-steps">
            ${trace}
          </ul>
        </div>
      `);
    }

    // ---------------------------------------------------------------------------
    // Evaluation
    // ---------------------------------------------------------------------------
    const runEvalBtn =
      document.getElementById(
        "run-eval"
      );

    if (runEvalBtn) {
      runEvalBtn.addEventListener(
        "click",
        async () => {
          runEvalBtn.disabled = true;
          runEvalBtn.textContent =
            "Running…";

          try {
            const res =
              await fetch(
                `${API_BASE}/api/eval`
              );

            if (!res.ok) {
              throw new Error(
                await res.text()
              );
            }

            const data =
              await res.json();

            renderEval(data);
          } catch (e) {
            const evalMetrics =
              document.getElementById(
                "eval-metrics"
              );

            if (evalMetrics) {
              evalMetrics.innerHTML = `
                <p
                  style="color:var(--risk-high)"
                >
                  Evaluation failed —
                  is the backend running?
                </p>
              `;
            }
          } finally {
            runEvalBtn.disabled =
              false;

            runEvalBtn.textContent =
              "Run evaluation";
          }
        }
      );
    }

    // ---------------------------------------------------------------------------
    // Evaluation rendering
    // ---------------------------------------------------------------------------
    function renderEval(data) {
      const evalMetrics =
        document.getElementById(
          "eval-metrics"
        );

      const evalDetail =
        document.getElementById(
          "eval-detail"
        );

      if (!evalMetrics || !evalDetail) {
        return;
      }

      const stat = (
        label,
        value
      ) => `
        <div class="eval-stat">

          <div class="eval-stat-value">
            ${esc(value)}
          </div>

          <div class="eval-stat-label">
            ${esc(label)}
          </div>

        </div>
      `;

      evalMetrics.innerHTML = [
        stat(
          "Labeled pairs",
          data.n_labels
        ),

        stat(
          "Retrieval recall@5",
          data.retrieval_recall_at_5
        ),

        stat(
          "Classification precision",
          data.classification_precision
        ),

        stat(
          "Classification recall",
          data.classification_recall
        ),

        stat(
          "Classification F1",
          data.classification_f1
        ),

        stat(
          "Impact-type accuracy",
          data.impact_type_accuracy ??
          "—"
        ),

        stat(
          "Citation hallucination rate",
          data.citation_hallucination_rate
        ),
      ].join("");

      const details =
        Array.isArray(
          data.detail
        )
          ? data.detail
          : [];

      evalDetail.innerHTML =
        details
          .map((d) => {
            const correct =
              d.actual_impacted ===
              d.predicted_impacted;

            return `
              <div class="eval-detail-row">

                <div class="row1">
                  ${esc(
              d.regulation_clause_text
            )}
                  →
                  <strong>
                    ${esc(
              d.target
            )}
                  </strong>
                </div>

                <div
                  class="
                    row2
                    ${correct
                ? "correct"
                : "incorrect"
              }
                  "
                >
                  retrieved:
                  ${d.target_retrieved}

                  · actual:
                  ${d.actual_impacted
                ? "impacted"
                : "not impacted"
              }

                  (
                  ${esc(
                d.true_impact_type
              )}
                  )

                  · predicted:
                  ${d.predicted_impacted
                ? "impacted"
                : "not impacted"
              }

                  ${d.predicted_impact_type
                ? `
                        (
                        ${esc(
                  d.predicted_impact_type
                )}
                        )
                      `
                : ""
              }
                </div>

              </div>
            `;
          })
          .join("");
    }

    // ---------------------------------------------------------------------------
    // Drawer
    // ---------------------------------------------------------------------------
    const drawer =
      document.getElementById(
        "drawer"
      );

    const drawerContent =
      document.getElementById(
        "drawer-content"
      );

    function showDrawer(html) {
      if (
        !drawer ||
        !drawerContent
      ) {
        return;
      }

      drawerContent.innerHTML =
        html;

      drawer.classList.add(
        "open"
      );
    }

    const drawerClose =
      document.getElementById(
        "drawer-close"
      );

    const drawerBackdrop =
      document.getElementById(
        "drawer-backdrop"
      );

    if (
      drawerClose &&
      drawer
    ) {
      drawerClose.addEventListener(
        "click",
        () => {
          drawer.classList.remove(
            "open"
          );
        }
      );
    }

    if (
      drawerBackdrop &&
      drawer
    ) {
      drawerBackdrop.addEventListener(
        "click",
        () => {
          drawer.classList.remove(
            "open"
          );
        }
      );
    }

    // ---------------------------------------------------------------------------
    // Initial loading
    // ---------------------------------------------------------------------------
    checkHealth();
    loadRegulations();
    refreshQueueCount();
  }

  // ---------------------------------------------------------------------------
  // Start only after DOM is ready
  // ---------------------------------------------------------------------------
  if (
    document.readyState ===
    "loading"
  ) {
    document.addEventListener(
      "DOMContentLoaded",
      initRCIA
    );
  } else {
    initRCIA();
  }
})();