/**
 * Real Estate Price Prediction & Dynamic Property Matchmaker
 * Frontend Application Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  const state = {
    config: null,
    modelsSummary: null,
    datasetStats: null,
    featureChart: null,
    scatterChart: null,
    activeTab: "tab-predictor",
    activeSubtab: "exact",
    cspResults: null,
    catalogPage: 1,
    catalogTotalPages: 1,
    selectedCspLocations: new Set()
  };

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------
  async function init() {
    setupTabNavigation();
    setupRangeInputs();
    await loadConfig();
    await Promise.all([
      loadModelsSummary(),
      loadDatasetStats(),
      fetchCatalog(1),
      runInitialPrediction()
    ]);
    setupEventListeners();
  }

  // -------------------------------------------------------------------------
  // Navigation
  // -------------------------------------------------------------------------
  function setupTabNavigation() {
    const tabButtons = document.querySelectorAll(".nav-tabs .tab-btn");
    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-tab");
        tabButtons.forEach(b => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        const targetSection = document.getElementById(targetId);
        if (targetSection) targetSection.classList.add("active");
        state.activeTab = targetId;
      });
    });

    // Subtab navigation in CSP panel (Exact vs Relaxed)
    const btnExact = document.getElementById("subtab-exact");
    const btnRelaxed = document.getElementById("subtab-relaxed");
    if (btnExact && btnRelaxed) {
      btnExact.addEventListener("click", () => {
        btnExact.classList.add("active");
        btnRelaxed.classList.remove("active");
        state.activeSubtab = "exact";
        renderCspMatchesList();
      });
      btnRelaxed.addEventListener("click", () => {
        btnRelaxed.classList.add("active");
        btnExact.classList.remove("active");
        state.activeSubtab = "relaxed";
        renderCspMatchesList();
      });
    }
  }

  // -------------------------------------------------------------------------
  // Input Bindings
  // -------------------------------------------------------------------------
  function setupRangeInputs() {
    // Sqft sync
    const rangeSqft = document.getElementById("range-sqft");
    const inputSqft = document.getElementById("input-sqft");
    if (rangeSqft && inputSqft) {
      rangeSqft.addEventListener("input", (e) => {
        inputSqft.value = e.target.value;
      });
      inputSqft.addEventListener("input", (e) => {
        rangeSqft.value = e.target.value;
      });
    }

    // Distance sync
    const rangeDist = document.getElementById("input-distance");
    const distVal = document.getElementById("dist-val");
    if (rangeDist && distVal) {
      rangeDist.addEventListener("input", (e) => {
        distVal.textContent = e.target.value;
      });
    }

    // Commute sync (CSP)
    const rangeCommute = document.getElementById("csp-commute");
    const commuteVal = document.getElementById("csp-commute-val");
    if (rangeCommute && commuteVal) {
      rangeCommute.addEventListener("input", (e) => {
        commuteVal.textContent = e.target.value;
      });
    }
  }

  // -------------------------------------------------------------------------
  // Fetch Config (Locations, Amenities)
  // -------------------------------------------------------------------------
  async function loadConfig() {
    try {
      const res = await fetch("/api/config");
      const data = await res.json();
      state.config = data;

      // Populate Predictor Location Select
      const selectLoc = document.getElementById("input-location");
      const catalogLoc = document.getElementById("catalog-filter-location");
      if (selectLoc) {
        selectLoc.innerHTML = data.locations.map(loc => `<option value="${loc}">${loc}</option>`).join("");
      }
      if (catalogLoc) {
        data.locations.forEach(loc => {
          const opt = document.createElement("option");
          opt.value = loc;
          opt.textContent = loc;
          catalogLoc.appendChild(opt);
        });
      }

      // Populate Predictor Amenities Checkboxes
      const amenitiesContainer = document.getElementById("amenities-checkbox-grid");
      if (amenitiesContainer) {
        amenitiesContainer.innerHTML = data.amenities.map(am => `
          <label class="custom-checkbox">
            <input type="checkbox" name="pred-amenity" value="${am}">
            <span>${am}</span>
          </label>
        `).join("");
      }

      // Populate CSP Location Chips
      const cspChips = document.getElementById("csp-locations-chips");
      if (cspChips) {
        cspChips.innerHTML = data.locations.map(loc => `
          <span class="chip" data-loc="${loc}">${loc}</span>
        `).join("");

        cspChips.querySelectorAll(".chip").forEach(chip => {
          chip.addEventListener("click", () => {
            const loc = chip.getAttribute("data-loc");
            if (state.selectedCspLocations.has(loc)) {
              state.selectedCspLocations.delete(loc);
              chip.classList.remove("selected");
            } else {
              state.selectedCspLocations.add(loc);
              chip.classList.add("selected");
            }
          });
        });
      }

      // Populate CSP Essential Amenities (Hard)
      const cspEssential = document.getElementById("csp-essential-amenities");
      if (cspEssential) {
        cspEssential.innerHTML = data.amenities.map(am => `
          <label class="custom-checkbox">
            <input type="checkbox" name="csp-essential" value="${am}">
            <span>${am}</span>
          </label>
        `).join("");
      }

      // Populate CSP Desired Amenities (Soft)
      const cspDesired = document.getElementById("csp-desired-amenities");
      if (cspDesired) {
        cspDesired.innerHTML = data.amenities.map(am => `
          <label class="custom-checkbox">
            <input type="checkbox" name="csp-desired" value="${am}">
            <span>${am}</span>
          </label>
        `).join("");
      }

    } catch (err) {
      console.error("Failed to load config:", err);
    }
  }

  // -------------------------------------------------------------------------
  // Models Summary & Analytics
  // -------------------------------------------------------------------------
  async function loadModelsSummary() {
    try {
      const res = await fetch("/api/models/summary");
      const data = await res.json();
      state.modelsSummary = data;

      // Populate Metrics Benchmark Table
      const tbody = document.getElementById("metrics-tbody");
      if (tbody && data.metrics) {
        tbody.innerHTML = Object.entries(data.metrics).map(([name, m]) => {
          const isEnsemble = name.includes("Random Forest");
          return `
            <tr>
              <td><strong>${name}</strong></td>
              <td><span class="badge-tag ${isEnsemble ? 'highlight' : ''}">${isEnsemble ? 'Non-Linear Ensemble' : 'Linear / Parametric'}</span></td>
              <td><span style="color: var(--accent-emerald); font-weight: 600;">${m.r2_score.toFixed(4)}</span></td>
              <td>$${m.rmse.toLocaleString()}</td>
              <td>$${m.mae.toLocaleString()}</td>
              <td>${m.mape.toFixed(2)}%</td>
            </tr>
          `;
        }).join("");
      }

      // Render SLR formula
      if (data.slr_stats) {
        const eqEl = document.getElementById("slr-equation");
        if (eqEl) eqEl.textContent = data.slr_stats.equation;
      }

      // Render Feature Importance Chart
      renderFeatureImportanceChart(data.feature_importances);

    } catch (err) {
      console.error("Failed to load models summary:", err);
    }
  }

  function renderFeatureImportanceChart(featureImportances) {
    const canvas = document.getElementById("chart-feature-importance");
    if (!canvas || !featureImportances) return;

    const rfData = featureImportances["Random Forest (Non-Linear)"] || {};
    const sortedFeatures = Object.entries(rfData)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8); // Top 8 features

    const labels = sortedFeatures.map(f => {
      // Clean display names
      return f[0].replace("loc_", "Area: ").replace("has_", "Has ");
    });
    const values = sortedFeatures.map(f => (f[1] * 100).toFixed(1));

    if (state.featureChart) {
      state.featureChart.destroy();
    }

    const ctx = canvas.getContext("2d");
    state.featureChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Feature Importance (%)",
          data: values,
          backgroundColor: "rgba(6, 182, 212, 0.65)",
          borderColor: "#06b6d4",
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` Importance: ${ctx.parsed.x}%`
            }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#9ca3af", callback: (val) => val + "%" }
          },
          y: {
            grid: { display: false },
            ticks: { color: "#f3f4f6", font: { size: 11 } }
          }
        }
      }
    });
  }

  // -------------------------------------------------------------------------
  // Price Prediction Logic
  // -------------------------------------------------------------------------
  async function runInitialPrediction() {
    const form = document.getElementById("prediction-form");
    if (form) {
      executePrediction();
    }
  }

  async function executePrediction() {
    const sqft = parseFloat(document.getElementById("input-sqft").value) || 1850;
    const location = document.getElementById("input-location").value || "Downtown";
    const bedrooms = parseInt(document.getElementById("input-bedrooms").value) || 3;
    const bathrooms = parseFloat(document.getElementById("input-bathrooms").value) || 2.0;
    const age = parseInt(document.getElementById("input-age").value) || 8;
    const garage = parseInt(document.getElementById("input-garage").value) || 2;
    const distance = parseFloat(document.getElementById("input-distance").value) || 4.5;
    const selectedModel = document.getElementById("selected-model-selector").value;

    const checkedAmenities = Array.from(
      document.querySelectorAll("input[name='pred-amenity']:checked")
    ).map(cb => cb.value);

    const payload = {
      sqft: sqft,
      location: location,
      bedrooms: bedrooms,
      bathrooms: bathrooms,
      age: age,
      garage: garage,
      distance_to_center_km: distance,
      amenities: checkedAmenities,
      selected_model: selectedModel
    };

    try {
      const res = await fetch("/api/models/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      renderPredictionResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
    }
  }

  function renderPredictionResult(data) {
    const priceEl = document.getElementById("predicted-price");
    const sqftPriceEl = document.getElementById("predicted-price-sqft");
    const ciTextEl = document.getElementById("ci-range-text");
    const activeModelTitle = document.getElementById("active-model-title");

    if (priceEl) priceEl.textContent = `$${data.estimated_price.toLocaleString()}`;
    if (sqftPriceEl) sqftPriceEl.textContent = `$${data.price_per_sqft.toLocaleString()}/sqft`;
    if (ciTextEl) {
      ciTextEl.textContent = `$${data.confidence_interval_80.lower.toLocaleString()} to $${data.confidence_interval_80.upper.toLocaleString()}`;
    }
    if (activeModelTitle) {
      activeModelTitle.textContent = `Target Inference Model: ${data.selected_model}`;
    }

    // Comparison cards for all models
    const grid = document.getElementById("models-comparison-grid");
    if (grid && data.all_model_predictions) {
      grid.innerHTML = Object.entries(data.all_model_predictions).map(([name, price]) => {
        const isSelected = name === data.selected_model;
        const diff = price - data.estimated_price;
        const diffFormatted = diff === 0 ? "Baseline" : (diff > 0 ? `+$${diff.toLocaleString()}` : `-$${Math.abs(diff).toLocaleString()}`);
        return `
          <div class="model-stat-card ${isSelected ? 'active-model' : ''}">
            <div class="model-stat-name" title="${name}">${name}</div>
            <div class="model-stat-val">$${price.toLocaleString()}</div>
            <div class="model-stat-diff">${diffFormatted}</div>
          </div>
        `;
      }).join("");
    }

    // SLR Formula Calculation update
    const slrStep = document.getElementById("slr-calculation-step");
    if (slrStep && state.modelsSummary && state.modelsSummary.slr_stats) {
      const slope = state.modelsSummary.slr_stats.slope;
      const intercept = state.modelsSummary.slr_stats.intercept;
      const slrVal = intercept + (slope * data.input_summary.sqft);
      slrStep.textContent = `Calculated: $${intercept.toLocaleString(undefined, {maximumFractionDigits:0})} + ($${slope.toFixed(2)} × ${data.input_summary.sqft}) = $${Math.round(slrVal).toLocaleString()}`;
    }
  }

  // -------------------------------------------------------------------------
  // AI CSP Matchmaker Logic
  // -------------------------------------------------------------------------
  async function executeCspMatch(e) {
    if (e) e.preventDefault();

    const maxBudget = parseFloat(document.getElementById("csp-budget").value) || 650000;
    const minBedrooms = parseInt(document.getElementById("csp-bedrooms").value) || 3;
    const minBathrooms = parseFloat(document.getElementById("csp-bathrooms").value) || 2.0;
    const maxCommute = parseFloat(document.getElementById("csp-commute").value) || 12;
    const allowRelaxation = document.getElementById("csp-allow-relaxation").checked;

    const essentialAmenities = Array.from(
      document.querySelectorAll("input[name='csp-essential']:checked")
    ).map(cb => cb.value);

    const desiredAmenities = Array.from(
      document.querySelectorAll("input[name='csp-desired']:checked")
    ).map(cb => cb.value);

    const locations = Array.from(state.selectedCspLocations);

    const payload = {
      max_budget: maxBudget,
      preferred_locations: locations,
      min_bedrooms: minBedrooms,
      min_bathrooms: minBathrooms,
      essential_amenities: essentialAmenities,
      desired_amenities: desiredAmenities,
      max_commute_km: maxCommute,
      allow_relaxation: allowRelaxation
    };

    const btn = document.getElementById("btn-csp-solve");
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span class="btn-icon">⏳</span> Solving CSP Propagation...`;
    btn.disabled = true;

    try {
      const res = await fetch("/api/csp/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      state.cspResults = data;
      renderCspExecutionDashboard(data);
    } catch (err) {
      console.error("CSP match error:", err);
    } finally {
      btn.innerHTML = originalText;
      btn.disabled = false;
    }
  }

  function renderCspExecutionDashboard(data) {
    // 1. Update status banner
    document.getElementById("csp-exact-count").textContent = data.exact_matches_count;
    document.getElementById("csp-relaxed-count").textContent = data.relaxed_matches_count;
    document.getElementById("csp-exec-time").textContent = `${data.execution_time_ms} ms`;
    
    const statusEl = document.getElementById("csp-state-status");
    if (data.is_over_constrained) {
      statusEl.textContent = "Over-Constrained";
      statusEl.style.color = "var(--accent-amber)";
    } else {
      statusEl.textContent = "Satisfied";
      statusEl.style.color = "var(--accent-emerald)";
    }

    document.getElementById("exact-badge").textContent = data.exact_matches_count;
    document.getElementById("relaxed-badge").textContent = data.relaxed_matches_count;

    // 2. Render Forward Checking Trace Pipeline
    renderCspTrace(data.search_trace);

    // 3. Render Match Listings
    // Auto switch to relaxed if 0 exact
    if (data.exact_matches_count === 0 && data.relaxed_matches_count > 0) {
      document.getElementById("subtab-relaxed").click();
    } else {
      document.getElementById("subtab-exact").click();
    }
  }

  function renderCspTrace(traceSteps) {
    const container = document.getElementById("csp-trace-container");
    if (!container || !traceSteps) return;

    container.innerHTML = traceSteps.map(step => `
      <div class="trace-step-item">
        <div class="step-badge">${step.step}</div>
        <div class="step-content">
          <div class="step-title">
            <span>${step.phase}</span>
            <span class="step-constraint">[${step.constraint}]</span>
          </div>
          <div class="step-desc">${step.description}</div>
        </div>
        <div class="step-domain-change">
          <div class="domain-size">${step.domain_size} left</div>
          ${step.pruned > 0 ? `<div class="domain-pruned">-${step.pruned} pruned</div>` : ''}
        </div>
      </div>
    `).join("");
  }

  function renderCspMatchesList() {
    const container = document.getElementById("csp-matches-list");
    if (!container || !state.cspResults) return;

    const list = state.activeSubtab === "exact" 
      ? state.cspResults.exact_matches 
      : state.cspResults.relaxed_matches;

    if (!list || list.length === 0) {
      container.innerHTML = `
        <div class="placeholder-text" style="padding: 2rem; text-align: center;">
          ${state.activeSubtab === 'exact' 
            ? 'No exact matches found for these strict constraints (Over-constrained). Check the Relaxed Alternatives tab.' 
            : 'No relaxed candidates found within current tolerance thresholds.'}
        </div>
      `;
      return;
    }

    container.innerHTML = list.map(prop => {
      const relaxBadges = prop.relaxations && prop.relaxations.length > 0
        ? `<div class="relax-badge-strip">
             ${prop.relaxations.map(r => `<span class="relax-badge">⚡ ${r}</span>`).join("")}
           </div>`
        : '';

      const checklistHtml = prop.constraint_breakdown
        ? `<div class="constraint-checklist">
             ${prop.constraint_breakdown.map(c => `
               <span class="check-pill ${c.satisfied ? 'satisfied' : ''}">
                 ${c.satisfied ? '✓' : '✗'} ${c.name}: ${c.value}
               </span>
             `).join("")}
           </div>`
        : '';

      return `
        <div class="property-match-card" data-pid="${prop.property_id}">
          <div class="match-card-top">
            <div>
              <div class="match-title">${prop.title}</div>
              <div class="match-location">📍 ${prop.location} &bull; ${prop.distance_to_center_km} km to Center</div>
            </div>
            <div class="score-radial">
              <span class="score-val">${prop.satisfaction_score}%</span>
              <span class="score-label">CSP Score</span>
            </div>
          </div>

          <div class="match-specs-bar">
            <span>Price: <strong style="color: var(--accent-emerald);">$${prop.price.toLocaleString()}</strong></span>
            <span>Area: <strong>${prop.sqft.toLocaleString()} sqft</strong></span>
            <span>Rooms: <strong>${prop.bedrooms} Bed / ${prop.bathrooms} Bath</strong></span>
            <span>Age: <strong>${prop.age} yrs</strong></span>
          </div>

          ${relaxBadges}
          ${checklistHtml}
        </div>
      `;
    }).join("");
  }

  // -------------------------------------------------------------------------
  // Catalog Explorer & Market Stats
  // -------------------------------------------------------------------------
  async function loadDatasetStats() {
    try {
      const res = await fetch("/api/dataset/stats");
      const data = await res.json();
      state.datasetStats = data;

      document.getElementById("stat-total-props").textContent = data.total_properties;
      document.getElementById("stat-avg-price").textContent = `$${Math.round(data.overall_avg_price).toLocaleString()}`;
      document.getElementById("stat-min-price").textContent = `$${data.overall_min_price.toLocaleString()}`;
      document.getElementById("stat-max-price").textContent = `$${data.overall_max_price.toLocaleString()}`;

      renderMarketScatterChart(data.scatter_sample);
    } catch (err) {
      console.error("Failed to load dataset stats:", err);
    }
  }

  function renderMarketScatterChart(scatterData) {
    const canvas = document.getElementById("chart-market-scatter");
    if (!canvas || !scatterData) return;

    if (state.scatterChart) {
      state.scatterChart.destroy();
    }

    const points = scatterData.map(p => ({
      x: p.sqft,
      y: p.price,
      loc: p.location
    }));

    const ctx = canvas.getContext("2d");
    state.scatterChart = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [{
          label: "Properties (Area vs Price)",
          data: points,
          backgroundColor: "rgba(99, 102, 241, 0.75)",
          borderColor: "#818cf8",
          borderWidth: 1,
          pointRadius: 4.5,
          pointHoverRadius: 7
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const pt = ctx.raw;
                return ` ${pt.loc}: ${pt.x} sqft → $${pt.y.toLocaleString()}`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: "Living Area (sqft)", color: "#9ca3af" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#9ca3af" }
          },
          y: {
            title: { display: true, text: "Price ($)", color: "#9ca3af" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: {
              color: "#9ca3af",
              callback: (val) => `$${(val / 1000).toFixed(0)}k`
            }
          }
        }
      }
    });
  }

  async function fetchCatalog(page = 1) {
    const loc = document.getElementById("catalog-filter-location")?.value || "All";
    const beds = document.getElementById("catalog-filter-beds")?.value || "";
    const sortBy = document.getElementById("catalog-sort-by")?.value || "price";
    const sortOrder = document.getElementById("catalog-sort-order")?.value || "asc";

    let url = `/api/properties?page=${page}&limit=12&sort_by=${sortBy}&sort_order=${sortOrder}`;
    if (loc && loc !== "All") url += `&location=${encodeURIComponent(loc)}`;
    if (beds) url += `&min_beds=${beds}`;

    try {
      const res = await fetch(url);
      const data = await res.json();
      state.catalogPage = data.page;
      state.catalogTotalPages = data.total_pages;

      renderCatalogGrid(data.items);
      updatePaginationControls();
    } catch (err) {
      console.error("Error fetching catalog:", err);
    }
  }

  function renderCatalogGrid(items) {
    const grid = document.getElementById("catalog-grid");
    if (!grid) return;

    if (!items || items.length === 0) {
      grid.innerHTML = `<div class="placeholder-text" style="grid-column: 1/-1; text-align: center; padding: 3rem;">No properties found matching current criteria.</div>`;
      return;
    }

    grid.innerHTML = items.map(p => `
      <div class="catalog-item-card" data-pid="${p.property_id}">
        <div>
          <span class="catalog-loc-badge">${p.location}</span>
          <h4 style="font-size: 1.05rem; margin-top: 0.5rem; color: #fff;">${p.title}</h4>
          <div class="catalog-price">$${p.price.toLocaleString()}</div>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); border-top: 1px solid var(--border-subtle); padding-top: 0.6rem; margin-top: 0.6rem;">
          <div>📐 ${p.sqft.toLocaleString()} sqft &bull; 🛏️ ${p.bedrooms} Beds &bull; 🚿 ${p.bathrooms} Baths</div>
          <div style="margin-top: 0.25rem;">🚗 ${p.garage} Garage &bull; 📍 ${p.distance_to_center_km} km to Center</div>
        </div>
      </div>
    `).join("");

    // Click handler on catalog items to view details
    grid.querySelectorAll(".catalog-item-card").forEach(card => {
      card.addEventListener("click", () => {
        const pid = card.getAttribute("data-pid");
        const found = items.find(x => x.property_id === pid);
        if (found) showPropertyModal(found);
      });
    });
  }

  function updatePaginationControls() {
    const btnPrev = document.getElementById("btn-prev-page");
    const btnNext = document.getElementById("btn-next-page");
    const indicator = document.getElementById("page-indicator");

    if (btnPrev) btnPrev.disabled = state.catalogPage <= 1;
    if (btnNext) btnNext.disabled = state.catalogPage >= state.catalogTotalPages;
    if (indicator) indicator.textContent = `Page ${state.catalogPage} of ${state.catalogTotalPages}`;
  }

  function showPropertyModal(prop) {
    const modal = document.getElementById("property-modal");
    const content = document.getElementById("modal-content");
    if (!modal || !content) return;

    const amenitiesList = (prop.amenities || "").split(";").filter(Boolean);

    content.innerHTML = `
      <div style="margin-bottom: 1.25rem;">
        <span class="catalog-loc-badge">${prop.location}</span>
        <h2 style="font-family: var(--font-heading); color: #fff; margin: 0.4rem 0;">${prop.title}</h2>
        <div style="font-size: 1.75rem; font-weight: 700; color: var(--accent-emerald); font-family: var(--font-heading);">
          $${prop.price.toLocaleString()}
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; font-size: 0.9rem; margin-bottom: 1.25rem;">
        <div class="card" style="padding: 0.85rem; background: rgba(0,0,0,0.3);">
          <div style="color: var(--text-dim); font-size: 0.75rem;">SQUARE FOOTAGE</div>
          <div style="color: #fff; font-weight: 600;">${prop.sqft.toLocaleString()} sqft</div>
        </div>
        <div class="card" style="padding: 0.85rem; background: rgba(0,0,0,0.3);">
          <div style="color: var(--text-dim); font-size: 0.75rem;">BED / BATH</div>
          <div style="color: #fff; font-weight: 600;">${prop.bedrooms} Bed / ${prop.bathrooms} Bath</div>
        </div>
        <div class="card" style="padding: 0.85rem; background: rgba(0,0,0,0.3);">
          <div style="color: var(--text-dim); font-size: 0.75rem;">AGE & GARAGE</div>
          <div style="color: #fff; font-weight: 600;">${prop.age} years old / ${prop.garage} car(s)</div>
        </div>
        <div class="card" style="padding: 0.85rem; background: rgba(0,0,0,0.3);">
          <div style="color: var(--text-dim); font-size: 0.75rem;">COMMUTE DISTANCE</div>
          <div style="color: #fff; font-weight: 600;">${prop.distance_to_center_km} km to Downtown</div>
        </div>
      </div>

      <div>
        <h4 style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem;">AMENITIES</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">
          ${amenitiesList.map(a => `<span class="chip selected">${a.trim()}</span>`).join("")}
        </div>
      </div>
    `;

    modal.classList.add("active");
  }

  // -------------------------------------------------------------------------
  // Event Listeners
  // -------------------------------------------------------------------------
  function setupEventListeners() {
    // Prediction Form Submit
    const predForm = document.getElementById("prediction-form");
    if (predForm) {
      predForm.addEventListener("submit", (e) => {
        e.preventDefault();
        executePrediction();
      });
    }

    // CSP Form Submit
    const cspForm = document.getElementById("csp-form");
    if (cspForm) {
      cspForm.addEventListener("submit", executeCspMatch);
    }

    // Catalog Filter Button
    const btnFilter = document.getElementById("btn-apply-filters");
    if (btnFilter) {
      btnFilter.addEventListener("click", () => fetchCatalog(1));
    }

    // Pagination buttons
    const btnPrev = document.getElementById("btn-prev-page");
    const btnNext = document.getElementById("btn-next-page");
    if (btnPrev) {
      btnPrev.addEventListener("click", () => {
        if (state.catalogPage > 1) fetchCatalog(state.catalogPage - 1);
      });
    }
    if (btnNext) {
      btnNext.addEventListener("click", () => {
        if (state.catalogPage < state.catalogTotalPages) fetchCatalog(state.catalogPage + 1);
      });
    }

    // Modal Close
    const modalClose = document.getElementById("modal-close-btn");
    const modal = document.getElementById("property-modal");
    if (modalClose && modal) {
      modalClose.addEventListener("click", () => modal.classList.remove("active"));
      modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.remove("active");
      });
    }
  }

  // Run start
  init();
});
