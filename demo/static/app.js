const state = {
  summary: null,
  sessions: [],
  defaults: {},
};

const $ = (sel) => document.querySelector(sel);

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

function fmt(n, digits = 3) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

function buildFeatureForm(featureMeta, values) {
  const form = $("#feature-form");
  form.innerHTML = "";
  for (const meta of featureMeta) {
    const label = document.createElement("label");
    const help = meta.help || "";
    label.title = help;
    const strong = document.createElement("strong");
    strong.textContent = meta.name;
    label.appendChild(strong);

    const input = document.createElement("input");
    input.name = meta.name;
    input.dataset.feature = meta.name;
    if (meta.dtype === "bool") {
      input.type = "number";
      input.min = "0";
      input.max = "1";
      input.step = "1";
    } else {
      input.type = "number";
      input.step = "any";
    }
    const v = values[meta.name];
    input.value = v == null || Number.isNaN(v) ? "" : v;
    input.placeholder = "missing / NaN";
    label.appendChild(input);
    form.appendChild(label);
  }
}

function readFeatures() {
  const features = {};
  for (const input of document.querySelectorAll("#feature-form input")) {
    const name = input.dataset.feature;
    if (input.value === "") {
      features[name] = null;
    } else {
      features[name] = Number(input.value);
    }
  }
  return features;
}

function fillSessionSelect(sessions) {
  const select = $("#session-select");
  const current = select.value;
  select.innerHTML = '<option value="">Custom / blank form</option>';
  const normals = sessions.filter((s) => s.tag === "normal");
  const anomalies = sessions.filter((s) => s.tag === "anomaly");

  const addGroup = (label, items) => {
    const group = document.createElement("optgroup");
    group.label = label;
    for (const s of items) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.subtype || s.taxonomy_class || s.tag} · ${s.id.slice(0, 42)}`;
      group.appendChild(opt);
    }
    select.appendChild(group);
  };

  addGroup("Normal sessions", normals);
  addGroup("Anomalous sessions", anomalies);
  if ([...select.options].some((o) => o.value === current)) {
    select.value = current;
  }
}

function applySession(session) {
  const meta = $("#session-meta");
  if (!session) {
    meta.hidden = true;
    buildFeatureForm(state.summary.feature_meta, state.defaults);
    return;
  }
  meta.hidden = false;
  $("#meta-label").textContent = session.label_name;
  $("#meta-subtype").textContent = session.subtype || "—";
  $("#meta-taxonomy").textContent = session.taxonomy_class || "—";
  buildFeatureForm(state.summary.feature_meta, session.features);
}

function renderMetrics(summary) {
  const tbody = $("#metrics-table tbody");
  tbody.innerHTML = "";
  for (const row of summary.metrics.test) {
    const tr = document.createElement("tr");
    if (row.model === summary.best_model) tr.classList.add("best");
    tr.innerHTML = `
      <td>${row.model}${row.model === summary.best_model ? " ★" : ""}</td>
      <td class="mono">${fmt(row.threshold, 4)}</td>
      <td class="mono">${fmt(row.precision)}</td>
      <td class="mono">${fmt(row.recall)}</td>
      <td class="mono">${fmt(row.f1)}</td>
      <td class="mono">${fmt(row.pr_auc)}</td>
      <td class="mono">${fmt(row.roc_auc)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderHero(summary) {
  const rate = ((summary.n_anomaly / summary.n_sessions) * 100).toFixed(1) + "%";
  const values = [
    summary.n_sessions.toLocaleString(),
    String(summary.n_features),
    rate,
  ];
  $("#hero-stats").querySelectorAll("strong").forEach((el, i) => {
    el.textContent = values[i];
  });
  $("#citation").textContent = summary.citation || "";
}

function renderResult(payload) {
  $(".result-idle").hidden = true;
  $(".result-body").hidden = false;

  const verdict = $("#verdict");
  verdict.classList.toggle("anomaly", payload.prediction === 1);
  verdict.classList.toggle("normal", payload.prediction === 0);
  $("#pred-label").textContent = payload.prediction_label;
  $("#pred-prob").textContent = fmt(payload.anomaly_probability, 4);
  $("#pred-threshold").textContent = `(threshold ${fmt(payload.threshold, 4)})`;

  const pct = Math.max(0, Math.min(100, payload.anomaly_probability * 100));
  $("#meter-fill").style.width = `${pct}%`;
  $("#meter-threshold").style.left = `${payload.threshold * 100}%`;

  const list = $("#contrib-list");
  list.innerHTML = "";
  const maxAbs = Math.max(
    ...payload.contributions.map((c) => Math.abs(c.contribution)),
    1e-9
  );
  for (const c of payload.contributions) {
    const li = document.createElement("li");
    const width = (Math.abs(c.contribution) / maxAbs) * 100;
    li.innerHTML = `
      <span class="name">${c.feature}</span>
      <span class="val">${c.contribution >= 0 ? "+" : ""}${fmt(c.contribution, 4)} → ${c.direction}</span>
      <div class="bar-wrap"><div class="bar ${c.direction}" style="width:${width}%"></div></div>
    `;
    list.appendChild(li);
  }
  $("#contrib-note").textContent = payload.note || "";
}

async function runPredict() {
  const model = $("#model-select").value;
  const features = readFeatures();
  const btn = $("#btn-predict");
  btn.disabled = true;
  btn.textContent = "Classifying…";
  try {
    const payload = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, features }),
    });
    renderResult(payload);
  } catch (err) {
    alert(`Prediction failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Classify session";
  }
}

function pickRandomSession() {
  if (!state.sessions.length) return;
  const s = state.sessions[Math.floor(Math.random() * state.sessions.length)];
  $("#session-select").value = s.id;
  applySession(s);
  document.querySelector("#analyzer").scrollIntoView({ behavior: "smooth" });
  runPredict();
}

async function init() {
  const [summary, sessionPayload] = await Promise.all([
    api("/api/summary"),
    api("/api/sessions?limit=80"),
  ]);
  state.summary = summary;
  state.sessions = sessionPayload.sessions;
  state.defaults = summary.defaults || {};

  const modelSelect = $("#model-select");
  modelSelect.innerHTML = "";
  for (const name of summary.models) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name === summary.best_model ? `${name} (best)` : name;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = summary.best_model;

  fillSessionSelect(state.sessions);
  buildFeatureForm(summary.feature_meta, state.defaults);
  renderMetrics(summary);
  renderHero(summary);

  $("#session-select").addEventListener("change", (e) => {
    const id = e.target.value;
    const session = state.sessions.find((s) => s.id === id);
    applySession(session || null);
  });
  $("#btn-predict").addEventListener("click", runPredict);
  $("#btn-reset").addEventListener("click", () => {
    $("#session-select").value = "";
    applySession(null);
  });
  $("#btn-random").addEventListener("click", pickRandomSession);
}

init().catch((err) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div style="padding:1rem 4vw;background:#f8d7da;color:#842029;">
      Failed to load demo artifacts. Train models first:<br/>
      <code>python -m src.train_and_export</code><br/>
      then restart the server. (${err.message})
    </div>`
  );
});
