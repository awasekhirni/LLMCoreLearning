/* ============================================================
   AI Risk Periodic Table — Application v1.3 (fixed + enhanced)
   ============================================================ */

const FALLBACK_DATA = {
  meta: {
    version: "1.2.0",
    last_updated: "2026-10-17T10:00:00Z",
    thresholds: { safe: 25, low: 50, medium: 75, high: 90 },
    scoring_model: "score = (likelihood/5*100)*0.4 + (impact/5*100)*0.6"
  },
  risks: [
    {"id":"AI-R-001","symbol":"BS","name":"Bias","description":"Unfair or skewed outputs.","category":"Data","likelihood":3,"impact":4,"score":72,"trend":"up","mitigation_status":"in_progress","owner":"Data Governance","mitigations":["Bias audits","Fairness metrics","Dataset stratification"],"references":["NIST AI RMF MAP 1.1","ISO/IEC 42001 6.1.2"],"tags":["fairness","regulatory"],"last_reviewed":"2026-09-14"},
    {"id":"AI-R-002","symbol":"PV","name":"Privacy","description":"Sensitive data exposure.","category":"Data","likelihood":2,"impact":4,"score":64,"trend":"flat","mitigation_status":"open","owner":"Privacy Office","mitigations":["PII redaction","Data minimization","Consent tracking"],"references":["GDPR Art. 5","NIST Privacy Framework"],"tags":["privacy","compliance"],"last_reviewed":"2026-08-30"},
    {"id":"AI-R-003","symbol":"DQ","name":"Data Quality","description":"Inaccurate or incomplete data.","category":"Data","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"Data Engineering","mitigations":["Data profiling","Schema validation","DQ SLAs"],"references":["DAMA-DMBOK"],"tags":["quality","reliability"],"last_reviewed":"2026-09-20"},
    {"id":"AI-R-004","symbol":"DL","name":"Data Lineage","description":"Unclear data provenance.","category":"Data","likelihood":2,"impact":2,"score":40,"trend":"flat","mitigation_status":"open","owner":"Data Engineering","mitigations":["Lineage tooling","Column-level tagging"],"references":["ISO 8000-61"],"tags":["provenance","audit"],"last_reviewed":"2026-07-11"},
    {"id":"AI-R-005","symbol":"DR","name":"Data Drift","description":"Data changes over time.","category":"Data","likelihood":3,"impact":3,"score":60,"trend":"up","mitigation_status":"in_progress","owner":"ML Platform","mitigations":["Drift monitors","Automated retraining"],"references":["NIST AI RMF MEASURE 2.5"],"tags":["monitoring","mlops"],"last_reviewed":"2026-10-02"},
    {"id":"AI-R-006","symbol":"LE","name":"Data Leakage","description":"Sensitive data leaks out.","category":"Model","likelihood":4,"impact":5,"score":92,"trend":"up","mitigation_status":"open","owner":"Security","mitigations":["Egress controls","DLP","Model output filters"],"references":["OWASP LLM06","CIS Controls 3"],"tags":["exploit","security-critical"],"last_reviewed":"2026-10-10"},
    {"id":"AI-R-007","symbol":"CR","name":"Consent Risk","description":"Missing or invalid consent.","category":"Model","likelihood":2,"impact":3,"score":52,"trend":"flat","mitigation_status":"in_progress","owner":"Legal","mitigations":["Consent ledger","Opt-out flows"],"references":["GDPR Art. 7"],"tags":["legal","compliance"],"last_reviewed":"2026-08-15"},
    {"id":"AI-R-008","symbol":"SD","name":"Synthetic Data","description":"Fake data causes false confidence.","category":"Model","likelihood":2,"impact":2,"score":40,"trend":"down","mitigation_status":"mitigated","owner":"ML Platform","mitigations":["Synthetic data validation","Provenance tags"],"references":["NIST AI 100-1"],"tags":["validation"],"last_reviewed":"2026-06-05"},
    {"id":"AI-R-009","symbol":"ES","name":"Embedded Secrets","description":"Secrets hidden in data.","category":"Security","likelihood":3,"impact":4,"score":72,"trend":"flat","mitigation_status":"in_progress","owner":"AppSec","mitigations":["Secret scanners","Vault integration"],"references":["CIS Secrets Mgmt"],"tags":["credentials"],"last_reviewed":"2026-09-01"},
    {"id":"AI-R-010","symbol":"PI","name":"PII Exposure","description":"Personal data is exposed.","category":"Governance","likelihood":4,"impact":5,"score":92,"trend":"up","mitigation_status":"open","owner":"Privacy Office","mitigations":["PII classifiers","Redaction","Audit logs"],"references":["GDPR Art. 32","NIST SP 800-122"],"tags":["privacy","regulatory"],"last_reviewed":"2026-10-12"},
    {"id":"AI-R-011","symbol":"HL","name":"Hallucination","description":"Outputs are fabricated.","category":"Model","likelihood":5,"impact":3,"score":76,"trend":"up","mitigation_status":"in_progress","owner":"ML Research","mitigations":["RAG grounding","Citation enforcement"],"references":["NIST AI 600-1"],"tags":["quality","trust"],"last_reviewed":"2026-10-08"},
    {"id":"AI-R-012","symbol":"EX","name":"Explainability","description":"Decisions lack transparency.","category":"Model","likelihood":3,"impact":3,"score":60,"trend":"flat","mitigation_status":"in_progress","owner":"ML Research","mitigations":["SHAP/LIME","Decision logs"],"references":["EU AI Act Art. 13"],"tags":["transparency","compliance"],"last_reviewed":"2026-09-22"},
    {"id":"AI-R-013","symbol":"OF","name":"Overfitting","description":"Performs well on training only.","category":"Model","likelihood":2,"impact":2,"score":40,"trend":"flat","mitigation_status":"mitigated","owner":"ML Research","mitigations":["Cross-validation","Holdout sets"],"references":["NIST AI RMF MEASURE 2.3"],"tags":["quality"],"last_reviewed":"2026-05-18"},
    {"id":"AI-R-014","symbol":"UF","name":"Underfitting","description":"Fails to learn patterns.","category":"Model","likelihood":1,"impact":1,"score":20,"trend":"down","mitigation_status":"mitigated","owner":"ML Research","mitigations":["Model capacity review","Feature engineering"],"references":["NIST AI RMF MEASURE 2.3"],"tags":["quality"],"last_reviewed":"2026-04-10"},
    {"id":"AI-R-015","symbol":"RB","name":"Robustness","description":"Breaks under change.","category":"Model","likelihood":3,"impact":3,"score":60,"trend":"up","mitigation_status":"in_progress","owner":"ML Research","mitigations":["Adversarial testing","Chaos engineering"],"references":["NIST AI 100-2"],"tags":["reliability"],"last_reviewed":"2026-09-15"},
    {"id":"AI-R-016","symbol":"EV","name":"Evaluation Failure","description":"Poor evaluation methods.","category":"Model","likelihood":4,"impact":3,"score":68,"trend":"flat","mitigation_status":"in_progress","owner":"ML Platform","mitigations":["Golden datasets","Eval CI/CD"],"references":["NIST AI RMF MEASURE 1.1"],"tags":["quality"],"last_reviewed":"2026-08-20"},
    {"id":"AI-R-017","symbol":"VG","name":"Validation Gap","description":"Missing or weak validation.","category":"Model","likelihood":3,"impact":4,"score":72,"trend":"up","mitigation_status":"open","owner":"ML Platform","mitigations":["Independent validation","Red-team reviews"],"references":["NIST AI RMF MEASURE 1.3"],"tags":["assurance"],"last_reviewed":"2026-09-05"},
    {"id":"AI-R-018","symbol":"MD","name":"Model Drift","description":"Model degrades over time.","category":"Model","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"ML Platform","mitigations":["Performance monitors","Retraining pipelines"],"references":["NIST AI RMF MEASURE 2.5"],"tags":["monitoring","mlops"],"last_reviewed":"2026-10-04"},
    {"id":"AI-R-019","symbol":"PS","name":"Prompt Sensitivity","description":"Overly sensitive to prompts.","category":"Model","likelihood":3,"impact":2,"score":48,"trend":"flat","mitigation_status":"open","owner":"ML Research","mitigations":["Prompt regression tests","Input normalization"],"references":["OWASP LLM01"],"tags":["quality"],"last_reviewed":"2026-07-25"},
    {"id":"AI-R-020","symbol":"MB","name":"Metric Blindness","description":"Wrong metrics mislead.","category":"Model","likelihood":2,"impact":2,"score":40,"trend":"flat","mitigation_status":"in_progress","owner":"ML Research","mitigations":["Metric review board","Business-aligned KPIs"],"references":["NIST AI RMF MEASURE 2.1"],"tags":["measurement"],"last_reviewed":"2026-06-12"},
    {"id":"AI-R-021","symbol":"AH","name":"Autonomy Risk","description":"Too much autonomy.","category":"Agent","likelihood":3,"impact":4,"score":72,"trend":"up","mitigation_status":"in_progress","owner":"Agent Platform","mitigations":["Action approval gates","Scope limits"],"references":["NIST AI 600-1"],"tags":["agentic","control"],"last_reviewed":"2026-10-06"},
    {"id":"AI-R-022","symbol":"TM","name":"Tool Misuse","description":"Tools used incorrectly.","category":"Agent","likelihood":3,"impact":3,"score":60,"trend":"up","mitigation_status":"open","owner":"Agent Platform","mitigations":["Tool allowlists","Sandboxed exec"],"references":["OWASP LLM08"],"tags":["agentic","tooling"],"last_reviewed":"2026-09-10"},
    {"id":"AI-R-023","symbol":"UA","name":"Unintended Actions","description":"Actions not as intended.","category":"Agent","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"Agent Platform","mitigations":["Dry-run mode","Confirmation flows"],"references":["NIST AI RMF MANAGE 2.2"],"tags":["agentic","safety"],"last_reviewed":"2026-10-01"},
    {"id":"AI-R-024","symbol":"PE","name":"Permission Escalation","description":"Excessive privileges.","category":"Security","likelihood":4,"impact":5,"score":92,"trend":"up","mitigation_status":"open","owner":"AppSec","mitigations":["Least privilege","Scoped tokens","Runtime policy"],"references":["CIS Controls 4","OWASP LLM02"],"tags":["exploit","security-critical"],"last_reviewed":"2026-10-11"},
    {"id":"AI-R-025","symbol":"IN","name":"Prompt Injection","description":"Malicious prompt manipulation.","category":"Security","likelihood":5,"impact":5,"score":100,"trend":"up","mitigation_status":"open","owner":"AppSec","mitigations":["Input sanitization","Instruction hierarchy","Sandboxed execution"],"references":["OWASP LLM01","NIST AI 100-2"],"tags":["exploit","security-critical"],"last_reviewed":"2026-10-14"},
    {"id":"AI-R-026","symbol":"SM","name":"Social Manipulation","description":"Exploited by social tricks.","category":"Security","likelihood":3,"impact":4,"score":72,"trend":"up","mitigation_status":"in_progress","owner":"Security","mitigations":["Content moderation","User education"],"references":["NIST AI RMF MANAGE 2.1"],"tags":["abuse"],"last_reviewed":"2026-09-28"},
    {"id":"AI-R-027","symbol":"DG","name":"Delegation Risk","description":"Risky task delegation.","category":"Agent","likelihood":2,"impact":3,"score":52,"trend":"flat","mitigation_status":"in_progress","owner":"Agent Platform","mitigations":["Delegation policies","Human-in-loop"],"references":["NIST AI RMF GOVERN 1.2"],"tags":["agentic"],"last_reviewed":"2026-08-08"},
    {"id":"AI-R-028","symbol":"MR","name":"Memory Risk","description":"Sensitive memory exposure.","category":"Agent","likelihood":3,"impact":3,"score":60,"trend":"up","mitigation_status":"open","owner":"Agent Platform","mitigations":["Memory TTLs","PII scrubbing"],"references":["GDPR Art. 5"],"tags":["privacy","agentic"],"last_reviewed":"2026-09-19"},
    {"id":"AI-R-029","symbol":"LF","name":"Loop Failure","description":"Agents repeat bad actions.","category":"Agent","likelihood":2,"impact":3,"score":52,"trend":"flat","mitigation_status":"in_progress","owner":"Agent Platform","mitigations":["Step budgets","Loop detection"],"references":["NIST AI 600-1"],"tags":["agentic","reliability"],"last_reviewed":"2026-08-22"},
    {"id":"AI-R-030","symbol":"RG","name":"Rollback Gap","description":"Harmful actions cannot be safely reverted.","category":"Agent","likelihood":3,"impact":4,"score":72,"trend":"up","mitigation_status":"open","owner":"Agent Platform","mitigations":["Transactional actions","Snapshots"],"references":["NIST AI RMF MANAGE 3.1"],"tags":["agentic","recovery"],"last_reviewed":"2026-09-30"},
    {"id":"AI-R-031","symbol":"VP","name":"Vulnerability","description":"Exploitable weaknesses.","category":"Security","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"AppSec","mitigations":["SAST/DAST","Dependency scanning"],"references":["CIS Controls 16"],"tags":["exploit"],"last_reviewed":"2026-10-05"},
    {"id":"AI-R-032","symbol":"CT","name":"Cyber Threats","description":"External attacks and intrusions.","category":"Security","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"Security","mitigations":["WAF","Rate limiting","Threat intel"],"references":["CIS Controls 13"],"tags":["exploit"],"last_reviewed":"2026-10-03"},
    {"id":"AI-R-033","symbol":"AB","name":"API Abuse","description":"APIs misused or overused.","category":"Security","likelihood":3,"impact":3,"score":60,"trend":"flat","mitigation_status":"in_progress","owner":"AppSec","mitigations":["Quotas","Anomaly detection"],"references":["OWASP API Top 10"],"tags":["abuse"],"last_reviewed":"2026-08-28"},
    {"id":"AI-R-034","symbol":"TT","name":"Token Theft","description":"Keys or tokens stolen.","category":"Security","likelihood":3,"impact":4,"score":72,"trend":"flat","mitigation_status":"in_progress","owner":"AppSec","mitigations":["Short-lived tokens","mTLS","Rotation"],"references":["CIS Controls 6"],"tags":["credentials"],"last_reviewed":"2026-09-12"},
    {"id":"AI-R-035","symbol":"MU","name":"Malicious Use","description":"AI used for harmful intent.","category":"Security","likelihood":4,"impact":5,"score":92,"trend":"up","mitigation_status":"open","owner":"Trust & Safety","mitigations":["Abuse detection","Acceptable use policy"],"references":["NIST AI 100-2"],"tags":["abuse","policy"],"last_reviewed":"2026-10-13"},
    {"id":"AI-R-036","symbol":"DE","name":"Data Exfiltration","description":"Data stolen or exfiltrated.","category":"Security","likelihood":4,"impact":5,"score":92,"trend":"up","mitigation_status":"open","owner":"Security","mitigations":["Egress filtering","DLP","SIEM alerts"],"references":["CIS Controls 13","OWASP LLM06"],"tags":["exploit","security-critical"],"last_reviewed":"2026-10-14"},
    {"id":"AI-R-037","symbol":"DP","name":"Deployment Risk","description":"Unsafe or unstable deployment.","category":"Security","likelihood":3,"impact":3,"score":60,"trend":"flat","mitigation_status":"in_progress","owner":"Platform Eng","mitigations":["Canary deploys","Blue/green","Rollback SLAs"],"references":["Google SRE"],"tags":["ops"],"last_reviewed":"2026-09-06"},
    {"id":"AI-R-038","symbol":"IG","name":"Integration Risk","description":"Failures in integrations.","category":"Security","likelihood":2,"impact":3,"score":52,"trend":"flat","mitigation_status":"in_progress","owner":"Platform Eng","mitigations":["Contract tests","Circuit breakers"],"references":["NIST AI RMF MANAGE 2.3"],"tags":["ops"],"last_reviewed":"2026-08-18"},
    {"id":"AI-R-039","symbol":"LT","name":"Latency Risk","description":"Slow responses impact users.","category":"Security","likelihood":2,"impact":2,"score":40,"trend":"down","mitigation_status":"mitigated","owner":"Platform Eng","mitigations":["SLOs","Caching","Async processing"],"references":["Google SRE"],"tags":["ops"],"last_reviewed":"2026-07-01"},
    {"id":"AI-R-040","symbol":"CO","name":"Cost Overrun","description":"Costs exceed expectations.","category":"Security","likelihood":3,"impact":4,"score":72,"trend":"up","mitigation_status":"in_progress","owner":"FinOps","mitigations":["Budgets & alerts","Model routing","Caching"],"references":["FinOps Foundation"],"tags":["cost"],"last_reviewed":"2026-09-25"},
    {"id":"AI-R-041","symbol":"MG","name":"Monitoring Gap","description":"Issues go undetected.","category":"Governance","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"ML Platform","mitigations":["Observability stack","Alerting SLOs"],"references":["NIST AI RMF MEASURE 2.4"],"tags":["monitoring"],"last_reviewed":"2026-10-07"},
    {"id":"AI-R-042","symbol":"DO","name":"Decision Ownership","description":"Unclear who owns decisions.","category":"Governance","likelihood":3,"impact":3,"score":60,"trend":"flat","mitigation_status":"in_progress","owner":"AI Governance","mitigations":["RACI matrices","Sign-off workflows"],"references":["NIST AI RMF GOVERN 1.2"],"tags":["governance"],"last_reviewed":"2026-08-14"},
    {"id":"AI-R-043","symbol":"HO","name":"Human Oversight","description":"Insufficient human supervision.","category":"Governance","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"AI Governance","mitigations":["HITL workflows","Escalation paths"],"references":["EU AI Act Art. 14"],"tags":["oversight"],"last_reviewed":"2026-09-21"},
    {"id":"AI-R-044","symbol":"TR","name":"Traceability","description":"Actions are not traceable.","category":"Governance","likelihood":3,"impact":3,"score":60,"trend":"flat","mitigation_status":"in_progress","owner":"AI Governance","mitigations":["Audit logs","Correlation IDs"],"references":["ISO/IEC 42001"],"tags":["audit"],"last_reviewed":"2026-09-02"},
    {"id":"AI-R-045","symbol":"CP","name":"Compliance Risk","description":"Violates laws or standards.","category":"Governance","likelihood":4,"impact":5,"score":92,"trend":"up","mitigation_status":"open","owner":"Legal","mitigations":["Control mapping","Compliance monitoring"],"references":["EU AI Act","ISO/IEC 42001"],"tags":["compliance","regulatory"],"last_reviewed":"2026-10-09"},
    {"id":"AI-R-046","symbol":"AG","name":"Audit Gap","description":"Incomplete audit records.","category":"Governance","likelihood":3,"impact":4,"score":72,"trend":"flat","mitigation_status":"in_progress","owner":"Internal Audit","mitigations":["Immutable logs","Audit playbooks"],"references":["SOC 2","ISO 27001"],"tags":["audit"],"last_reviewed":"2026-09-16"},
    {"id":"AI-R-047","symbol":"LG","name":"Legal Risk","description":"Legal exposure or liability.","category":"Governance","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"Legal","mitigations":["Contract review","Disclaimers"],"references":["EU AI Act"],"tags":["legal"],"last_reviewed":"2026-09-26"},
    {"id":"AI-R-048","symbol":"IP","name":"IP & Copyright","description":"Rights and ownership issues.","category":"Governance","likelihood":3,"impact":3,"score":60,"trend":"flat","mitigation_status":"in_progress","owner":"Legal","mitigations":["License scanning","Output attribution"],"references":["DMCA","EU Copyright Directive"],"tags":["legal","ip"],"last_reviewed":"2026-08-05"},
    {"id":"AI-R-049","symbol":"ET","name":"Ethics Risk","description":"Unethical or harmful impact.","category":"Governance","likelihood":4,"impact":4,"score":80,"trend":"up","mitigation_status":"in_progress","owner":"Ethics Board","mitigations":["Ethics review","Impact assessments"],"references":["OECD AI Principles"],"tags":["ethics"],"last_reviewed":"2026-10-02"},
    {"id":"AI-R-050","symbol":"RR","name":"Regulatory Risk","description":"Violates legal requirements.","category":"Governance","likelihood":4,"impact":5,"score":92,"trend":"up","mitigation_status":"open","owner":"Legal","mitigations":["Regulatory watch","Control testing"],"references":["EU AI Act","NIST AI RMF"],"tags":["regulatory"],"last_reviewed":"2026-10-12"}
  ]
};

/* ---------- State ---------- */
const state = {
  data: null,
  filters: { category: 'all', severity: 'all', owner: 'all', status: 'all', search: '' },
  sort: 'score-desc',
  view: 'grid'
};

let previouslyFocused = null;
let searchTimer = null;

/* ---------- DOM Elements ---------- */
const el = {
  grid:        document.getElementById('grid-view'),
  matrix:      document.getElementById('matrix-view'),
  table:       document.getElementById('periodic-table'),
  matrixSvg:   document.getElementById('risk-matrix'),
  drawer:      document.getElementById('drawer'),
  drawerPanel: document.getElementById('drawer-panel'),
  drawerContent: document.getElementById('drawer-content'),
  drawerClose: document.getElementById('drawer-close'),
  drawerBackdrop: document.getElementById('drawer-backdrop'),
  statusLine:  document.getElementById('status-line'),
  filterSeverity: document.getElementById('filter-severity'),
  filterOwner:  document.getElementById('filter-owner'),
  filterStatus: document.getElementById('filter-status'),
  sortBy:       document.getElementById('sort-by'),
  resetBtn:     document.getElementById('reset-filters'),
  chips:        document.querySelectorAll('.chip'),
  viewBtns:     document.querySelectorAll('.view-btn'),

  searchInput:  document.getElementById('search-input'),
  searchKbd:    document.getElementById('search-kbd'),
  searchClear:  document.getElementById('search-clear'),
  themeToggle:  document.getElementById('theme-toggle'),
  mobileToggle: document.getElementById('mobile-menu-toggle'),
  mobileNav:    document.getElementById('mobile-nav'),
  emptyState:   document.getElementById('empty-state'),
  emptyReset:   document.getElementById('empty-reset'),
  kpiTotal:     document.getElementById('kpi-total'),
  kpiShown:     document.getElementById('kpi-shown'),
  kpiCritical:  document.getElementById('kpi-critical'),
  kpiHigh:      document.getElementById('kpi-high'),
  kpiAvg:       document.getElementById('kpi-avg'),
  kpiMitigated: document.getElementById('kpi-mitigated'),
  kpiMitigatedPct: document.getElementById('kpi-mitigated-pct'),
  distBar:      document.getElementById('dist-bar'),
  footerYear:   document.getElementById('footer-year'),
  footerVersion:document.getElementById('footer-version'),
  footerUpdated:document.getElementById('footer-updated'),
  footerScoring:document.getElementById('footer-scoring'),
  exportBtns:   document.querySelectorAll('[data-export]')
};

/* ---------- Helpers ---------- */
function severityTier(score, t) {
  if (score > t.high)     return 'critical';
  if (score > t.medium)   return 'high';
  if (score > t.low)      return 'medium';
  if (score > t.safe)     return 'low';
  return 'safe';
}
const trendArrow = t => ({ up: '↗', down: '↘', flat: '→' }[t] || '→');
const trendLabel = t => ({ up: 'Increasing', down: 'Decreasing', flat: 'Stable' }[t] || 'Stable');
const statusLabel = s => ({ open: 'Open', in_progress: 'In progress', mitigated: 'Mitigated' }[s] || s);

function escapeHTML(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function relativeTime(iso) {
  const then = new Date(iso).getTime();
  if (isNaN(then)) return '—';
  const diff = Math.max(0, Date.now() - then);
  const m = Math.floor(diff / 60000);
  if (m < 1)   return 'just now';
  if (m < 60)  return `${m} min ago`;
  const h = Math.floor(m / 60);
  if (h < 24)  return `${h} h ago`;
  const d = Math.floor(h / 24);
  if (d < 30)  return `${d} d ago`;
  return new Date(iso).toLocaleDateString();
}

/* ---------- Data loading ---------- */
async function loadData() {
  try {
    const res = await fetch('data.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    if (!json.risks || !Array.isArray(json.risks)) throw new Error('Invalid data shape');
    return json;
  } catch (err) {
    console.warn('Could not load data.json (using fallback). Reason:', err.message);
    return FALLBACK_DATA;
  }
}

/* ---------- URL state ---------- */
function readStateFromURL() {
  const hash = location.hash.replace(/^#/, '');
  if (!hash || !hash.includes('=')) return;
  const p = new URLSearchParams(hash);
  if (p.get('cat'))    state.filters.category = p.get('cat');
  if (p.get('sev'))    state.filters.severity = p.get('sev');
  if (p.get('owner'))  state.filters.owner    = p.get('owner');
  if (p.get('status')) state.filters.status   = p.get('status');
  if (p.get('q'))      state.filters.search   = p.get('q');
  if (p.get('sort'))   state.sort             = p.get('sort');
  if (p.get('view'))   state.view             = p.get('view');
}

function syncStateToURL() {
  const p = new URLSearchParams();
  if (state.filters.category !== 'all') p.set('cat', state.filters.category);
  if (state.filters.severity !== 'all') p.set('sev', state.filters.severity);
  if (state.filters.owner    !== 'all') p.set('owner', state.filters.owner);
  if (state.filters.status   !== 'all') p.set('status', state.filters.status);
  if (state.filters.search) p.set('q', state.filters.search);
  if (state.sort !== 'score-desc') p.set('sort', state.sort);
  if (state.view !== 'grid') p.set('view', state.view);
  const hash = p.toString();
  try {
    history.replaceState(null, '', hash ? `#${hash}` : location.pathname + location.search);
  } catch (_) { /* ignore */ }
}

function syncControlsFromState() {
  el.filterSeverity.value = state.filters.severity;
  el.filterStatus.value   = state.filters.status;
  if ([...el.filterOwner.options].some(o => o.value === state.filters.owner)) {
    el.filterOwner.value = state.filters.owner;
  }
  el.sortBy.value = state.sort;
  el.searchInput.value = state.filters.search;
  el.chips.forEach(c => c.classList.toggle('active', c.dataset.category === state.filters.category));
  el.viewBtns.forEach(b => {
    const on = b.dataset.view === state.view;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', String(on));
  });
  el.grid.hidden = state.view !== 'grid';
  el.matrix.hidden = state.view !== 'matrix';
  updateSearchIndicators();
}

/* ---------- Filtering / sorting ---------- */
function applyFiltersAndSort() {
  const { category, severity, owner, status, search } = state.filters;
  const thresholds = state.data.meta.thresholds;

  let list = state.data.risks.filter(r => {
    if (category !== 'all' && r.category !== category) return false;
    if (owner !== 'all' && r.owner !== owner) return false;
    if (status !== 'all' && r.mitigation_status !== status) return false;
    if (severity !== 'all' && severityTier(r.score, thresholds) !== severity) return false;
    return true;
  });

  const q = (search || '').trim().toLowerCase();
  if (q) {
    list = list.filter(r => {
      const haystack = [r.symbol, r.name, r.description, r.category, r.owner,
        ...(r.tags || []), ...(r.mitigations || []), ...(r.references || [])].join(' ').toLowerCase();
      return haystack.includes(q);
    });
  }

  const sorters = {
    'score-desc':      (a, b) => b.score - a.score,
    'score-asc':       (a, b) => a.score - b.score,
    'likelihood-desc': (a, b) => b.likelihood - a.likelihood || b.impact - a.impact,
    'impact-desc':     (a, b) => b.impact - a.impact || b.likelihood - a.likelihood,
    'name-asc':        (a, b) => a.name.localeCompare(b.name),
  };
  list.sort(sorters[state.sort] || sorters['score-desc']);
  return list;
}

/* ---------- KPIs ---------- */
function renderKPIs(shown, all) {
  const t = state.data.meta.thresholds;
  const tierOf = s => severityTier(s, t);

  const criticalCount = shown.filter(r => tierOf(r.score) === 'critical').length;
  const highCount     = shown.filter(r => tierOf(r.score) === 'high').length;
  const avg           = shown.length ? Math.round(shown.reduce((s, r) => s + r.score, 0) / shown.length) : 0;
  const mitigated     = shown.filter(r => r.mitigation_status === 'mitigated').length;
  const mitigatedPct  = shown.length ? Math.round((mitigated / shown.length) * 100) : 0;

  el.kpiTotal.textContent     = all.length;
  el.kpiShown.textContent     = `${shown.length} shown`;
  el.kpiCritical.textContent  = criticalCount;
  el.kpiHigh.textContent      = highCount;
  el.kpiAvg.textContent       = shown.length ? avg : '—';
  el.kpiMitigated.textContent = mitigated;
  el.kpiMitigatedPct.textContent = `${mitigatedPct}% of shown`;

  const tiers = ['safe', 'low', 'medium', 'high', 'critical'];
  const counts = Object.fromEntries(tiers.map(k => [k, 0]));
  shown.forEach(r => { counts[tierOf(r.score)] += 1; });
  const total = shown.length || 1;

  el.distBar.innerHTML = tiers.map(k => {
    const w = (counts[k] / total) * 100;
    return w > 0 ? `<div class="dist-seg sev-${k}" style="width:${w}%" title="${counts[k]} ${k}"></div>` : '';
  }).join('');
  el.distBar.setAttribute('aria-label', tiers.map(k => `${counts[k]} ${k}`).join(', '));
}

/* ---------- Views ---------- */
function renderGrid(list) {
  const thresholds = state.data.meta.thresholds;
  const frag = document.createDocumentFragment();

  list.forEach(risk => {
    const tier = severityTier(risk.score, thresholds);
    const card = document.createElement('button');
    card.type = 'button';
    card.className = `element category-${risk.category} sev-${tier}`;
    card.setAttribute('aria-label',
      `${risk.name}, ${risk.category}. Risk score ${risk.score}. Severity ${tier}. Press Enter for details.`);
    card.dataset.riskId = risk.id;

    card.innerHTML = `
      <div class="element-header">
        <span class="symbol">${risk.symbol}</span>
        <span class="name">${escapeHTML(risk.name)}</span>
      </div>
      <div class="element-description">${escapeHTML(risk.description)}</div>
      <div class="risk-capsule" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${risk.score}">
        <div class="capsule-fill" style="width:${risk.score}%"></div>
      </div>
      <div class="capsule-meta">
        <span class="score">
          ${risk.score}
          <span class="trend trend-${risk.trend}" aria-hidden="true">${trendArrow(risk.trend)}</span>
        </span>
        <span class="status">${statusLabel(risk.mitigation_status)}</span>
      </div>
    `;
    card.addEventListener('click', () => openDrawer(risk, card));
    frag.appendChild(card);
  });

  el.table.innerHTML = '';
  el.table.appendChild(frag);
}

function renderMatrix(list) {
  const thresholds = state.data.meta.thresholds;
  const quadrants = `
    <div class="quadrant q-tr">Act Now</div>
    <div class="quadrant q-tl">Plan</div>
    <div class="quadrant q-br">Monitor</div>
    <div class="quadrant q-bl">Accept</div>
    <div class="matrix-axis-x">Likelihood →</div>
    <div class="matrix-axis-y">Impact →</div>
  `;

  const dots = list.map(risk => {
    const tier = severityTier(risk.score, thresholds);
    const size = 14 + (risk.score / 100) * 22;
    const fontSize = Math.max(8, size * 0.34);
    const left = (risk.likelihood - 1) * 20 + 10;
    const bottom = (risk.impact - 1) * 20 + 10;
    return `
      <button class="matrix-dot sev-${tier}"
        style="left:${left}%;bottom:${bottom}%;width:${size}px;height:${size}px;font-size:${fontSize}px;"
        data-risk-id="${risk.id}"
        aria-label="${escapeHTML(risk.name)}: score ${risk.score}"
        title="${escapeHTML(risk.name)} — L${risk.likelihood}/I${risk.impact} (${risk.score})">
        ${risk.symbol}
      </button>`;
  }).join('');

  el.matrixSvg.innerHTML = quadrants + dots;

  el.matrixSvg.querySelectorAll('.matrix-dot').forEach(dot => {
    const risk = state.data.risks.find(r => r.id === dot.dataset.riskId);
    if (risk) dot.addEventListener('click', () => openDrawer(risk, dot));
  });
}

function updateStatusLine(shown, total, list) {
  const thresholds = state.data.meta.thresholds;
  const critical = list.filter(r => severityTier(r.score, thresholds) === 'critical').length;
  const high     = list.filter(r => severityTier(r.score, thresholds) === 'high').length;
  el.statusLine.innerHTML = `
    Showing <strong>${shown}</strong> of <strong>${total}</strong> risks
    ${critical ? `<span class="pill critical">${critical} critical</span>` : ''}
    ${high ? `<span class="pill high">${high} high</span>` : ''}
  `;
}

function renderFooterMeta() {
  const { version, last_updated, scoring_model } = state.data.meta;
  el.footerYear.textContent = new Date().getFullYear();
  el.footerVersion.textContent = `v${version}`;
  el.footerScoring.textContent = scoring_model;
  el.footerUpdated.textContent = `Last updated ${relativeTime(last_updated)}`;
}

function populateOwnerFilter() {
  const owners = [...new Set(state.data.risks.map(r => r.owner))].sort();
  const current = el.filterOwner.value;
  el.filterOwner.innerHTML = '<option value="all">All</option>' +
    owners.map(o => `<option value="${escapeHTML(o)}">${escapeHTML(o)}</option>`).join('');
  if (owners.includes(current)) el.filterOwner.value = current;
}

/* ---------- Main render ---------- */
function render() {
  const list = applyFiltersAndSort();
  updateStatusLine(list.length, state.data.risks.length, list);
  renderKPIs(list, state.data.risks);

  const isEmpty = list.length === 0;
  el.emptyState.hidden = !isEmpty;

  if (state.view === 'grid') {
    if (!isEmpty) renderGrid(list);
    el.grid.hidden = isEmpty;
    el.matrix.hidden = true;
  } else {
    if (!isEmpty) renderMatrix(list);
    el.matrix.hidden = isEmpty;
    el.grid.hidden = true;
  }

  syncStateToURL();
}

/* ---------- Drawer ---------- */
function openDrawer(risk, triggerEl) {
  previouslyFocused = triggerEl || document.activeElement;
  const thresholds = state.data.meta.thresholds;
  const tier = severityTier(risk.score, thresholds);

  el.drawerContent.innerHTML = `
    <div class="drawer-head">
      <span class="drawer-symbol cat-${risk.category}">${risk.symbol}</span>
      <div>
        <h2 id="drawer-title">${escapeHTML(risk.name)}</h2>
        <span class="drawer-cat">${risk.category} Risk · ${risk.id}</span>
      </div>
    </div>
    <div class="drawer-score sev-${tier}">
      <span class="drawer-score-value">${risk.score}</span>
      <span class="drawer-score-label">Risk Score · ${tier}</span>
    </div>
    <div class="drawer-capsule">
      <div class="capsule-fill" style="width:${risk.score}%;background:var(--sev-${tier});"></div>
    </div>
    <dl class="drawer-meta">
      <div><dt>Likelihood</dt><dd>${risk.likelihood} / 5</dd></div>
      <div><dt>Impact</dt><dd>${risk.impact} / 5</dd></div>
      <div><dt>Trend</dt><dd><span class="trend trend-${risk.trend}">${trendArrow(risk.trend)}</span> ${trendLabel(risk.trend)}</dd></div>
      <div><dt>Status</dt><dd>${statusLabel(risk.mitigation_status)}</dd></div>
      <div><dt>Owner</dt><dd>${escapeHTML(risk.owner)}</dd></div>
      <div><dt>Last reviewed</dt><dd>${escapeHTML(risk.last_reviewed || '—')}</dd></div>
    </dl>
    <div class="drawer-section">
      <h3>Description</h3>
      <p>${escapeHTML(risk.description)}</p>
    </div>
    <div class="drawer-section">
      <h3>Mitigations</h3>
      <ul>${(risk.mitigations || []).map(m => `<li>${escapeHTML(m)}</li>`).join('') || '<li>—</li>'}</ul>
    </div>
    <div class="drawer-section">
      <h3>References</h3>
      <ul>${(risk.references || []).map(r => `<li>${escapeHTML(r)}</li>`).join('') || '<li>—</li>'}</ul>
    </div>
    <div class="drawer-section">
      <h3>Tags</h3>
      <div class="tag-list">
        ${(risk.tags || []).map(t => `<span class="tag">${escapeHTML(t)}</span>`).join('') || '<span class="tag">—</span>'}
      </div>
    </div>
  `;

  setBackgroundInert(true);
  el.drawer.classList.add('open');
  setTimeout(() => el.drawerClose.focus(), 50);
}

function closeDrawer() {
  if (!el.drawer.classList.contains('open')) return;
  el.drawer.classList.remove('open');
  setBackgroundInert(false);
  if (previouslyFocused && typeof previouslyFocused.focus === 'function' &&
      document.contains(previouslyFocused)) {
    previouslyFocused.focus();
  }
  previouslyFocused = null;
}

function setBackgroundInert(inert) {
  const targets = document.querySelectorAll('.site-header, main, .site-footer');
  targets.forEach(t => {
    if (inert) t.setAttribute('inert', '');
    else t.removeAttribute('inert');
  });
}

function trapFocus(e) {
  if (e.key !== 'Tab') return;
  const focusables = el.drawerPanel.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
  if (!focusables.length) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

/* ---------- Theme ---------- */
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('ai-risk-theme', theme); } catch (_) {}
}
function initTheme() {
  let theme;
  try { theme = localStorage.getItem('ai-risk-theme'); } catch (_) {}
  if (!theme) theme = window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  applyTheme(theme);
}

/* ---------- Search indicators ---------- */
function updateSearchIndicators() {
  const hasValue = el.searchInput.value.length > 0;
  el.searchClear.hidden = !hasValue;
  if (el.searchKbd) el.searchKbd.hidden = hasValue;
}

/* ---------- Export ---------- */
function downloadBlob(filename, mime, content) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportJSON() {
  const list = applyFiltersAndSort();
  downloadBlob(
    `ai-risks-${new Date().toISOString().slice(0, 10)}.json`,
    'application/json',
    JSON.stringify({ meta: state.data.meta, risks: list }, null, 2)
  );
}
function exportCSV() {
  const list = applyFiltersAndSort();
  const cols = ['id','symbol','name','category','score','likelihood','impact','trend','mitigation_status','owner','last_reviewed'];
  const esc = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const rows = [cols.join(',')];
  list.forEach(r => rows.push(cols.map(c => esc(r[c])).join(',')));
  downloadBlob(`ai-risks-${new Date().toISOString().slice(0, 10)}.csv`, 'text/csv', rows.join('\n'));
}

/* ---------- Keyboard grid navigation ---------- */
function handleGridKeydown(e) {
  const card = e.target.closest('.element');
  if (!card) return;
  const cards = Array.from(el.table.querySelectorAll('.element'));
  if (!cards.length) return;
  const idx = cards.indexOf(card);
  if (idx < 0) return;

  const colCount = getComputedStyle(el.table).gridTemplateColumns.split(' ').filter(Boolean).length || 1;
  let next = -1;
  switch (e.key) {
    case 'ArrowRight': next = idx + 1; break;
    case 'ArrowLeft':  next = idx - 1; break;
    case 'ArrowDown':  next = idx + colCount; break;
    case 'ArrowUp':    next = idx - colCount; break;
    case 'Home':       next = 0; break;
    case 'End':        next = cards.length - 1; break;
    default: return;
  }
  if (next >= 0 && next < cards.length) {
    e.preventDefault();
    cards[next].focus();
  }
}

/* ---------- Event wiring ---------- */
function wireEvents() {
  // View toggle
  el.viewBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view;
      if (view === state.view) return;
      state.view = view;
      el.viewBtns.forEach(b => {
        const on = b.dataset.view === view;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', String(on));
      });
      render();
    });
  });

  // Category chips
  el.chips.forEach(chip => {
    chip.addEventListener('click', () => {
      state.filters.category = chip.dataset.category;
      el.chips.forEach(c => c.classList.toggle('active', c === chip));
      render();
    });
  });

  // Selects
  el.filterSeverity.addEventListener('change', e => { state.filters.severity = e.target.value; render(); });
  el.filterOwner.addEventListener('change',    e => { state.filters.owner    = e.target.value; render(); });
  el.filterStatus.addEventListener('change',   e => { state.filters.status   = e.target.value; render(); });
  el.sortBy.addEventListener('change',         e => { state.sort             = e.target.value; render(); });

  // Reset
  const resetAll = () => {
    state.filters = { category: 'all', severity: 'all', owner: 'all', status: 'all', search: '' };
    state.sort = 'score-desc';
    el.filterSeverity.value = 'all';
    el.filterOwner.value = 'all';
    el.filterStatus.value = 'all';
    el.sortBy.value = 'score-desc';
    el.searchInput.value = '';
    el.chips.forEach(c => c.classList.toggle('active', c.dataset.category === 'all'));
    updateSearchIndicators();
    render();
  };
  el.resetBtn.addEventListener('click', resetAll);
  el.emptyReset.addEventListener('click', resetAll);

  // Search (debounced)
  el.searchInput.addEventListener('input', e => {
    updateSearchIndicators();
    clearTimeout(searchTimer);
    const val = e.target.value;
    searchTimer = setTimeout(() => {
      state.filters.search = val;
      render();
    }, 140);
  });
  el.searchClear.addEventListener('click', () => {
    el.searchInput.value = '';
    state.filters.search = '';
    updateSearchIndicators();
    el.searchInput.focus();
    render();
  });

  // Keyboard shortcuts / drawer
  document.addEventListener('keydown', e => {
    const tag = document.activeElement?.tagName;
    if (e.key === '/' && !['INPUT','TEXTAREA','SELECT'].includes(tag)) {
      e.preventDefault();
      el.searchInput.focus();
      el.searchInput.select();
    }
    if (e.key === 'Escape') closeDrawer();
    if (el.drawer.classList.contains('open')) trapFocus(e);
  });

  // Keyboard navigation inside the grid
  el.table.addEventListener('keydown', handleGridKeydown);

  // Theme
  el.themeToggle.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    applyTheme(cur === 'dark' ? 'light' : 'dark');
  });

  // Mobile nav
  el.mobileToggle.addEventListener('click', () => {
    const wasHidden = el.mobileNav.hidden;
    el.mobileNav.hidden = !wasHidden;
    el.mobileToggle.setAttribute('aria-expanded', String(wasHidden));
    el.mobileToggle.setAttribute('aria-label', wasHidden ? 'Close menu' : 'Open menu');
  });

  // Export menu
  document.querySelectorAll('[data-menu] > .icon-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const m = btn.parentElement;
      const open = m.classList.toggle('open');
      btn.setAttribute('aria-expanded', String(open));
    });
  });
  document.addEventListener('click', () => {
    document.querySelectorAll('.menu.open').forEach(m => {
      m.classList.remove('open');
      m.querySelector('.icon-btn')?.setAttribute('aria-expanded', 'false');
    });
  });
  el.exportBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.export;
      if (mode === 'json') exportJSON();
      if (mode === 'csv') exportCSV();
      if (mode === 'print') window.print();
    });
  });

  // Drawer
  el.drawerClose.addEventListener('click', closeDrawer);
  el.drawerBackdrop.addEventListener('click', closeDrawer);
}

/* ---------- Init ---------- */
async function init() {
  initTheme();
  state.data = await loadData();
  populateOwnerFilter();
  readStateFromURL();
  syncControlsFromState();
  renderFooterMeta();
  wireEvents();
  render();
}

document.addEventListener('DOMContentLoaded', init);
