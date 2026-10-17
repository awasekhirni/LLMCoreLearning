#MVP Model Risk Dashboard
# Awase Khirni Syed

Enhancements to the existing data.json 

Predictive dimension	Enhanced KPI / KRI	Purpose
Risk velocity / momentum	30/90-day score delta, likelihood/impact delta, trend acceleration, days since last change	Predict which risks are emerging or worsening before they breach thresholds
Control effectiveness	Mitigation implementation %, control test pass rate, evidence coverage, residual_score = inherent_score × (1 − control_effectiveness)	Predict real residual exposure, not just “in progress” status
Detection & response	MTTD, MTTR, alert coverage, rollback/restore time, dry-run success rate	Predict incident duration and recoverability
Exposure / blast radius	Affected systems, records, users, models, agents; PII/secret sensitivity; external-facing flag; privilege level; autonomy tier	Predict how far an incident could spread
Assurance / validation coverage	Eval coverage, adversarial/red-team coverage, framework control mapping %, evidence freshness, independent validation	Predict assurance gaps and audit weakness
Data/model leading indicators	Drift PSI/KS, DQ pass rate, hallucination rate, eval regression pass rate, prompt-sensitivity variance	Predict model/data degradation before score moves
Security exploitability	Injection success rate, secret-scan findings, token TTL, egress-block coverage, vuln age/EPSS	Predict exploit likelihood and attack surface
Agentic control	Tool-allowlist coverage, approval-gate coverage, step-budget breaches, permission-scope score, loop-detection rate	Predict agent misuse, unintended actions, permission escalation
Governance health	Review overdue days, owner risk load, RACI coverage, sign-off SLA, audit-finding closure rate	Predict accountability and process failure
Compliance exposure	Control-to-regulation mapping %, audit findings, compliance test pass rate, regulatory-change impact	Predict compliance/regulatory breach likelihood
Incident / recurrence	Incidents and near-misses in 90 days, repeat-incident rate, root-cause closure rate	Predict recurrence and control failure
Confidence / evidence quality	Assessment confidence, evidence completeness, source count, review freshness	Weight predictions and avoid false precision
Dependency / cascade	Dependency criticality, third-party single points of failure, correlated tags/categories	Predict systemic and correlated risk
Portfolio concentration	High-risk count by owner/category/tag, overlapping high-risk risks, heat concentration	Support portfolio prioritization and resource planning
Cost / ops predictive	Budget burn rate, cost-per-transaction trend, latency SLO burn	Predict cost overrun and operational degradation
