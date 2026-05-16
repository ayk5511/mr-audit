# Cover letter — mr-audit paper

This document holds the cover letter and SSRN submission metadata. Two
versions: a short SSRN-form abstract (for the SSRN upload form) and a
longer journal cover letter (for *Journal of Financial Data Science* or
*Journal of Risk and Financial Management* future submission).

---

## SSRN submission metadata

**Title:** An Audit-Trail Framework for Machine-Learning Pipelines under SR 11-7 and the EU AI Act: A Demonstration on Volatility Forecasting

**Author:** Akram Khan (Independent Researcher)
**ORCID:** 0009-0002-7521-8648
**Affiliation field:** Independent Researcher
**Email:** 1819ak@gmail.com

**Keywords:** model risk management, SR 11-7, OCC 2011-12, EU AI Act, audit trail, machine learning, reproducibility, compliance, regulatory technology, volatility forecasting

**JEL codes:** C52, C53, C88, G17, G28

**SSRN networks (suggested, four-network limit):**
- Risk Management eJournal
- Financial Engineering eJournal
- Econometric Modeling: Capital Markets - Forecasting eJournal
- Banking & Insurance eJournal

**Distribution:** Yes (free, immediate)
**Public:** Yes
**License:** CC BY 4.0 (so the package's MIT license and the paper's open license align)

**Abstract** (paste verbatim into the SSRN abstract field; 4 paragraphs, ~3,600 chars; under SSRN's 5,000-char limit):

> The U.S. Federal Reserve's SR 11-7 (April 2011) and the EU AI Act (Regulation 2024/1689, Articles 12 and 19) require firms deploying machine-learning models in regulated domains to maintain audit trails of every prediction. Code state, data state, model state, compute environment, and the prediction itself must all be recorded, with enough integrity protection that an external reviewer can verify any claim ex post.
>
> Existing open-source ML tooling (MLflow, Weights & Biases, DVC, SageMaker Model Monitor) targets experiment tracking, model registry, or cloud-managed monitoring. None of these produce an SR 11-7-shaped audit trail with low-friction integration, vendor-neutral storage, and regulator-export capability out of the box. This paper introduces mr-audit, an open-source Python package that fills the gap. The package writes one immutable, hash-chained record per prediction, capturing git commit hash, library versions, input hash, model parameters, compute environment, random seed, and prediction value. Records form a Merkle-style chain via SHA-256 hash linking, so tampering with any single record breaks the chain. The package ships with a replay engine, a drift-detection module (Kolmogorov-Smirnov plus Population Stability Index), and a regulator-export bundle that produces a self-contained zip whose integrity can be verified with off-the-shelf tools.
>
> We demonstrate the package by retrospectively instrumenting the seven-model panel of Khan (2026, "Volatility Forecasting with Machine Learning," SSRN 6663418): S&P 500 5-day realized volatility, 980 trading days from 2022-01-03 to 2025-11-26. The instrumentation produces a 6,825-record audit log. Five use cases are demonstrated end-to-end. (i) Reproducibility audit: 5/5 randomly sampled records replay bit-identically. (ii) Drift detection: the KS test flags drift in 44 of 44 evaluated months relative to the 2022 Q1 baseline, consistent with the documented regime shift between high-vol 2022 and lower-vol 2023+. (iii) Per-model validation report generated from the log alone. (iv) Regulator-export bundle (841 KB zip; all SHA-256 checks pass). (v) Right-to-explanation: per-prediction reconstruction satisfying GDPR Article 22 and EU AI Act Article 86 obligations.
>
> The package, the demonstration scripts, the 6,825-record audit log, and an audit script that re-derives every numerical claim in this paper from JSON outputs are released at github.com/ayk5511/mr-audit, scoring 2 on the Reproducibility Disclosure Score rubric of Khan (2026, "Machine Learning in Quantitative Finance: A Systematic Review," SSRN 6562398).

---

## Journal cover letter (for future submission to JFDS / JRFM / similar)

> *To the Editors:*
>
> I am pleased to submit the manuscript "An Audit-Trail Framework for Machine-Learning Pipelines under SR 11-7 and the EU AI Act: A Demonstration on Volatility Forecasting" for consideration. The paper has two contributions. First, it introduces mr-audit, an open-source Python package that produces immutable, hash-chained, replayable audit trails for ML inference pipelines under SR 11-7, OCC 2011-12, and EU AI Act Articles 12 and 19. Second, it demonstrates the package by retrospectively instrumenting the seven-model volatility-forecasting panel of Khan (2026) and exercising five use cases end-to-end on the resulting 6,825-record audit log.
>
> The paper is of interest to the journal's readership for three reasons.
>
> *First, the regulatory tailwind is strengthening.* The EU AI Act's high-risk regime becomes fully applicable on August 2, 2026. SR 11-7 has been in force for fifteen years but supervisory enforcement against ML-specific failures is sharpening. Firms deploying ML in regulated domains will need audit-trail infrastructure within the next 18-24 months, and the existing open-source MLOps stack does not provide it.
>
> *Second, the tooling gap is documented quantitatively.* Section 2.5 of the manuscript includes a capability-comparison table across six existing tools (MLflow, Weights & Biases, DVC, Pachyderm, SageMaker Model Monitor, IBM AI FactSheets) along nine audit-trail dimensions derived from SR 11-7 and EU AI Act requirements. mr-audit is the only entry that satisfies all nine out of the box. The table is narrow in scope and explicit about it (it does not score the tools on dimensions where they are clearly stronger), so the comparison is defensible.
>
> *Third, the demonstration is end-to-end at a non-toy scale.* The 6,825-record audit log captures one record per (date, model) cell of the Khan (2026) forecast panel. All five use cases pass: 5/5 replays bit-identical, 44/44 months drift-flagged consistently with the documented regime shift, per-model validation report computable from the log alone, 841 KB regulator-export bundle with all SHA-256 verification checks passing, and per-prediction right-to-explanation reconstruction satisfying GDPR Article 22 and EU AI Act Article 86. Synthetic scaling benchmarks at n=1,000 and n=10,000 records (Section 4.1.1) anchor write and verify throughput at realistic deployment sizes.
>
> The paper is honest about its limitations. Section 5.3 enumerates five: the demonstration is retrospective rather than live; the "drift in 44 of 44 months" result is partly a feature of the simple fixed-baseline design rather than purely a model-quality signal; hash-chain integrity is not cryptographically signed by an external authority; feature values are stored as summaries by default; and the package targets Python pipelines exclusively.
>
> The manuscript has not been submitted elsewhere and is not under consideration at any other journal. I have no competing interests to declare. All code, data, and audit logs are publicly available; the package is MIT-licensed at github.com/ayk5511/mr-audit and the demonstration's 6,825-record audit log is committed to the same repository. The paper itself ships with an audit script that re-derives every numerical claim from JSON outputs, providing a mechanical guard against transcription errors of the kind documented in Khan (2026, "Volatility Forecasting").
>
> Thank you for your consideration.
>
> Sincerely,
>
> Akram Khan
> Independent Researcher
> ORCID: 0009-0002-7521-8648
> Email: 1819ak@gmail.com

---

## SSRN submission steps (operational checklist)

1. Log in to ssrn.com with `1819ak@gmail.com` / Author ID 11116668.
2. "Submit a Paper" → "Submit New Paper".
3. Upload `paper/submission-ssrn/Khan_2026_mr_audit_v0.2.pdf` (the current versioned PDF).
4. Title field: paste full title from above.
5. Abstract field: paste the abstract above (under 5000 chars).
6. Author Information: confirm pre-filled details.
7. Keywords: paste from above (10 keywords).
8. JEL Codes: paste from above (5 codes: C52, C53, C88, G17, G28).
9. Networks: select the four listed (or as many as SSRN allows).
10. Distribution: Public, Free, Immediate.
11. Verify the upload preview matches the PDF before final submit.
12. Wait 1-3 business days for SSRN approval.
13. Once approved: link from your SSRN profile, post to LinkedIn, file to ORCID.

---

## After SSRN: post-submission checklist

- [ ] SSRN approval email received → save screenshot to `evidence/criterion-6-articles/`
- [ ] Add Paper 4 to SSRN author profile
- [ ] Update Google Scholar profile with assigned SSRN ID
- [ ] Update website (`now.qmd`, `research/audit-trail.qmd`, `cv.qmd`) with SSRN URL
- [ ] Update STATUS.md, `_claims.yaml`, `_metrics.yaml`, `_eb1a_mapping.yaml`
- [ ] Regenerate `EB1A_DASHBOARD.md` (`scripts/eb1a_dashboard.py`)
- [ ] LinkedIn post announcing the paper
- [ ] PyPI release of `mr-audit` v0.1.0
- [ ] Bank model-risk-management outreach (see `papers/_eb1a_mapping.yaml` artefact `bank_mrmg_outreach`)
- [ ] Press pitch to Risk.net pegged to the SR 11-7 / EU AI Act framing
- [ ] Begin Paper 5 v1.0 (now that Papers 1-4 are all distributed)
