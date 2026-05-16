# AI accountability outreach — Paper 4 (mr-audit)

Five emails to send AFTER Paper 4 has been SSRN-distributed and the abstract ID is assigned. The current Paper 4 v1.0 commits the work to GitHub; the email needs the SSRN URL once approved.

Each email is independent. Do not BCC or thread them. Send one per day across a 5-day window so any of them can produce a reply without conflict.

## How to find current email addresses

Do NOT trust any email address listed below. Look up current addresses on the recipient's institution page:

- Margaret Mitchell: `huggingface.co/meg` or `huggingface.co/about` → contact link
- Inioluwa Deborah Raji: `stanford.edu` directory + `iraji.com`
- Markus Anderljung: `governance.ai/about/markus-anderljung`
- Solon Barocas: `microsoft.com/en-us/research/people/sobaroca/`
- Roel Dobbe: `tudelft.nl` directory + `roeldobbe.org`

All five maintain public-facing institutional pages.

## Common signature block

Use this verbatim after every email:

```
Akram Khan
Independent Researcher
ORCID: 0009-0002-7521-8648
SSRN: https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=11116668
Google Scholar: [your URL once active]
GitHub: https://github.com/ayk5511
Email: 1819ak@gmail.com
```

---

## Email 1: Margaret Mitchell (Hugging Face) — Model Cards author

**Subject:** Audit-trail framework for SR 11-7 / EU AI Act, complementing Model Cards

Dear Dr. Mitchell,

I have just released **mr-audit**, an open-source Python package that produces immutable, hash-chained, per-prediction audit trails for machine-learning inference pipelines under SR 11-7, OCC 2011-12, EU AI Act Articles 12 and 19, NIST AI 100-1, and ISO/IEC 42001. The package is at https://github.com/ayk5511/mr-audit (MIT-licensed, 54 unit tests passing). The companion paper is on SSRN at [SSRN_URL] (39 pages).

The paper engages directly with Model Cards (Mitchell et al. 2019). My framing is that Model Cards are point-in-time documentation artefacts that answer "what is this model" while audit trails are per-prediction provenance records that answer "what was this specific prediction." The two are complementary, not substitutable; deployment in regulated settings (US bank model risk, EU AI Act high-risk) requires both. I argue this explicitly in §1.3 of the paper.

The reason I am writing to you specifically: the Model Cards work has set the field's vocabulary for ML documentation, and your judgment on whether the audit-trail framing I propose is a useful complement (vs. an overlapping or redundant artefact) would be more useful to me than any other review I could solicit. If you have time to skim §1.3 (Related work) and §2 (Regulatory framework, particularly the cross-regulation traceability matrix in Table 5), I would value your reaction. Critical responses are as useful as supportive ones.

The work is part of a five-paper portfolio on regulator-defensible AI for quantitative finance; this paper is the audit-trail contribution. I am not asking you to read the full paper, only the framing of how it relates to Model Cards.

Thank you for your work on Model Cards; it has shaped how this field communicates ML systems publicly.

Best regards,

[signature block]

---

## Email 2: Inioluwa Deborah Raji (Stanford) — Algorithmic Auditing

**Subject:** Operationalizing the audit-trail mechanism from your Closing-the-AI-Accountability-Gap framework

Dear Dr. Raji,

I have just released **mr-audit**, an open-source Python package that produces immutable, hash-chained, per-prediction audit trails for ML inference pipelines. The package is at https://github.com/ayk5511/mr-audit (MIT-licensed, 54 unit tests passing). The companion paper is on SSRN at [SSRN_URL] (39 pages).

Your "Closing the AI Accountability Gap" paper (Raji et al. 2020) is one of the four anchor citations in my Related Work section. Specifically, you and your co-authors argue that compliance-grade internal auditing requires verifiable artefacts at each stage of the model lifecycle, not just at deployment. My contribution is to operationalize the audit-trail mechanism for the deployment phase specifically, in a form that satisfies SR 11-7 (US bank model risk), EU AI Act Articles 12 and 19, NIST AI 100-1, and ISO/IEC 42001 simultaneously. Table 5 of the paper is a clause-by-clause cross-walk across all five frameworks.

The paper also includes an adversarial-robustness section (Table 11) where I deliberately tamper with the audit log under four scenarios and report detection rates. Three scenarios are detected; one (tail truncation) is not, which I flag explicitly as the v0.2.0 external-attestation roadmap item. The willingness to publish an honest negative result on integrity protection is, I think, the kind of scientific seriousness your end-to-end-framework paper called for.

The reason for writing to you: your framework's emphasis on per-stage verifiable artefacts is the conceptual scaffold for the package. If you see places where the package's design choices diverge from how an internal auditor would actually use the framework, I would value the correction.

Best regards,

[signature block]

---

## Email 3: Markus Anderljung (GovAI, Oxford) — Verifiable Claims

**Subject:** Open-source audit-trail substrate for the verifiable-claims agenda

Dear Dr. Anderljung,

I have just released **mr-audit**, an open-source Python package that produces immutable, hash-chained per-prediction audit trails for ML inference pipelines. The package is at https://github.com/ayk5511/mr-audit; the companion paper is on SSRN at [SSRN_URL].

The Brundage et al. (2020) "Toward Trustworthy AI Development" paper, on which you are a co-author, lists tamper-evident audit trails among the institutional, software, and hardware mechanisms required for verifiable AI claims. My contribution is one specific software mechanism in the form of an open-source Python package, calibrated against existing regulatory text (SR 11-7, EU AI Act Arts 12/19, NIST AI 100-1, ISO/IEC 42001) rather than against the AI-safety literature directly. The companion paper is on SSRN; the package is on GitHub under MIT license.

The reason I am writing to you: the verifiable-claims agenda is one of two intellectual anchors for this work (the other is regulated-industry model risk management). I would value your reaction to whether the package's design choices reasonably implement the kind of "tamper-evident audit trail" your paper called for, or whether the design has diverged from that vision in ways worth flagging. If the answer is that this is a useful contribution to operationalize one specific mechanism among the dozen your paper enumerates, I would be grateful to be told. If it is that the design has missed something important, I would be doubly grateful.

The package's MIT license and the paper's CC-BY 4.0 license mean adoption by any institution requires no permissions discussion.

Best regards,

[signature block]

---

## Email 4: Solon Barocas (Microsoft Research / Cornell) — Fair ML

**Subject:** Audit-trail framework for ML deployment under SR 11-7 / EU AI Act

Dear Dr. Barocas,

I have just released **mr-audit**, an open-source Python package and a companion paper documenting an audit-trail framework for ML inference pipelines under SR 11-7, OCC 2011-12, EU AI Act, NIST AI 100-1, and ISO/IEC 42001. Code: https://github.com/ayk5511/mr-audit (MIT). Paper: [SSRN_URL] (39 pages).

I am writing to you because much of the algorithmic-accountability literature has focused on training-time and validation-time artefacts (Model Cards, Datasheets, Data Cards, FactSheets). What gets less attention is the deployment-time per-prediction record that a regulator examining a deployed system can use to reconstruct any individual prediction. That gap is the empirical focus of the paper, and I would value your reaction.

The paper's empirical demonstration retrospectively instruments a 7-model financial volatility-forecasting pipeline and exercises five use cases on the resulting 6,825-record audit log (reproducibility audit, drift detection, validation report, regulator export, GDPR Article 22 right-to-explanation). The package also has documented cross-domain validation: the same schema works on a credit-classifier (an EU AI Act Annex III §5(b) high-risk use case) and a macro-economic forecaster without modification.

I am an independent researcher with a model-risk day job at a major US bank, working on a five-paper portfolio for regulator-defensible AI in quantitative finance. Your work on the broader landscape of algorithmic-accountability research is one of the lenses I have used to position this paper; your reaction (positive, negative, or "have you considered X") would be more useful than any other piece of feedback.

Best regards,

[signature block]

---

## Email 5: Roel Dobbe (TU Delft) — Sociotechnical Systems

**Subject:** Open-source audit-trail substrate for sociotechnical AI governance

Dear Dr. Dobbe,

I have just released **mr-audit**, an open-source Python package producing per-prediction audit trails for ML inference pipelines under multiple regulatory regimes (SR 11-7 US bank model risk, EU AI Act, NIST AI 100-1, ISO/IEC 42001). Code: https://github.com/ayk5511/mr-audit. Paper: [SSRN_URL].

Your work on sociotechnical AI systems argues that AI governance must engage with the implementation substrate, not just policy. My paper is on one specific implementation substrate: the per-prediction record-keeping infrastructure that an SR 11-7 / EU AI Act regime requires. I attempt to ground the package in the regulatory text rather than in any one institution's audit practice; the cross-regulation traceability matrix (Table 5 of the paper) maps 18 audit-trail field requirements across 5 frameworks.

The narrowest claim of the paper is that audit-grade ML logging in regulated industries does not require commercial cloud platforms or bespoke internal solutions; it can be done with a one-line decorator and a context manager on top of any Python ML pipeline. The MIT license and the audit-script reproducibility commitment (every numerical claim in the paper is re-derivable from JSON outputs by a script in the repository) are intended to support adoption.

The reason for writing to you: your TU Delft / Sociotechnical Systems work spans the institutional / technical interface that this kind of tooling either bridges or fails to bridge. I would value your reaction to whether the design choices the paper documents reasonably engage with the sociotechnical reality (firms with model-risk teams, regulators with limited engineering capacity, auditors who need verifiable artefacts) or whether they are technically interesting but institutionally naive.

Best regards,

[signature block]

---

## After-the-email follow-up plan

For each email sent:

1. **Tag in Gmail.** Label "P4-outreach-{recipient}" so you can find replies easily.
2. **48-hour quiet period.** Do not chase. If they reply, respond within 24 hours.
3. **Two-week silent reply.** If no reply after 14 days, that is the signal. Do not re-send. The probability of a substantive reply after a silent two weeks is essentially zero.
4. **Save replies.** Any substantive reply (positive, negative, or neutral) becomes evidence for `evidence/criterion-5-contributions/expert-acknowledgements/`. Even a "thanks, will read when I have time" reply is evidence of recognition.

## Probability estimates (honest)

| Target | P(any reply) | P(substantive reply) | P(citation in 12mo) |
|--------|--------------|----------------------|---------------------|
| Mitchell | 35% | 15% | 8% |
| Raji | 30% | 15% | 8% |
| Anderljung | 25% | 12% | 6% |
| Barocas | 25% | 10% | 5% |
| Dobbe | 35% | 18% | 10% |

**Combined: ~80% probability of at least one reply, ~50% probability of at least one substantive reply, ~30% probability of at least one citation within 12 months.**

These are not high probabilities. They are the realistic baseline for cold-outreach to senior researchers. The asymmetric payoff comes from the fact that even one substantive reply is petition-changing: it becomes evidence for criterion-5 expert recognition and a potential letter-writer for the I-140.

## What to do with positive replies

A substantive reply opens several follow-up paths in this rough priority:

1. **Ask for feedback.** "I would value any specific reaction you have." This is the simplest follow-up and the one most likely to produce continued engagement.
2. **Offer a video call.** "Would you be open to a 20-minute call to discuss?" Reserve for replies that explicitly ask further questions.
3. **Cite their adjacent work in v1.1.** If they suggest a related paper, read it carefully and cite it in the next revision. Email them when the revision is up.
4. **Eventually: letter of recommendation request.** Only after at least two rounds of substantive correspondence. Do not ask in the first reply. The EB1A timeline supports this if you start outreach now (May 2026) and approach for letters in November 2026 for a February-March 2027 filing.

Do NOT in the first reply:
- Ask for a citation
- Ask for a letter
- Ask them to share with their network
- Ask anything beyond their reaction

The most credible letter-writer is one who has read your work in their own time and engaged with it because they found it interesting. Not one who was solicited to engage.
