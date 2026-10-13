# Annotation Guidelines (schema v1.0)

> For the human gold set (Phase 1, issue #10) **and** the LLM teacher prompt (Phase 2). The machine form of these
> rules is `src/medsignal/labeling/schema.py`; if this document and the code disagree, fix one of them in the same PR.
>
> Examples are **constructed sentences modelled on real reviews** (the review ID in brackets lets you look up the
> original locally in `data/interim/reviews.parquet`); review text is not redistributed (docs/DATA.md).

## What you label, per review

| Field | Values | One-line rule |
|---|---|---|
| **ADE mentions** | 0..n verbatim spans | Unwanted or neutral bodily/mental effects the reviewer attributes to **this** drug |
| ↳ `severity` | `mild` · `moderate` · `severe` · `not_stated` | From the reviewer's words, not your medical judgement |
| ↳ `withdrawal` | true / false | Effect appeared when stopping, reducing, or missing doses |
| ↳ `beneficial` | true / false | Reviewer presents a non-therapeutic effect as welcome (e.g. no periods, weight loss) |
| **discontinued** | `yes` · `planned` · `no` · `not_stated` | Did they stop / switch away from **this** drug? |
| **stop_reasons** | one or more of the list below | Only when discontinued is `yes` or `planned` |
| **effectiveness** | `effective` · `partial` · `not_effective` · `not_stated` | For the condition it was taken for |

Always label from the reviewer's point of view and only what the text says. When unsure, prefer `not_stated`
and add a short `notes` entry; uncertain items are reviewed later.

---

## 1. ADE mentions

### 1.1 Label it if ALL of these hold
1. It is an effect on body, mind, behaviour, or a lab/vital value (rash, insomnia, "felt like a zombie", gained 20 lbs, high liver enzymes).
2. The reviewer experienced it (not hypothetical, not someone else's).
3. The reviewer attributes it to **the reviewed drug** (explicitly or by clear implication).
4. It is not the therapeutic goal and not a symptom of the condition being treated.

### 1.2 Do NOT label
| Case | Example | Why |
|---|---|---|
| Negated | "Two weeks in. No side effects." [d2c7d88c77bb77f1] | Did not happen |
| Hypothetical / worry | "Sex drive is fine so far, I was worried about that." [7ada7d702838fad4] | Did not happen |
| Symptom of the treated condition, or its relief | "It stops my migraine pain, nausea and light sensitivity within an hour." [072a1400d23c36c5] | Indication, not ADE (→ effectiveness) |
| Another drug's effect | "On the Depo shot I bled for 2 months and gained 40 pounds." (review is of Nexplanon) [2eb2aee09417fbcb] | Wrong drug |
| The condition getting worse | "It made my muscle spasm worse." [8cf1ac4480eaaf13] | → `effectiveness: not_effective` |
| Generic statements | "This drug has bad side effects." (no effect named) | Nothing specific to span |

### 1.3 Span rules
- Copy the **shortest text that still names the effect**, verbatim, including its body location if given: `pain in my legs`, `couldn't sleep`, `gained 40 lbs`, `long QT interval`.
- Leave severity words **out** of the span when they are separable and put them in `severity`: "extreme fatigue" → span `fatigue`, severity `severe`. Keep them if they are inseparable from the effect ("felt drunk").
- One span per effect. "headache, nausea and dizziness" → three spans.
- The same effect mentioned twice → label each mention (the aligner keeps them in reading order).
- Do not include the drug name, time expressions, or "side effects of".

### 1.4 Severity (from the reviewer's words)
| Value | Use when the text says or clearly implies |
|---|---|
| `severe` | hospital / ER, life-threatening, unable to function or work, "unbearable", "horrific", "extreme", "worst ever" |
| `moderate` | clearly bothersome or the reason for stopping, but none of the `severe` cues |
| `mild` | "slight", "minor", "a little", "went away quickly", "tolerable" |
| `not_stated` | the effect is named with no intensity cue |

### 1.5 `withdrawal` and `beneficial`
- `withdrawal: true` when the effect followed stopping, tapering, or a missed dose: "dizzy whenever I forget a dose" [851113d9f281eb35], "went off it and had a hard withdrawal" [75dde7bc509771e8] → span `withdrawal`.
- `beneficial: true` for a non-therapeutic effect the reviewer welcomes: "No period at all since month three, which is great" on a contraceptive [b13f36f207483301]. Still label it: it is a real drug effect (amenorrhoea) and FAERS-style comparisons need it.

### 1.6 Procedures and delayed effects
- Effects of administration count (insertion cramping, injection-site pain) [45dee5fe98699031].
- Delayed effects after finishing a course count if attributed to the drug: "a month after finishing it I developed severe joint pain" [2b203831ef51e7c5].

---

## 2. Discontinuation

| Value | Use when |
|---|---|
| `yes` | stopped, quit, came off, switched to another drug, had it removed, finished early |
| `planned` | "I'm getting it removed next month", "after this pack I'm going off it", "today is my last day" |
| `no` | still taking it / says they will continue / restarted after a break |
| `not_stated` | nothing about continuing or stopping |

Finishing a prescribed course is `yes` with reason `course_completed`. A dose change alone is **not** discontinuation.

### 2.1 Stop reasons (one or more; only for `yes` / `planned`)
| Reason | Example cue |
|---|---|
| `adverse_event` | "stopped because of the headaches"; also when an ADE is described and the stop follows it without another reason [7b3686734924f223] |
| `lack_of_efficacy` | "never worked", "stopped working", "condition got worse" |
| `cost_access` | "too expensive", "lost my insurance", "pharmacy couldn't get it" [3eb6da83d459e443] |
| `convenience` | dosing too often, hard to take, injections, "too many pills" |
| `course_completed` | finished the prescription / no longer needed |
| `pregnancy_related` | trying to conceive, pregnant, breastfeeding |
| `clinician_decision` | "my doctor switched me" with **no** other reason given |
| `other` | any other stated reason (add a note) |
| `not_stated` | stopped, but no reason given (cannot be combined with others) |

If the doctor switched them *because of* side effects, use `adverse_event` (not `clinician_decision`).

---

## 3. Effectiveness

| Value | Use when |
|---|---|
| `effective` | clear benefit for the treated condition |
| `partial` | some benefit, mixed results, or it worked and then wore off |
| `not_effective` | no benefit, or the treated condition got worse |
| `not_stated` | nothing about benefit |

Judge the **final** state described. "Worked great for months, then stopped working, so I quit" → `partial` + stop reason `lack_of_efficacy`.

---

## 4. Worked examples

**A.** "Within two weeks on Paxil I had constant migraines, extreme fatigue and constipation. My doctor moved me to Zoloft and I feel much better." [7b3686734924f223]
- ADEs: `migraines` (moderate: reason for stopping) · `fatigue` (severe: "extreme") · `constipation` (moderate)
- discontinued `yes` · reasons `adverse_event` (implied by the sequence) · effectiveness `not_stated`

**B.** "Worked great for four years with no side effects. Had to stop because it got too expensive." [3eb6da83d459e443]
- ADEs: none · discontinued `yes` · reasons `cost_access` · effectiveness `effective`

**C.** "Ended up in hospital with a severely long QT interval. Get an ECG before starting. The withdrawal was horrific too." [d4cc6ba02fa9d6bf]
- ADEs: `long QT interval` (severe) · `withdrawal` (severe, withdrawal=true)
- discontinued `yes` (withdrawal implies stopping) · reasons `adverse_event` · effectiveness `not_stated`

**D.** "No period at all after month three, which is great because my cramps were awful. But I've become very depressed and gained a lot of weight. After this pack I'm going off it." [b13f36f207483301]
- ADEs: `No period` (not_stated, beneficial=true) · `depressed` (moderate: reason for stopping; "very" alone is not a severe cue) · `gained a lot of weight` (moderate)
- discontinued `planned` · reasons `adverse_event` · effectiveness `effective`

**E.** "It seemed to make my shoulder spasm worse, and on the second try I also got a rash. Not for me." [8cf1ac4480eaaf13]
- ADEs: `rash` (moderate: part of the reason for stopping) — the spasm worsening is **not** an ADE
- discontinued `yes` · reasons `lack_of_efficacy`, `adverse_event` · effectiveness `not_effective`

**F.** "Insertion felt like a bad cramp. A few hours later, minor cramping and very little spotting." [45dee5fe98699031]
- ADEs: `cramp` (moderate: "bad") · `cramping` (mild) · `spotting` (mild)
- discontinued `not_stated` · effectiveness `not_stated`

**G.** "On it for 5 days, no side effects yet, feeling better than in months." [d561cfca2d218ddd]
- ADEs: none · discontinued `not_stated` ("on it for 5 days" implies current use but does not say they will continue; use `no` only when continuation is explicit) · effectiveness `effective`

---

## 5. Process

1. Annotate in the local tool (issue #9). Reviews are **pre-filled by the local LLM**: check every field; add, fix or delete spans. Pre-annotation is recorded in the export (`pre_annotated_by`).
2. Gold set: 300 full-text reviews (`possibly_truncated == False`), stratified by rating and condition; split 100 **dev** (prompt design, guideline fixes) / 200 **test** (never used for tuning; AGENTS.md §7.3).
3. Self-agreement: re-annotate 30 reviews after at least a week without looking at the first pass; report agreement (span F1, Cohen's κ for discontinued/effectiveness).
4. Guideline changes after annotation starts are versioned (schema v1.x) and noted in `docs/DECISIONS.md`; affected reviews are re-checked.
