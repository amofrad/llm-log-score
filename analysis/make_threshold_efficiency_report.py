"""Summarize the exploratory comparison without changing manuscript outputs."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/exploratory/threshold_efficiency'
NAMES = {'gemini35flash':'Gemini', 'sonnet46':'Sonnet', 'deepseekv32maas':'DeepSeek'}
LABELS = {'bonferroni':'Current', 'holm':'Holm', 'fallback':'Ordered fallback',
          'multistart':'Multi-start sequence', 'tied_bonferroni':'Partial ties + Bonferroni',
          'tied_fallback':'Partial ties + fallback'}


def pct(v):
    return '—' if pd.isna(v) else f'{100*v:.1f}'


def main():
    summary = pd.concat([pd.read_csv(OUT/f'{stage}_summary.csv')
                         for stage in ['deterministic','ties']],ignore_index=True)
    primary = pd.concat([pd.read_csv(OUT/f'{stage}_primary_results.csv')
                         for stage in ['deterministic','ties']],ignore_index=True)
    paired = pd.concat([pd.read_csv(OUT/f'{stage}_paired.csv')
                        for stage in ['deterministic','ties']],ignore_index=True)
    selected = summary[summary.coin_rep.isin([-1,0])]
    methods=list(LABELS)
    lines=['# Can less conservative decisions return more answers?', '',
        'Exploratory comparison dated 2026-09-14. The paper, SI, poster, and their figures and tables are unchanged. This analysis uses the same 4,326 questions, original parsed top candidates, stored grades, 102 thresholds, confidence 95%, and error tolerances 0.20–0.40. No model was retrained or queried.', '',
        'The primary partition and all 100 additional partitions are retained, under both 50–50 and 70–30 allocations and both graders. Methods are paired on identical test questions within an allocation. Tables below use the primary grader unless explicitly stated otherwise.', '',
        '## Assessment', '',
        '**There is an improvement, but no alternative dominates in every setting.** The clearest gain that retains the simultaneous-in-alpha guarantee is partial inclusion of tied reports for Gemini at alpha=0.40: mean answered percentage rises from 62.8% to 70.2%, with median conditional test error changing from 30.1% to 33.7%. The mean answered percentage remains between 70.2% and 70.8% across the five auxiliary randomizations. The gain also appears with the second grader and the 70–30 allocation.', '',
        '**Changing only the multiple-testing correction gives smaller gains.** Holm changes little. Ordered fallback raises Sonnet’s mean answer rate at alpha=0.35 from 13.4% to 17.7%, while median error rises from 22.2% to 28.9%. Multi-start testing increases Sonnet’s qualification at alpha=0.30 from 53 to 77 of the 100 additional partitions, with mean answer rate increasing from 4.9% to 7.7%.', '',
        '**Combining partial ties with ordered fallback gives the largest gains at alpha=0.40**, raising Gemini’s mean answer rate to 76.4% and Sonnet’s to 27.0%, with median test errors of 36.6% and 34.1%. This version has a guarantee for each fixed alpha, not the current joint guarantee over alpha, and it uses randomized decisions. At this setting, test errors exceed alpha in 4/100 and 7/100 partitions, respectively; these finite-test exceedances are not estimates of population-guarantee failure.', '',
        '**The extra search can also hurt.** Partial ties with Bonferroni searches 408 rules instead of 102. It reduces qualification at some stricter tolerances, including Gemini at 0.25 and Sonnet at 0.30 and 0.35. DeepSeek remains unsupported at all tested tolerances. Thus, the new family should not replace the current rule solely on its strongest Gemini result.', '',
        '**Recommendation:** retain the present manuscript analysis pending a decision about whether randomized boundary decisions and a fixed-alpha formulation fit the intended contribution. A partial-tie extension with Bonferroni is the clearest way to demonstrate that probability ties can leave usable error tolerance, while preserving the existing guarantee scope. Ordered testing is a useful deterministic comparison if the guarantee is explicitly scoped to a chosen alpha. Both require clear presentation of gains and losses and independent confirmation of any selected method.', '',
        '## What the alternatives change', '',
        '- **Holm** relaxes the testing correction as hypotheses are rejected.',
        '- **Ordered fallback** passes a rejected threshold’s available testing budget to the next lower threshold. Failed tests retain their budgets.',
        '- **Multi-start sequence** runs descending tests from ten fixed starting thresholds, stopping each sequence at its first failure.',
        '- **Partial ties** returns answers above t and independently returns each answer exactly at t with probability gamma, chosen from 0.25, 0.50, 0.75, 1.00. The 408-rule search receives its full testing correction. Counts are integer realized randomized decisions, not fractional binomial observations.', '',
        'The **current** and **partial ties + Bonferroni** methods retain a guarantee simultaneously across all alpha values, separately for each fixed method, model, grader, allocation, and grid. Holm and both ordered testing procedures have a guarantee for each fixed alpha; they do not automatically retain the joint guarantee across alpha. For partial ties, risk and confidence also account for independent randomization.', '',
        '## Repeated-partition comparison: primary 50–50 allocation', '',
        'Each entry is **mean answered percentage / median conditional test-error percentage** across the 100 additional partitions. Answer rates include zero when unsupported; error medians use only partitions returning answers. These two summaries should not be interpreted as one pooled operating point. Partial-tie results use auxiliary randomization 0, specified before running the extension.', '',
        '| Model | alpha | Current | Holm | Ordered fallback | Multi-start sequence | Partial ties + Bonferroni | Partial ties + fallback |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in NAMES:
        for alpha in [.2,.25,.3,.35,.4]:
            vals=[]
            for method in methods:
                r=selected[(selected.grader=='openai')&(selected.allocation=='50-50')&
                           (selected.model_key==model)&(selected.alpha==alpha)&(selected.method==method)].iloc[0]
                vals.append(f'{pct(r.mean_answer_rate)} / {pct(r.median_error)}')
            lines.append(f'| {NAMES[model]} | {alpha:.2f} | '+' | '.join(vals)+' |')
    lines += ['', '## Qualification and finite test errors', '',
        'A test error above alpha is not by itself a violation of the guarantee on underlying conditional risk. These overlapping partitions are not independent replications, so these counts are not estimates of the guarantee’s failure probability.', '',
        '| Model | alpha | Method | Qualifying /100 | Answering /100 | Test error above alpha / answering |',
        '|---|---:|---|---:|---:|---:|']
    for model in NAMES:
        for alpha in [.2,.25,.3,.35,.4]:
            for method in methods:
                r=selected[(selected.grader=='openai')&(selected.allocation=='50-50')&
                           (selected.model_key==model)&(selected.alpha==alpha)&(selected.method==method)].iloc[0]
                lines.append(f'| {NAMES[model]} | {alpha:.2f} | {LABELS[method]} | {r.n_supported} | {r.n_answering} | {r.n_test_errors_above_alpha}/{r.n_answering} |')
    lines += ['', '## Original partition: thresholds and outcomes', '',
        'Each method uses 2,163 selection and 2,163 test questions. The table shows realized test errors and pointwise exact 95% intervals. For gamma below one, the threshold is the probability at which answers are partially included; above it all nonempty reports answer. Gamma=1 is the usual inclusive threshold rule.', '',
        '| Model | alpha | Method | t | gamma | Test answers | Answered (%) | Test error (%) [95% interval] |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in primary[(primary.grader=='openai')&(primary.allocation=='50-50')&
                     primary.coin_rep.isin([-1,0])].itertuples():
        t='—' if pd.isna(r.threshold) else f'{r.threshold:.3f}'
        gamma='—' if pd.isna(r.gamma) else f'{r.gamma:.2f}'
        err='—' if pd.isna(r.observed_error) else f'{pct(r.observed_error)} [{pct(r.error_lo)}, {pct(r.error_hi)}]'
        lines.append(f'| {NAMES[r.model_key]} | {r.alpha:.2f} | {LABELS[r.method]} | {t} | {gamma} | {r.n_answered} | {pct(r.answer_rate)} | {err} |')
    lines += ['', '## Consistency across graders and allocations', '',
        'Entries are paired changes in mean test answer rate, in percentage points, relative to the current method on the same test questions. Each allocation uses its own held-out sample; this table does not attribute differences between allocations solely to the selected rules. Main partial-tie randomization only.', '',
        '| Model | alpha | Method | Primary, 50–50 | Second grader, 50–50 | Primary, 70–30 | Second grader, 70–30 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for model in NAMES:
        for alpha in [.2,.25,.3,.35,.4]:
            for method in methods[1:]:
                values=[]
                for grader,allocation in [('openai','50-50'),('gemini','50-50'),('openai','70-30'),('gemini','70-30')]:
                    r=paired[(paired.grader==grader)&(paired.allocation==allocation)&
                             (paired.model_key==model)&(paired.alpha==alpha)&(paired.method==method)&
                             paired.coin_rep.isin([-1,0])].iloc[0]
                    values.append(f'{100*r.mean_answer_rate_gain:+.2f}')
                lines.append(f'| {NAMES[model]} | {alpha:.2f} | {LABELS[method]} | '+' | '.join(values)+' |')
    lines += ['', '## Sensitivity to auxiliary randomization', '',
        'Ranges below are across five preplanned auxiliary randomizations, each evaluated across the same 100 additional partitions. They describe sensitivity, not confidence intervals or independent datasets. Primary grader, 50–50 allocation.', '',
        '| Model | alpha | Method | Mean answered (%), range | Mean expected answered (%), range | Median error (%), range | Qualifying partitions, range |',
        '|---|---:|---|---:|---:|---:|---:|']
    ties=summary[(summary.grader=='openai')&(summary.allocation=='50-50')&(summary.coin_rep>=0)]
    for (model,alpha,method),g in ties.groupby(['model_key','alpha','method'],sort=False):
        def span(col):return f'{pct(g[col].min())}–{pct(g[col].max())}'
        lines.append(f'| {NAMES[model]} | {alpha:.2f} | {LABELS[method]} | {span("mean_answer_rate")} | {span("mean_expected_answer_rate")} | {span("median_error")} | {g.n_supported.min()}–{g.n_supported.max()} |')
    lines += ['', '“Expected answered” averages each selected rule’s acceptance probabilities on the fixed test questions, integrating out test-time coins. Selection still uses realized independent coins. No fractional-count confidence intervals are used.', '',
        '## Validation and interpretation', '',
        'All 61,812 saved 50–50 baseline bounds and all 6,060 baseline selections/test evaluations across both allocations and graders were reproduced. Implementation checks cover integer binomial counts, empty lists, nested answer sets, recovery of the original rules at gamma=1, preservation of Bonferroni rejections where required, stopping and budget-transfer behavior, and absence of test-grade input to selection.', '',
        'These are exploratory comparisons on a previously examined benchmark. Any later method choice should be stated explicitly and confirmed independently. In particular, selecting the best method separately for each model/alpha after seeing these outcomes is not covered by the individual methods’ guarantees. There is no need for observed test error to equal alpha; the objective is more answered questions at controlled underlying conditional error.', '',
        'The methods use established exact binomial testing and multiple-testing arguments. See [Learn then Test, Sections 2.3.1–2.3.2](https://arxiv.org/html/2110.01052v5#S2.SS3). The theory note, all selections, and paired summaries are retained in the repository.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    selected.to_csv(OUT/'comparison_summary.csv',index=False)
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
