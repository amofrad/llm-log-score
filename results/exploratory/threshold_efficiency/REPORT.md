# Can less conservative decisions return more answers?

Exploratory comparison dated 2026-09-14. The paper, SI, poster, and their figures and tables are unchanged. This analysis uses the same 4,326 questions, original parsed top candidates, stored grades, 102 thresholds, confidence 95%, and error tolerances 0.20–0.40. No model was retrained or queried.

The primary partition and all 100 additional partitions are retained, under both 50–50 and 70–30 allocations and both graders. Methods are paired on identical test questions within an allocation. Tables below use the primary grader unless explicitly stated otherwise.

## Assessment

**There is an improvement, but no alternative dominates in every setting.** The clearest gain that retains the simultaneous-in-alpha guarantee is partial inclusion of tied reports for Gemini at alpha=0.40: mean answered percentage rises from 62.8% to 70.2%, with median conditional test error changing from 30.1% to 33.7%. The mean answered percentage remains between 70.2% and 70.8% across the five auxiliary randomizations. The gain also appears with the second grader and the 70–30 allocation.

**Changing only the multiple-testing correction gives smaller gains.** Holm changes little. Ordered fallback raises Sonnet’s mean answer rate at alpha=0.35 from 13.4% to 17.7%, while median error rises from 22.2% to 28.9%. Multi-start testing increases Sonnet’s qualification at alpha=0.30 from 53 to 77 of the 100 additional partitions, with mean answer rate increasing from 4.9% to 7.7%.

**Combining partial ties with ordered fallback gives the largest gains at alpha=0.40**, raising Gemini’s mean answer rate to 76.4% and Sonnet’s to 27.0%, with median test errors of 36.6% and 34.1%. This version has a guarantee for each fixed alpha, not the current joint guarantee over alpha, and it uses randomized decisions. At this setting, test errors exceed alpha in 4/100 and 7/100 partitions, respectively; these finite-test exceedances are not estimates of population-guarantee failure.

**The extra search can also hurt.** Partial ties with Bonferroni searches 408 rules instead of 102. It reduces qualification at some stricter tolerances, including Gemini at 0.25 and Sonnet at 0.30 and 0.35. DeepSeek remains unsupported at all tested tolerances. Thus, the new family should not replace the current rule solely on its strongest Gemini result.

**Recommendation:** retain the present manuscript analysis pending a decision about whether randomized boundary decisions and a fixed-alpha formulation fit the intended contribution. A partial-tie extension with Bonferroni is the clearest way to demonstrate that probability ties can leave usable error tolerance, while preserving the existing guarantee scope. Ordered testing is a useful deterministic comparison if the guarantee is explicitly scoped to a chosen alpha. Both require clear presentation of gains and losses and independent confirmation of any selected method.

## What the alternatives change

- **Holm** relaxes the testing correction as hypotheses are rejected.
- **Ordered fallback** passes a rejected threshold’s available testing budget to the next lower threshold. Failed tests retain their budgets.
- **Multi-start sequence** runs descending tests from ten fixed starting thresholds, stopping each sequence at its first failure.
- **Partial ties** returns answers above t and independently returns each answer exactly at t with probability gamma, chosen from 0.25, 0.50, 0.75, 1.00. The 408-rule search receives its full testing correction. Counts are integer realized randomized decisions, not fractional binomial observations.

The **current** and **partial ties + Bonferroni** methods retain a guarantee simultaneously across all alpha values, separately for each fixed method, model, grader, allocation, and grid. Holm and both ordered testing procedures have a guarantee for each fixed alpha; they do not automatically retain the joint guarantee across alpha. For partial ties, risk and confidence also account for independent randomization.

## Repeated-partition comparison: primary 50–50 allocation

Each entry is **mean answered percentage / median conditional test-error percentage** across the 100 additional partitions. Answer rates include zero when unsupported; error medians use only partitions returning answers. These two summaries should not be interpreted as one pooled operating point. Partial-tie results use auxiliary randomization 0, specified before running the extension.

| Model | alpha | Current | Holm | Ordered fallback | Multi-start sequence | Partial ties + Bonferroni | Partial ties + fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini | 0.20 | 0.1 / 19.9 | 0.1 / 19.9 | 0.2 / 22.5 | 0.5 / 22.1 | 0.0 / — | 0.0 / — |
| Gemini | 0.25 | 12.3 / 18.6 | 12.3 / 18.6 | 15.1 / 19.5 | 16.8 / 18.8 | 10.6 / 18.3 | 14.1 / 20.9 |
| Gemini | 0.30 | 28.3 / 20.6 | 28.3 / 20.6 | 28.9 / 20.6 | 28.7 / 20.6 | 31.0 / 20.9 | 37.3 / 24.9 |
| Gemini | 0.35 | 60.4 / 30.1 | 61.2 / 30.1 | 62.8 / 30.1 | 62.5 / 30.1 | 59.7 / 29.7 | 63.3 / 30.1 |
| Gemini | 0.40 | 62.8 / 30.1 | 62.8 / 30.1 | 62.8 / 30.1 | 62.8 / 30.1 | 70.2 / 33.7 | 76.4 / 36.6 |
| Sonnet | 0.20 | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — |
| Sonnet | 0.25 | 0.1 / 22.2 | 0.1 / 22.2 | 0.1 / 22.1 | 0.7 / 24.4 | 0.0 / 24.7 | 0.0 / 24.7 |
| Sonnet | 0.30 | 4.9 / 22.6 | 4.9 / 22.6 | 5.0 / 22.6 | 7.7 / 22.1 | 4.1 / 22.5 | 4.9 / 24.4 |
| Sonnet | 0.35 | 13.4 / 22.2 | 13.5 / 22.2 | 17.7 / 28.9 | 17.0 / 28.2 | 12.7 / 24.8 | 17.8 / 29.5 |
| Sonnet | 0.40 | 22.3 / 32.1 | 22.5 / 32.1 | 24.3 / 32.1 | 23.7 / 32.1 | 22.5 / 31.3 | 27.0 / 34.1 |
| DeepSeek | 0.20 | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — |
| DeepSeek | 0.25 | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — |
| DeepSeek | 0.30 | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — |
| DeepSeek | 0.35 | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — |
| DeepSeek | 0.40 | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — | 0.0 / — |

## Qualification and finite test errors

A test error above alpha is not by itself a violation of the guarantee on underlying conditional risk. These overlapping partitions are not independent replications, so these counts are not estimates of the guarantee’s failure probability.

| Model | alpha | Method | Qualifying /100 | Answering /100 | Test error above alpha / answering |
|---|---:|---|---:|---:|---:|
| Gemini | 0.20 | Current | 1 | 1 | 0/1 |
| Gemini | 0.20 | Holm | 1 | 1 | 0/1 |
| Gemini | 0.20 | Ordered fallback | 1 | 1 | 1/1 |
| Gemini | 0.20 | Multi-start sequence | 3 | 3 | 3/3 |
| Gemini | 0.20 | Partial ties + Bonferroni | 0 | 0 | 0/0 |
| Gemini | 0.20 | Partial ties + fallback | 0 | 0 | 0/0 |
| Gemini | 0.25 | Current | 67 | 67 | 0/67 |
| Gemini | 0.25 | Holm | 67 | 67 | 0/67 |
| Gemini | 0.25 | Ordered fallback | 67 | 67 | 0/67 |
| Gemini | 0.25 | Multi-start sequence | 76 | 76 | 0/76 |
| Gemini | 0.25 | Partial ties + Bonferroni | 56 | 56 | 0/56 |
| Gemini | 0.25 | Partial ties + fallback | 56 | 56 | 2/56 |
| Gemini | 0.30 | Current | 100 | 100 | 0/100 |
| Gemini | 0.30 | Holm | 100 | 100 | 0/100 |
| Gemini | 0.30 | Ordered fallback | 100 | 100 | 2/100 |
| Gemini | 0.30 | Multi-start sequence | 100 | 100 | 2/100 |
| Gemini | 0.30 | Partial ties + Bonferroni | 100 | 100 | 1/100 |
| Gemini | 0.30 | Partial ties + fallback | 100 | 100 | 4/100 |
| Gemini | 0.35 | Current | 100 | 100 | 0/100 |
| Gemini | 0.35 | Holm | 100 | 100 | 0/100 |
| Gemini | 0.35 | Ordered fallback | 100 | 100 | 0/100 |
| Gemini | 0.35 | Multi-start sequence | 100 | 100 | 0/100 |
| Gemini | 0.35 | Partial ties + Bonferroni | 100 | 100 | 0/100 |
| Gemini | 0.35 | Partial ties + fallback | 100 | 100 | 5/100 |
| Gemini | 0.40 | Current | 100 | 100 | 0/100 |
| Gemini | 0.40 | Holm | 100 | 100 | 0/100 |
| Gemini | 0.40 | Ordered fallback | 100 | 100 | 0/100 |
| Gemini | 0.40 | Multi-start sequence | 100 | 100 | 0/100 |
| Gemini | 0.40 | Partial ties + Bonferroni | 100 | 100 | 1/100 |
| Gemini | 0.40 | Partial ties + fallback | 100 | 100 | 4/100 |
| Sonnet | 0.20 | Current | 0 | 0 | 0/0 |
| Sonnet | 0.20 | Holm | 0 | 0 | 0/0 |
| Sonnet | 0.20 | Ordered fallback | 0 | 0 | 0/0 |
| Sonnet | 0.20 | Multi-start sequence | 0 | 0 | 0/0 |
| Sonnet | 0.20 | Partial ties + Bonferroni | 0 | 0 | 0/0 |
| Sonnet | 0.20 | Partial ties + fallback | 0 | 0 | 0/0 |
| Sonnet | 0.25 | Current | 2 | 2 | 0/2 |
| Sonnet | 0.25 | Holm | 2 | 2 | 0/2 |
| Sonnet | 0.25 | Ordered fallback | 2 | 2 | 0/2 |
| Sonnet | 0.25 | Multi-start sequence | 9 | 9 | 4/9 |
| Sonnet | 0.25 | Partial ties + Bonferroni | 1 | 1 | 0/1 |
| Sonnet | 0.25 | Partial ties + fallback | 1 | 1 | 0/1 |
| Sonnet | 0.30 | Current | 53 | 53 | 0/53 |
| Sonnet | 0.30 | Holm | 53 | 53 | 0/53 |
| Sonnet | 0.30 | Ordered fallback | 53 | 53 | 1/53 |
| Sonnet | 0.30 | Multi-start sequence | 77 | 77 | 1/77 |
| Sonnet | 0.30 | Partial ties + Bonferroni | 47 | 47 | 0/47 |
| Sonnet | 0.30 | Partial ties + fallback | 47 | 47 | 0/47 |
| Sonnet | 0.35 | Current | 100 | 100 | 1/100 |
| Sonnet | 0.35 | Holm | 100 | 100 | 1/100 |
| Sonnet | 0.35 | Ordered fallback | 100 | 100 | 5/100 |
| Sonnet | 0.35 | Multi-start sequence | 100 | 100 | 5/100 |
| Sonnet | 0.35 | Partial ties + Bonferroni | 94 | 94 | 1/94 |
| Sonnet | 0.35 | Partial ties + fallback | 94 | 94 | 5/94 |
| Sonnet | 0.40 | Current | 100 | 100 | 0/100 |
| Sonnet | 0.40 | Holm | 100 | 100 | 0/100 |
| Sonnet | 0.40 | Ordered fallback | 100 | 100 | 7/100 |
| Sonnet | 0.40 | Multi-start sequence | 100 | 100 | 3/100 |
| Sonnet | 0.40 | Partial ties + Bonferroni | 100 | 100 | 0/100 |
| Sonnet | 0.40 | Partial ties + fallback | 100 | 100 | 7/100 |
| DeepSeek | 0.20 | Current | 0 | 0 | 0/0 |
| DeepSeek | 0.20 | Holm | 0 | 0 | 0/0 |
| DeepSeek | 0.20 | Ordered fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.20 | Multi-start sequence | 0 | 0 | 0/0 |
| DeepSeek | 0.20 | Partial ties + Bonferroni | 0 | 0 | 0/0 |
| DeepSeek | 0.20 | Partial ties + fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.25 | Current | 0 | 0 | 0/0 |
| DeepSeek | 0.25 | Holm | 0 | 0 | 0/0 |
| DeepSeek | 0.25 | Ordered fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.25 | Multi-start sequence | 0 | 0 | 0/0 |
| DeepSeek | 0.25 | Partial ties + Bonferroni | 0 | 0 | 0/0 |
| DeepSeek | 0.25 | Partial ties + fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.30 | Current | 0 | 0 | 0/0 |
| DeepSeek | 0.30 | Holm | 0 | 0 | 0/0 |
| DeepSeek | 0.30 | Ordered fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.30 | Multi-start sequence | 0 | 0 | 0/0 |
| DeepSeek | 0.30 | Partial ties + Bonferroni | 0 | 0 | 0/0 |
| DeepSeek | 0.30 | Partial ties + fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.35 | Current | 0 | 0 | 0/0 |
| DeepSeek | 0.35 | Holm | 0 | 0 | 0/0 |
| DeepSeek | 0.35 | Ordered fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.35 | Multi-start sequence | 0 | 0 | 0/0 |
| DeepSeek | 0.35 | Partial ties + Bonferroni | 0 | 0 | 0/0 |
| DeepSeek | 0.35 | Partial ties + fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.40 | Current | 0 | 0 | 0/0 |
| DeepSeek | 0.40 | Holm | 0 | 0 | 0/0 |
| DeepSeek | 0.40 | Ordered fallback | 0 | 0 | 0/0 |
| DeepSeek | 0.40 | Multi-start sequence | 0 | 0 | 0/0 |
| DeepSeek | 0.40 | Partial ties + Bonferroni | 0 | 0 | 0/0 |
| DeepSeek | 0.40 | Partial ties + fallback | 0 | 0 | 0/0 |

## Original partition: thresholds and outcomes

Each method uses 2,163 selection and 2,163 test questions. The table shows realized test errors and pointwise exact 95% intervals. For gamma below one, the threshold is the probability at which answers are partially included; above it all nonempty reports answer. Gamma=1 is the usual inclusive threshold rule.

| Model | alpha | Method | t | gamma | Test answers | Answered (%) | Test error (%) [95% interval] |
|---|---:|---|---:|---:|---:|---:|---:|
| Gemini | 0.20 | Current | — | — | 0 | 0.0 | — |
| Gemini | 0.20 | Holm | — | — | 0 | 0.0 | — |
| Gemini | 0.20 | Ordered fallback | — | — | 0 | 0.0 | — |
| Gemini | 0.20 | Multi-start sequence | — | — | 0 | 0.0 | — |
| Gemini | 0.25 | Current | 0.910 | 1.00 | 329 | 15.2 | 15.5 [11.8, 19.9] |
| Gemini | 0.25 | Holm | 0.910 | 1.00 | 329 | 15.2 | 15.5 [11.8, 19.9] |
| Gemini | 0.25 | Ordered fallback | 0.910 | 1.00 | 329 | 15.2 | 15.5 [11.8, 19.9] |
| Gemini | 0.25 | Multi-start sequence | — | — | 0 | 0.0 | — |
| Gemini | 0.30 | Current | 0.810 | 1.00 | 625 | 28.9 | 18.6 [15.6, 21.8] |
| Gemini | 0.30 | Holm | 0.810 | 1.00 | 625 | 28.9 | 18.6 [15.6, 21.8] |
| Gemini | 0.30 | Ordered fallback | 0.810 | 1.00 | 625 | 28.9 | 18.6 [15.6, 21.8] |
| Gemini | 0.30 | Multi-start sequence | 0.810 | 1.00 | 625 | 28.9 | 18.6 [15.6, 21.8] |
| Gemini | 0.35 | Current | 0.460 | 1.00 | 1339 | 61.9 | 28.7 [26.3, 31.2] |
| Gemini | 0.35 | Holm | 0.410 | 1.00 | 1381 | 63.8 | 29.8 [27.4, 32.2] |
| Gemini | 0.35 | Ordered fallback | 0.410 | 1.00 | 1381 | 63.8 | 29.8 [27.4, 32.2] |
| Gemini | 0.35 | Multi-start sequence | 0.410 | 1.00 | 1381 | 63.8 | 29.8 [27.4, 32.2] |
| Gemini | 0.40 | Current | 0.410 | 1.00 | 1381 | 63.8 | 29.8 [27.4, 32.2] |
| Gemini | 0.40 | Holm | 0.410 | 1.00 | 1381 | 63.8 | 29.8 [27.4, 32.2] |
| Gemini | 0.40 | Ordered fallback | 0.410 | 1.00 | 1381 | 63.8 | 29.8 [27.4, 32.2] |
| Gemini | 0.40 | Multi-start sequence | 0.410 | 1.00 | 1381 | 63.8 | 29.8 [27.4, 32.2] |
| Sonnet | 0.20 | Current | — | — | 0 | 0.0 | — |
| Sonnet | 0.20 | Holm | — | — | 0 | 0.0 | — |
| Sonnet | 0.20 | Ordered fallback | — | — | 0 | 0.0 | — |
| Sonnet | 0.20 | Multi-start sequence | — | — | 0 | 0.0 | — |
| Sonnet | 0.25 | Current | — | — | 0 | 0.0 | — |
| Sonnet | 0.25 | Holm | — | — | 0 | 0.0 | — |
| Sonnet | 0.25 | Ordered fallback | — | — | 0 | 0.0 | — |
| Sonnet | 0.25 | Multi-start sequence | 0.890 | 1.00 | 49 | 2.3 | 20.4 [10.2, 34.3] |
| Sonnet | 0.30 | Current | 0.857 | 1.00 | 85 | 3.9 | 16.5 [9.3, 26.1] |
| Sonnet | 0.30 | Holm | 0.857 | 1.00 | 85 | 3.9 | 16.5 [9.3, 26.1] |
| Sonnet | 0.30 | Ordered fallback | 0.857 | 1.00 | 85 | 3.9 | 16.5 [9.3, 26.1] |
| Sonnet | 0.30 | Multi-start sequence | 0.760 | 1.00 | 234 | 10.8 | 21.4 [16.3, 27.2] |
| Sonnet | 0.35 | Current | 0.760 | 1.00 | 234 | 10.8 | 21.4 [16.3, 27.2] |
| Sonnet | 0.35 | Holm | 0.760 | 1.00 | 234 | 10.8 | 21.4 [16.3, 27.2] |
| Sonnet | 0.35 | Ordered fallback | 0.660 | 1.00 | 418 | 19.3 | 28.5 [24.2, 33.1] |
| Sonnet | 0.35 | Multi-start sequence | 0.760 | 1.00 | 234 | 10.8 | 21.4 [16.3, 27.2] |
| Sonnet | 0.40 | Current | 0.560 | 1.00 | 516 | 23.9 | 32.0 [28.0, 36.2] |
| Sonnet | 0.40 | Holm | 0.560 | 1.00 | 516 | 23.9 | 32.0 [28.0, 36.2] |
| Sonnet | 0.40 | Ordered fallback | 0.560 | 1.00 | 516 | 23.9 | 32.0 [28.0, 36.2] |
| Sonnet | 0.40 | Multi-start sequence | 0.560 | 1.00 | 516 | 23.9 | 32.0 [28.0, 36.2] |
| DeepSeek | 0.20 | Current | — | — | 0 | 0.0 | — |
| DeepSeek | 0.20 | Holm | — | — | 0 | 0.0 | — |
| DeepSeek | 0.20 | Ordered fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.20 | Multi-start sequence | — | — | 0 | 0.0 | — |
| DeepSeek | 0.25 | Current | — | — | 0 | 0.0 | — |
| DeepSeek | 0.25 | Holm | — | — | 0 | 0.0 | — |
| DeepSeek | 0.25 | Ordered fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.25 | Multi-start sequence | — | — | 0 | 0.0 | — |
| DeepSeek | 0.30 | Current | — | — | 0 | 0.0 | — |
| DeepSeek | 0.30 | Holm | — | — | 0 | 0.0 | — |
| DeepSeek | 0.30 | Ordered fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.30 | Multi-start sequence | — | — | 0 | 0.0 | — |
| DeepSeek | 0.35 | Current | — | — | 0 | 0.0 | — |
| DeepSeek | 0.35 | Holm | — | — | 0 | 0.0 | — |
| DeepSeek | 0.35 | Ordered fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.35 | Multi-start sequence | — | — | 0 | 0.0 | — |
| DeepSeek | 0.40 | Current | — | — | 0 | 0.0 | — |
| DeepSeek | 0.40 | Holm | — | — | 0 | 0.0 | — |
| DeepSeek | 0.40 | Ordered fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.40 | Multi-start sequence | — | — | 0 | 0.0 | — |
| Gemini | 0.20 | Partial ties + Bonferroni | — | — | 0 | 0.0 | — |
| Gemini | 0.20 | Partial ties + fallback | — | — | 0 | 0.0 | — |
| Gemini | 0.25 | Partial ties + Bonferroni | — | — | 0 | 0.0 | — |
| Gemini | 0.25 | Partial ties + fallback | — | — | 0 | 0.0 | — |
| Gemini | 0.30 | Partial ties + Bonferroni | 0.800 | 0.25 | 743 | 34.4 | 22.7 [19.8, 25.9] |
| Gemini | 0.30 | Partial ties + fallback | 0.800 | 0.50 | 859 | 39.7 | 25.3 [22.4, 28.3] |
| Gemini | 0.35 | Partial ties + Bonferroni | 0.600 | 0.75 | 1291 | 59.7 | 28.6 [26.1, 31.1] |
| Gemini | 0.35 | Partial ties + fallback | 0.410 | 1.00 | 1381 | 63.8 | 29.8 [27.4, 32.2] |
| Gemini | 0.40 | Partial ties + Bonferroni | 0.400 | 0.25 | 1546 | 71.5 | 33.4 [31.0, 35.8] |
| Gemini | 0.40 | Partial ties + fallback | 0.400 | 0.50 | 1697 | 78.5 | 37.1 [34.8, 39.5] |
| Sonnet | 0.20 | Partial ties + Bonferroni | — | — | 0 | 0.0 | — |
| Sonnet | 0.20 | Partial ties + fallback | — | — | 0 | 0.0 | — |
| Sonnet | 0.25 | Partial ties + Bonferroni | 0.900 | 0.75 | 44 | 2.0 | 20.5 [9.8, 35.3] |
| Sonnet | 0.25 | Partial ties + fallback | 0.900 | 0.75 | 44 | 2.0 | 20.5 [9.8, 35.3] |
| Sonnet | 0.30 | Partial ties + Bonferroni | 0.850 | 0.25 | 119 | 5.5 | 15.1 [9.2, 22.8] |
| Sonnet | 0.30 | Partial ties + fallback | 0.850 | 0.75 | 165 | 7.6 | 20.0 [14.2, 26.9] |
| Sonnet | 0.35 | Partial ties + Bonferroni | 0.760 | 1.00 | 234 | 10.8 | 21.4 [16.3, 27.2] |
| Sonnet | 0.35 | Partial ties + fallback | 0.710 | 1.00 | 389 | 18.0 | 27.5 [23.1, 32.2] |
| Sonnet | 0.40 | Partial ties + Bonferroni | 0.600 | 0.75 | 485 | 22.4 | 31.5 [27.4, 35.9] |
| Sonnet | 0.40 | Partial ties + fallback | 0.550 | 0.25 | 558 | 25.8 | 33.0 [29.1, 37.0] |
| DeepSeek | 0.20 | Partial ties + Bonferroni | — | — | 0 | 0.0 | — |
| DeepSeek | 0.20 | Partial ties + fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.25 | Partial ties + Bonferroni | — | — | 0 | 0.0 | — |
| DeepSeek | 0.25 | Partial ties + fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.30 | Partial ties + Bonferroni | — | — | 0 | 0.0 | — |
| DeepSeek | 0.30 | Partial ties + fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.35 | Partial ties + Bonferroni | — | — | 0 | 0.0 | — |
| DeepSeek | 0.35 | Partial ties + fallback | — | — | 0 | 0.0 | — |
| DeepSeek | 0.40 | Partial ties + Bonferroni | — | — | 0 | 0.0 | — |
| DeepSeek | 0.40 | Partial ties + fallback | — | — | 0 | 0.0 | — |

## Consistency across graders and allocations

Entries are paired changes in mean test answer rate, in percentage points, relative to the current method on the same test questions. Each allocation uses its own held-out sample; this table does not attribute differences between allocations solely to the selected rules. Main partial-tie randomization only.

| Model | alpha | Method | Primary, 50–50 | Second grader, 50–50 | Primary, 70–30 | Second grader, 70–30 |
|---|---:|---|---:|---:|---:|---:|
| Gemini | 0.20 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| Gemini | 0.20 | Ordered fallback | +0.03 | +0.03 | +0.02 | +0.02 |
| Gemini | 0.20 | Multi-start sequence | +0.38 | +0.38 | +0.04 | -0.13 |
| Gemini | 0.20 | Partial ties + Bonferroni | -0.15 | -0.15 | -0.31 | -0.31 |
| Gemini | 0.20 | Partial ties + fallback | -0.15 | -0.15 | -0.31 | -0.31 |
| Gemini | 0.25 | Holm | +0.00 | +0.05 | +0.14 | +0.11 |
| Gemini | 0.25 | Ordered fallback | +2.82 | +2.74 | +6.31 | +6.00 |
| Gemini | 0.25 | Multi-start sequence | +4.50 | +4.47 | +6.08 | +6.49 |
| Gemini | 0.25 | Partial ties + Bonferroni | -1.69 | -1.53 | -0.37 | -0.61 |
| Gemini | 0.25 | Partial ties + fallback | +1.77 | +1.86 | +6.02 | +5.48 |
| Gemini | 0.30 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| Gemini | 0.30 | Ordered fallback | +0.58 | +0.58 | +0.00 | +0.00 |
| Gemini | 0.30 | Multi-start sequence | +0.46 | +0.23 | +0.00 | +0.00 |
| Gemini | 0.30 | Partial ties + Bonferroni | +2.74 | +2.54 | +5.46 | +5.23 |
| Gemini | 0.30 | Partial ties + fallback | +9.00 | +8.62 | +11.40 | +10.72 |
| Gemini | 0.35 | Holm | +0.76 | +0.52 | +0.06 | +0.06 |
| Gemini | 0.35 | Ordered fallback | +2.35 | +2.58 | +0.06 | +0.08 |
| Gemini | 0.35 | Multi-start sequence | +2.13 | +2.33 | +0.06 | +0.08 |
| Gemini | 0.35 | Partial ties + Bonferroni | -0.71 | -1.20 | -0.03 | -0.09 |
| Gemini | 0.35 | Partial ties + fallback | +2.92 | +3.07 | +0.60 | +0.46 |
| Gemini | 0.40 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| Gemini | 0.40 | Ordered fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| Gemini | 0.40 | Multi-start sequence | +0.00 | +0.00 | +0.00 | +0.00 |
| Gemini | 0.40 | Partial ties + Bonferroni | +7.41 | +7.13 | +9.22 | +8.71 |
| Gemini | 0.40 | Partial ties + fallback | +13.58 | +13.07 | +14.97 | +14.83 |
| Sonnet | 0.20 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| Sonnet | 0.20 | Ordered fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| Sonnet | 0.20 | Multi-start sequence | +0.00 | +0.00 | +0.00 | +0.00 |
| Sonnet | 0.20 | Partial ties + Bonferroni | +0.00 | +0.00 | +0.00 | +0.00 |
| Sonnet | 0.20 | Partial ties + fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| Sonnet | 0.25 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| Sonnet | 0.25 | Ordered fallback | +0.06 | +0.06 | +0.00 | +0.00 |
| Sonnet | 0.25 | Multi-start sequence | +0.64 | +0.40 | +0.29 | +0.13 |
| Sonnet | 0.25 | Partial ties + Bonferroni | -0.04 | +0.00 | -0.13 | -0.05 |
| Sonnet | 0.25 | Partial ties + fallback | -0.04 | +0.00 | -0.11 | -0.05 |
| Sonnet | 0.30 | Holm | +0.01 | +0.06 | +0.01 | +0.09 |
| Sonnet | 0.30 | Ordered fallback | +0.15 | +0.22 | +0.18 | +0.34 |
| Sonnet | 0.30 | Multi-start sequence | +2.77 | +2.47 | +1.10 | +1.87 |
| Sonnet | 0.30 | Partial ties + Bonferroni | -0.82 | -1.11 | -1.45 | -1.56 |
| Sonnet | 0.30 | Partial ties + fallback | +0.05 | -0.63 | +0.24 | -0.34 |
| Sonnet | 0.35 | Holm | +0.08 | +0.10 | +0.36 | +0.31 |
| Sonnet | 0.35 | Ordered fallback | +4.31 | +3.92 | +3.66 | +5.30 |
| Sonnet | 0.35 | Multi-start sequence | +3.65 | +3.12 | +2.70 | +4.16 |
| Sonnet | 0.35 | Partial ties + Bonferroni | -0.65 | -0.06 | -0.40 | +0.27 |
| Sonnet | 0.35 | Partial ties + fallback | +4.37 | +4.08 | +4.53 | +6.06 |
| Sonnet | 0.40 | Holm | +0.18 | +0.33 | +0.06 | +0.07 |
| Sonnet | 0.40 | Ordered fallback | +1.98 | +2.08 | +0.91 | +0.40 |
| Sonnet | 0.40 | Multi-start sequence | +1.41 | +1.64 | +0.18 | +0.40 |
| Sonnet | 0.40 | Partial ties + Bonferroni | +0.23 | -0.19 | +0.61 | +0.04 |
| Sonnet | 0.40 | Partial ties + fallback | +4.66 | +4.18 | +4.60 | +3.41 |
| DeepSeek | 0.20 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.20 | Ordered fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.20 | Multi-start sequence | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.20 | Partial ties + Bonferroni | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.20 | Partial ties + fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.25 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.25 | Ordered fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.25 | Multi-start sequence | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.25 | Partial ties + Bonferroni | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.25 | Partial ties + fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.30 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.30 | Ordered fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.30 | Multi-start sequence | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.30 | Partial ties + Bonferroni | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.30 | Partial ties + fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.35 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.35 | Ordered fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.35 | Multi-start sequence | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.35 | Partial ties + Bonferroni | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.35 | Partial ties + fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.40 | Holm | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.40 | Ordered fallback | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.40 | Multi-start sequence | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.40 | Partial ties + Bonferroni | +0.00 | +0.00 | +0.00 | +0.00 |
| DeepSeek | 0.40 | Partial ties + fallback | +0.00 | +0.00 | +0.00 | +0.00 |

## Sensitivity to auxiliary randomization

Ranges below are across five preplanned auxiliary randomizations, each evaluated across the same 100 additional partitions. They describe sensitivity, not confidence intervals or independent datasets. Primary grader, 50–50 allocation.

| Model | alpha | Method | Mean answered (%), range | Mean expected answered (%), range | Median error (%), range | Qualifying partitions, range |
|---|---:|---|---:|---:|---:|---:|
| Gemini | 0.20 | Partial ties + Bonferroni | 0.0–0.1 | 0.0–0.1 | 17.1–18.5 | 0–1 |
| Gemini | 0.20 | Partial ties + fallback | 0.0–0.1 | 0.0–0.1 | 17.1–18.5 | 0–1 |
| Gemini | 0.25 | Partial ties + Bonferroni | 10.3–11.0 | 10.3–11.0 | 18.3–18.7 | 56–60 |
| Gemini | 0.25 | Partial ties + fallback | 13.9–14.4 | 13.9–14.4 | 20.1–21.1 | 56–60 |
| Gemini | 0.30 | Partial ties + Bonferroni | 31.0–31.3 | 31.0–31.3 | 20.9–22.5 | 100–100 |
| Gemini | 0.30 | Partial ties + fallback | 37.1–37.7 | 37.1–37.7 | 24.8–25.2 | 100–100 |
| Gemini | 0.35 | Partial ties + Bonferroni | 59.6–59.8 | 59.6–59.8 | 29.7–29.8 | 100–100 |
| Gemini | 0.35 | Partial ties + fallback | 63.2–63.4 | 63.2–63.4 | 30.1–30.1 | 100–100 |
| Gemini | 0.40 | Partial ties + Bonferroni | 70.2–70.8 | 70.2–70.7 | 33.7–33.8 | 100–100 |
| Gemini | 0.40 | Partial ties + fallback | 76.2–76.9 | 76.2–76.8 | 36.6–36.8 | 100–100 |
| Sonnet | 0.20 | Partial ties + Bonferroni | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| Sonnet | 0.20 | Partial ties + fallback | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| Sonnet | 0.25 | Partial ties + Bonferroni | 0.0–0.1 | 0.0–0.1 | 20.6–24.7 | 1–2 |
| Sonnet | 0.25 | Partial ties + fallback | 0.0–0.2 | 0.0–0.1 | 21.1–24.7 | 1–2 |
| Sonnet | 0.30 | Partial ties + Bonferroni | 3.4–4.1 | 3.4–4.1 | 22.5–23.1 | 37–47 |
| Sonnet | 0.30 | Partial ties + fallback | 4.3–4.9 | 4.3–4.9 | 24.4–25.8 | 37–47 |
| Sonnet | 0.35 | Partial ties + Bonferroni | 12.7–13.2 | 12.8–13.2 | 24.5–25.3 | 94–98 |
| Sonnet | 0.35 | Partial ties + fallback | 17.8–18.3 | 17.8–18.3 | 29.2–29.5 | 94–98 |
| Sonnet | 0.40 | Partial ties + Bonferroni | 22.4–22.6 | 22.4–22.6 | 31.3–31.6 | 100–100 |
| Sonnet | 0.40 | Partial ties + fallback | 27.0–27.3 | 27.0–27.3 | 34.1–35.1 | 100–100 |
| DeepSeek | 0.20 | Partial ties + Bonferroni | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.20 | Partial ties + fallback | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.25 | Partial ties + Bonferroni | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.25 | Partial ties + fallback | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.30 | Partial ties + Bonferroni | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.30 | Partial ties + fallback | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.35 | Partial ties + Bonferroni | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.35 | Partial ties + fallback | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.40 | Partial ties + Bonferroni | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |
| DeepSeek | 0.40 | Partial ties + fallback | 0.0–0.0 | 0.0–0.0 | —–— | 0–0 |

“Expected answered” averages each selected rule’s acceptance probabilities on the fixed test questions, integrating out test-time coins. Selection still uses realized independent coins. No fractional-count confidence intervals are used.

## Validation and interpretation

All 61,812 saved 50–50 baseline bounds and all 6,060 baseline selections/test evaluations across both allocations and graders were reproduced. Implementation checks cover integer binomial counts, empty lists, nested answer sets, recovery of the original rules at gamma=1, preservation of Bonferroni rejections where required, stopping and budget-transfer behavior, and absence of test-grade input to selection.

These are exploratory comparisons on a previously examined benchmark. Any later method choice should be stated explicitly and confirmed independently. In particular, selecting the best method separately for each model/alpha after seeing these outcomes is not covered by the individual methods’ guarantees. There is no need for observed test error to equal alpha; the objective is more answered questions at controlled underlying conditional error.

The methods use established exact binomial testing and multiple-testing arguments. See [Learn then Test, Sections 2.3.1–2.3.2](https://arxiv.org/html/2110.01052v5#S2.SS3). The theory note, all selections, and paired summaries are retained in the repository.
