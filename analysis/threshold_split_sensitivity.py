"""Compare fixed-grid threshold decisions under 50–50 and 70–30 allocations.

Read existing per-question selections and grades; no model or grading calls.
Only numpy and scipy are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260912
REPEATS = 100
ALPHAS = (.20, .25, .30, .35, .40)
DELTA = .05
MODELS = {'gemini35flash': 'Gemini', 'sonnet46': 'Sonnet', 'deepseekv32maas': 'DeepSeek'}
GRADERS = {'openai': 'GPT-5.6 Terra (primary)', 'gemini': 'Gemini 3.5 Flash (second)'}
ALLOCATIONS = {'50-50': 5, '70-30': 7}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_indices(n, seed, tenths):
    if tenths not in (5, 7):
        raise ValueError('Only the prespecified 50–50 and 70–30 allocations are supported')
    order = np.random.default_rng(seed).permutation(n)
    cut = n * tenths // 10
    return order[:cut], order[cut:], order[n * 7 // 10:]


def bounds_at(u, z, grid):
    selected = u[:, None] >= grid[None, :]
    n = selected.sum(axis=0)
    k = (selected * (1 - z[:, None])).sum(axis=0)
    upper = np.ones(len(grid))
    valid = k < n
    upper[valid] = beta.ppf(1 - DELTA / len(grid), k[valid] + 1, n[valid] - k[valid])
    return n, k, upper


def choose(grid, n, upper, alpha):
    qualifying = np.flatnonzero((n > 0) & (upper <= alpha))
    return int(qualifying[0]) if len(qualifying) else None


def evaluate(u, z, threshold):
    selected = np.zeros(len(u), dtype=bool) if threshold is None else u >= threshold
    n = int(selected.sum())
    k = int((1-z[selected]).sum())
    return {'n_test': len(u), 'n_answered': n, 'n_incorrect': k,
            'answer_rate': n / len(u), 'observed_error': k / n if n else None,
            'error_lo': (0. if k == 0 else float(beta.ppf(.025, k, n-k+1))) if n else None,
            'error_hi': (1. if k == n else float(beta.ppf(.975, k+1, n-k))) if n else None}


def equal(value, reference):
    if value is None:
        return reference == ''
    return abs(float(value) - float(reference)) < 1e-11


def load_grader(grader, analysis_dir=None):
    analysis_dir = Path(analysis_dir) if analysis_dir else ROOT / 'results/exploratory'
    folder = analysis_dir / f'alpha_risk_control_{grader}'
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['seed'] == SEED and meta['additional_splits'] == REPEATS and meta['delta'] == DELTA
    for entry in meta['inputs']:
        assert sha(ROOT/entry['file']) == entry['sha256'], 'Source-response hash changed'
    raw = [r for r in read_csv(folder/'primary_inputs.csv') if r['representation'] == 'raw']
    frames = {}
    for model in MODELS:
        rows = sorted((r for r in raw if r['model_key'] == model), key=lambda r:r['question_id'])
        ids = [r['question_id'] for r in rows]
        assert len(ids) == len(set(ids)) == 4326
        u = np.array([float(r['u']) for r in rows])
        z = np.array([int(r['z']) for r in rows])
        assert np.all((u == -1) | ((u >= 0) & (u <= 1))) and np.all(np.isin(z, [0, 1]))
        assert np.all(z[u == -1] == 0)
        frames[model] = (ids, u, z)
    references = {}
    for r in read_csv(folder/'all_results.csv'):
        if r['representation'] == 'raw' and r['method'] == 'fixed_full':
            alpha = float(r['alpha'])
            if any(abs(alpha-a)<1e-12 for a in ALPHAS):
                key = (r['model_key'], int(r['seed']), round(alpha, 12))
                assert key not in references
                references[key] = r
    bound_refs = {}
    for r in read_csv(folder/'all_bounds.csv'):
        if r['representation'] == 'raw' and r['method'] == 'fixed_full':
            key = (r['model_key'], int(r['seed']), float(r['threshold']))
            assert key not in bound_refs
            bound_refs[key] = r
    assert len(references) == 3 * 101 * 5 and len(bound_refs) == 3 * 101 * 102
    return frames, np.array(meta['fixed_grid']), references, bound_refs, {
        'grader': grader, 'source_metadata_sha256': sha(folder/'metadata.json'),
        'per_question_inputs_sha256': sha(folder/'primary_inputs.csv'),
        'original_results_sha256': sha(folder/'all_results.csv'),
        'original_bounds_sha256': sha(folder/'all_bounds.csv'), 'responses': meta['inputs']}


def summarize(rows):
    summaries, paired = [], []
    for grader in GRADERS:
        for model in MODELS:
            for alpha in ALPHAS:
                groups = {}
                for allocation in ALLOCATIONS:
                    group = [r for r in rows if r['grader']==grader and r['model_key']==model
                             and r['alpha']==alpha and r['allocation']==allocation and r['seed']!=SEED]
                    assert len(group)==REPEATS
                    groups[allocation] = {r['seed']:r for r in group}
                    supported = [r for r in group if r['supported']]
                    answering = [r for r in group if r['n_answered']]
                    errors = [r['observed_error'] for r in answering]
                    widths = [r['error_hi']-r['error_lo'] for r in answering]
                    summaries.append({'grader':grader,'model_key':model,'alpha':alpha,'allocation':allocation,
                        'n_splits':len(group), 'n_supported':len(supported),
                        'mean_answer_rate':float(np.mean([r['answer_rate'] for r in group])),
                        'mean_common_answer_rate':float(np.mean([r['common_answer_rate'] for r in group])),
                        'median_threshold_when_supported':float(np.median([r['threshold'] for r in supported])) if supported else None,
                        'n_splits_with_test_answers':len(answering),
                        'median_test_error_when_answering':float(np.median(errors)) if errors else None,
                        'test_error_q25':float(np.quantile(errors,.25)) if errors else None,
                        'test_error_q75':float(np.quantile(errors,.75)) if errors else None,
                        'mean_test_interval_width':float(np.mean(widths)) if widths else None,
                        'n_test_errors_above_alpha':sum(e>alpha for e in errors),
                        'median_minimum_supported_alpha':float(np.median([r['minimum_supported_alpha'] for r in group]))})
                pairs = [(groups['50-50'][seed],groups['70-30'][seed]) for seed in sorted(groups['50-50'])]
                both = [(a,b) for a,b in pairs if a['supported'] and b['supported']]
                common_answering = [(a,b) for a,b in pairs if a['common_n_answered'] and b['common_n_answered']]
                deltas = [b['common_answer_rate']-a['common_answer_rate'] for a,b in pairs]
                paired.append({'grader':grader,'model_key':model,'alpha':alpha,
                    'n_support_gained':sum(not a['supported'] and b['supported'] for a,b in pairs),
                    'n_support_lost':sum(a['supported'] and not b['supported'] for a,b in pairs),
                    'n_both_supported':len(both),
                    'n_70_lower_threshold_when_both_supported':sum(b['threshold']<a['threshold'] for a,b in both),
                    'n_70_higher_threshold_when_both_supported':sum(b['threshold']>a['threshold'] for a,b in both),
                    'mean_common_answer_rate_change':float(np.mean(deltas)),
                    'common_answer_rate_change_q25':float(np.quantile(deltas,.25)),
                    'common_answer_rate_change_q75':float(np.quantile(deltas,.75)),
                    'n_both_answer_on_common_test':len(common_answering),
                    'mean_common_error_change_when_both_answer':float(np.mean([b['common_observed_error']-a['common_observed_error'] for a,b in common_answering])) if common_answering else None})
    return summaries, paired


def pct(v):
    return '—' if v is None else f'{100*v:.1f}'


def make_report(out, rows, summaries, paired):
    lines = ['# Sensitivity of threshold decisions to a 70–30 split', '',
        'The primary 50–50 analysis is unchanged. This comparison uses 3,028 questions for threshold selection and 1,298 for testing, versus 2,163 and 2,163. Both allocations use the same question permutation in each split, the same 102 thresholds, confidence 95%, and error limits from 0.20 to 0.40. No new model responses or grades were collected.', '',
        'All 61,812 original 50–50 bounds and all 3,030 selections/test evaluations were reproduced under both graders across the original split and 100 additional splits. Results below summarize the additional 100 separately from the original split.', '',
        '## Repeated-split comparison', '',
        'Qualified counts splits with at least one supported threshold. Answered is the mean percentage answered on the **same 1,298 test questions** under both rules, including zero for unsupported rules. Thus the answer-rate comparison is not due to different test questions. The means are descriptive because the splits overlap. Displayed rates and changes are rounded separately; changes are calculated before rounding.', '']
    def summary(grader, model, alpha, allocation):
        return next(r for r in summaries if r['grader']==grader and r['model_key']==model
                    and r['alpha']==alpha and r['allocation']==allocation)
    findings = ['## Main findings', '']
    for model, alpha in [('gemini35flash',.25), ('sonnet46',.30)]:
        a,b = [summary('openai',model,alpha,k) for k in ALLOCATIONS]
        aa,bb = [summary('gemini',model,alpha,k) for k in ALLOCATIONS]
        findings += [f"**{MODELS[model]} at α = {alpha:.2f}.** With the primary grader, qualifying splits increase from {a['n_supported']}/100 to {b['n_supported']}/100, and mean answered percentage on common test questions increases from {pct(a['mean_common_answer_rate'])}% to {pct(b['mean_common_answer_rate'])}%. With the second grader, qualifying splits increase from {aa['n_supported']}/100 to {bb['n_supported']}/100. This is the clearest benefit of providing more threshold-selection data.", '']
    findings += [
        '**Some decisions are unchanged.** At α = 0.30 and 0.40, Gemini selects exactly the same threshold under both allocations in every additional split, under both graders. Tighter bounds do not always change the selected decision rule.', '',
        '**Low error limits remain fragile.** At α = 0.20, Gemini qualifies in only 2/100 splits under either grader with 70–30, and the observed test errors in both qualifying splits exceed 20%. Sonnet at α = 0.25 qualifies in only 7/100 splits with the primary grader and 2/100 with the second. DeepSeek has no qualifying threshold at any tested error limit in any split under either allocation or grader.', '',
        '**More answers can come with higher error.** In the original split, at α = 0.30, Sonnet’s threshold changes from 0.857 to 0.760 under the primary grader. On the same 1,298 test questions, it answers 57 versus 143 questions (4.4% versus 11.0%), with errors of 19.3% versus 24.5%. Both observed errors are below 30%; the second rule answers more questions. Across additional splits at α = 0.35, Sonnet’s median error on each allocation’s own test set rises from 22.2% to 28.7% as more questions are answered.', '',
        '**The evaluation tradeoff is visible.** For Gemini at α = 0.30, where the selected threshold is unchanged, the average width of the pointwise 95% test-error interval increases from 6.6 to 8.6 percentage points with the smaller test sample. This is a loss of evaluation precision, not a change in the decision rule.', '',
        '**Assessment.** Keep 50–50 as the primary analysis and use 70–30 as sensitivity evidence. The results show that selection-sample size materially affects whether a useful threshold can be supported, especially for Gemini at 0.25 and Sonnet at 0.30. They do not establish that 70–30 is universally preferable or that test error should equal the chosen limit.', '',
        'Occasional support losses also occur. For example, Gemini at α = 0.25 gains support in 31 primary-grader splits but loses it in one; Sonnet at α = 0.30 gains support in 36 and loses it in one. All comparisons are retained in the saved outputs.', '']
    insert = lines.index('## Repeated-split comparison')
    lines[insert:insert] = findings
    for grader,label in GRADERS.items():
        lines += [f'### {label}', '', '| Model | α | Qualified: 50–50 | Qualified: 70–30 | Answered: 50–50 (%) | Answered: 70–30 (%) | Change (pp) |', '|---|---:|---:|---:|---:|---:|---:|']
        for model,name in MODELS.items():
            for alpha in ALPHAS:
                a,b=[next(r for r in summaries if r['grader']==grader and r['model_key']==model and r['alpha']==alpha and r['allocation']==allocation) for allocation in ALLOCATIONS]
                delta=b['mean_common_answer_rate']-a['mean_common_answer_rate']
                lines.append(f"| {name} | {alpha:.2f} | {a['n_supported']}/100 | {b['n_supported']}/100 | {pct(a['mean_common_answer_rate'])} | {pct(b['mean_common_answer_rate'])} | {100*delta:+.1f} |")
        lines += ['', '| Model | α | Median test error: 50–50 (%) | Median test error: 70–30 (%) | Test errors above α: 50–50 | Test errors above α: 70–30 |', '|---|---:|---:|---:|---:|---:|']
        for model,name in MODELS.items():
            for alpha in ALPHAS:
                a,b=[next(r for r in summaries if r['grader']==grader and r['model_key']==model and r['alpha']==alpha and r['allocation']==allocation) for allocation in ALLOCATIONS]
                lines.append(f"| {name} | {alpha:.2f} | {pct(a['median_test_error_when_answering'])} | {pct(b['median_test_error_when_answering'])} | {a['n_test_errors_above_alpha']}/{a['n_splits_with_test_answers']} | {b['n_test_errors_above_alpha']}/{b['n_splits_with_test_answers']} |")
        lines += ['', 'Conditional-error summaries use each allocation’s own test set and only splits with returned answers; 0/0 denotes no evaluable error rate. These test-error fractions are not estimates of the guarantee’s failure probability.', '']
    lines += ['## Original split: numerical comparison', '', 'The bounds below are from threshold selection; errors and exact pointwise 95% intervals are from each allocation’s own test questions. Thresholds are rounded to three decimals.', '']
    for grader,label in GRADERS.items():
        lines += [f'### {label}', '', '| Model | α | Allocation | Threshold | Selection bound (%) | Test answered n (%) | Test error (%) [95% interval] |', '|---|---:|---|---:|---:|---:|---|']
        for model,name in MODELS.items():
            for alpha in ALPHAS:
                for allocation in ALLOCATIONS:
                    r=next(r for r in rows if r['seed']==SEED and r['grader']==grader and r['model_key']==model and r['alpha']==alpha and r['allocation']==allocation)
                    threshold='—' if r['threshold'] is None else f"{r['threshold']:.3f}"
                    error='—' if r['observed_error'] is None else f"{pct(r['observed_error'])} [{pct(r['error_lo'])}, {pct(r['error_hi'])}]"
                    lines.append(f"| {name} | {alpha:.2f} | {allocation} | {threshold} | {pct(r['selection_upper'])} | {r['n_answered']:,} ({pct(r['answer_rate'])}) | {error} |")
        lines.append('')
    lines += ['## Interpretation and reproducibility', '',
        '- The objective is to support an upper limit on conditional error while answering as many questions as possible. It does not target equality between test error and α.',
        '- More selection data can tighten bounds, but support and the selected threshold can move in either direction in a particular split. The saved paired comparisons include both gains and losses.',
        '- Evaluating both rules on common test questions controls test-set composition. The rules can still answer different subsets of those questions.',
        '- Smaller test sets can make test-error estimates less precise at a fixed threshold. Actual precision also depends on how many answers the selected rule returns.',
        '- These are overlapping retrospective splits of an already examined benchmark, not independent external replications. The individual risk guarantee does not automatically extend to selecting the allocation with the best observed result.', '',
        'Reproduce from the repository root with `python3 analysis/threshold_split_sensitivity.py` (requires NumPy and SciPy).', '',
        'Saved outputs: `all_results.csv`, `all_bounds.csv`, `primary_results.csv`, `split_summary.csv`, `paired_comparisons.csv`, `primary_assignments.csv`, and `metadata.json`.', '']
    (out/'report.md').write_text('\n'.join(lines))


def run(out, analysis_dir=None):
    frames_by_grader, inputs, refs, bound_refs = {}, [], {}, {}
    grid = ids = None
    for grader in GRADERS:
        frames,g,ref,bref,metadata = load_grader(grader, analysis_dir)
        if grid is not None: np.testing.assert_array_equal(g,grid)
        grid=g
        for model,(model_ids,u,z) in frames.items():
            if ids is not None: assert model_ids==ids
            ids=model_ids
            if grader=='gemini':np.testing.assert_array_equal(u,frames_by_grader['openai'][model][1])
        frames_by_grader[grader]=frames;refs[grader]=ref;bound_refs[grader]=bref;inputs.append(metadata)
    assert len(grid)==102 and np.all(np.diff(grid)>0)
    results, bands, assignments = [], [], []
    for seed in range(SEED,SEED+REPEATS+1):
        for allocation,tenths in ALLOCATIONS.items():
            sel,test,common=split_indices(len(ids),seed,tenths)
            assert len(set(sel)&set(test))==0 and set(common)<=set(test)
            if seed==SEED:
                for i,q in enumerate(ids):
                    assignments.append({'question_id':q,'allocation':allocation,
                                        'split':'selection' if i in set(sel) else 'test',
                                        'common_test':i in set(common)})
            for grader in GRADERS:
                for model in MODELS:
                    _,u,z=frames_by_grader[grader][model]
                    n,k,upper=bounds_at(u[sel],z[sel],grid)
                    for j,t in enumerate(grid):
                        bands.append({'grader':grader,'model_key':model,'allocation':allocation,'seed':seed,
                                      'threshold':float(t),'n_selection':len(sel),'n_answered':int(n[j]),
                                      'n_incorrect':int(k[j]),'risk_upper':float(upper[j])})
                        if allocation=='50-50':
                            ref=bound_refs[grader][(model,seed,float(t))]
                            assert int(n[j])==int(ref['n_cert_answered']) and int(k[j])==int(ref['k_cert_incorrect'])
                            assert equal(upper[j],ref['risk_upper'])
                    minimum=float(upper[n>0].min()) if np.any(n>0) else 1.
                    previous=float('inf')
                    for alpha in ALPHAS:
                        j=choose(grid,n,upper,alpha)
                        t=float(grid[j]) if j is not None else None
                        assert (float('inf') if t is None else t)<=previous
                        previous=float('inf') if t is None else t
                        row={'grader':grader,'model_key':model,'allocation':allocation,'seed':seed,'alpha':alpha,
                             'n_selection':len(sel),'supported':j is not None,'threshold':t,
                             'selection_n_answered':int(n[j]) if j is not None else 0,
                             'selection_n_incorrect':int(k[j]) if j is not None else 0,
                             'selection_upper':float(upper[j]) if j is not None else None,
                             'minimum_supported_alpha':minimum,**evaluate(u[test],z[test],t),
                             **{'common_'+k:v for k,v in evaluate(u[common],z[common],t).items()}}
                        if allocation=='50-50':
                            ref=refs[grader][(model,seed,round(alpha,12))]
                            assert row['supported']==(ref['supported']=='True')
                            for field in ['threshold','n_test','n_answered','n_incorrect','answer_rate','observed_error','error_lo','error_hi']:
                                assert equal(row[field],ref[field]),(grader,model,seed,alpha,field)
                            for field,old in [('selection_upper','certification_upper'),('selection_n_answered','n_cert_answered'),('selection_n_incorrect','k_cert_incorrect')]:
                                assert equal(row[field],ref[old])
                        results.append(row)
        if (seed-SEED)%20==0:print(f'Completed split {seed-SEED+1}/101',flush=True)
    summary,paired=summarize(results)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('all_results',results),('all_bounds',bands),('primary_results',[r for r in results if r['seed']==SEED]),('split_summary',summary),('paired_comparisons',paired),('primary_assignments',assignments)]:
        write_csv(out/(name+'.csv'),rows)
    meta={'seed':SEED,'additional_splits':REPEATS,'alphas':ALPHAS,'delta':DELTA,'grid':grid.tolist(),
          'selection_sizes':{'50-50':2163,'70-30':3028},'test_sizes':{'50-50':2163,'70-30':1298},
          'primary_grader':'openai','representation':'raw','common_test_size':1298,
          'verification':{'original_bounds_reproduced':61812,'original_result_rows_reproduced':3030},
          'inputs':inputs,'source_sha256':sha(__file__),
          'software':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    make_report(out,results,summary,paired)
    print(f'Wrote {len(results)} result rows and {len(bands)} bounds to {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/exploratory/threshold_split_sensitivity')
    run(parser.parse_args().out_dir)
