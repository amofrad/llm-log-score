"""Export every manuscript table as a numbered CSV and a LaTeX tabular snippet.

Numerical rows come from the analysis outputs. The two descriptive tables
use versioned templates. Snippets use the manuscript's macros and package
settings; they are not standalone LaTeX documents.
"""
from pathlib import Path
import argparse
import json
import re
import shutil

import pandas as pd
from common import MODEL_ORDER, RUNS

TEMPLATES = Path(__file__).with_name('table_templates')
SOURCES = {
    'Table1_DecisionOutcomes': 'Table1_DecisionOutcomes.csv',
    'TableS1_LogLoss': 'theory_audit_realized_log_loss.csv',
    'TableS2_SetOutcomeIntervals': 'set_outcome_intervals.csv',
    'TableS3_MatchedAbstention': 'matched_abstention_gaps.csv',
    'TableS4_IdealListExtent': 'TableS4_IdealListExtent.csv',
    'TableS5_TokenCosts': 'token_costs.csv',
    'TableS6_IDKReliability': 'theory_audit_residual_reliability.csv',
    'TableS7_TopReliability': 'theory_audit_top_reliability.csv',
    'TableS8_ThresholdCalibration': 'TableS8_ThresholdCalibration.csv',
    'TableS9_RiskControl': 'TableS9_RiskControl.csv',
    'TableS10_SplitSensitivity': 'TableS10_SplitSensitivity.csv',
}


def export(source, primary=True):
    source = Path(source)
    target = source / 'tables'
    target.mkdir(exist_ok=True)
    bodies = (source / 'table_bodies.txt').read_text()
    def body(title):
        block = bodies.split('==== TABLE '+title+' BODY ====\n', 1)[1].split('\n====', 1)[0]
        block = block.split('\nS1 4dp', 1)[0].strip()
        return block.removesuffix(r'\midrule').rstrip()
    rows = {'Table1_DecisionOutcomes': body('1'), 'TableS1_LogLoss': body('S1'),
            'TableS3_MatchedAbstention': body('S3'),
            'TableS6_IDKReliability': body('S6 (residual)'),
            'TableS7_TopReliability': body('S7 (top)')}
    for stem in ('TableS2_SetOutcomeIntervals', 'TableS9_RiskControl', 'TableS10_SplitSensitivity'):
        rows[stem] = (source / (stem+'_rows.tex')).read_text().strip()
    tokens = pd.read_csv(source/'token_costs.csv').set_index('model')
    rows['TableS5_TokenCosts'] = '\n'.join(
        name + ' & ' + ' & '.join(f'{tokens.loc[RUNS[k]["label"], field]:,.0f}' for k in MODEL_ORDER) + r'\\'
        for name, field in [('RBD','log')]+[(f'EPP, $L={L}$',f'penalty_L{L}') for L in (0,3,6)])
    threshold = pd.read_csv(source/'TableS8_ThresholdCalibration.csv')
    parts = []
    for k in MODEL_ORDER:
        if parts: parts.append(r'\midrule')
        for i, (_,r) in enumerate(threshold[threshold.model_key==k].sort_values('L').iterrows()):
            name = r'\textbf{'+r.model+'}' if i == 0 else ''
            parts.append(name + f' & {int(r.L)} & {int(r.n_answered):,} & {100*r.reported_error:.1f}'
                         f' & {100*r.observed_error:.1f} & ${100*r.error_gap:.1f}$'
                         f' $[{100*r.error_gap_lo:.1f}, {100*r.error_gap_hi:.1f}]$'+r'\\')
    rows['TableS8_ThresholdCalibration'] = '\n'.join(parts)
    if primary:
        ideal = pd.read_csv(source/'TableS4_IdealListExtent.csv')
        rows['TableS4_IdealListExtent'] = '\n'.join(
            f'${r.rho:g}$ & {r.mean_optimal_list_size:.3f} & {r.all_idk_rate:.3f}'+r'\\'
            for _,r in ideal.iterrows())
    manifest = []
    for template in sorted(TEMPLATES.glob('*.tex')):
        stem = template.stem
        if stem == 'TableS4_IdealListExtent' and not primary: continue
        text = template.read_text()
        if '@ROWS@\n' in text:
            text = text.replace('@ROWS@\n', rows[stem]+'\n')
        (target/template.name).write_text(text)
        if stem in SOURCES:
            shutil.copy2(source/SOURCES[stem],target/(stem+'.csv'))
        manifest.append(dict(table=stem, latex=template.name, data=stem+'.csv',
                             source=SOURCES.get(stem, 'descriptive table template')))
    pd.DataFrame([dict(model=RUNS[k]['label'], api_identifier=RUNS[k]['model_id']) for k in MODEL_ORDER]).to_csv(
        target/'TableS11_Models.csv',index=False)
    # Preserve the exact comparison wording in a plain-data export as well.
    text=(TEMPLATES/'Table2_MethodComparison.tex').read_text()
    comparison=[]
    def strip(s):
        s=s.replace(r'\raggedright','').replace(r'\par','').strip()
        while True:
            new=re.sub(r'\\(?:textcolor\{black\}|textbf)\{([^{}]*)\}',r'\1',s)
            if new == s:return s
            s=new
    for line in text.splitlines():
        if r'\tabularnewline' in line:
            cells=line.replace(r'\tabularnewline','').split('&')
            comparison.append(dict(aspect=strip(cells[0]),RLCR=strip(cells[1]),RBD=strip(cells[2])))
    pd.DataFrame(comparison).to_csv(target/'Table2_MethodComparison.csv',index=False)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'README.md').write_text(
        '# Paper tables\n\nNumbering follows the current main text and SI Appendix. '
        'CSV files retain analysis precision and may include diagnostic columns or additional rows. '
        'LaTeX snippets select and round the displayed values, using the paper’s table layouts. '
        'Table 1 uses largest-remainder rounding so each three-outcome group totals 1.000.\n\n'
        'Rebuild with `python analysis/reproduce_all.py`, or export already computed results with '
        '`python analysis/paper_tables.py`. The two descriptive tables use versioned templates '
        'in `analysis/table_templates/`. Snippets require the manuscript’s macros and packages.\n')
    print(f'Exported {len(manifest)} manuscript tables to {target}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'results/figures')
    p.add_argument('--grader',choices=['openai','gemini'],default='openai')
    args=p.parse_args();export(args.source,primary=args.grader=='openai')
