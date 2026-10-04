"""Export every manuscript table as a numbered CSV.

Numerical tables retain the analysis outputs. The method comparison uses a
versioned CSV, and the model table uses the shared model configuration.
"""
from pathlib import Path
import argparse
import json
import shutil

import pandas as pd
from common import MODEL_ORDER, RUNS

TABLE_DATA = Path(__file__).with_name('table_data')
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
    manifest = []
    stems = sorted(set(SOURCES) | {'Table2_MethodComparison', 'TableS11_Models'})
    for stem in stems:
        if stem == 'TableS4_IdealListExtent' and not primary: continue
        if stem in SOURCES:
            shutil.copy2(source/SOURCES[stem],target/(stem+'.csv'))
            origin = SOURCES[stem]
        elif stem == 'Table2_MethodComparison':
            shutil.copy2(TABLE_DATA/(stem+'.csv'),target/(stem+'.csv'))
            origin = 'analysis/table_data/'+stem+'.csv'
        else:
            pd.DataFrame([dict(model=RUNS[k]['label'], api_identifier=RUNS[k]['model_id'])
                          for k in MODEL_ORDER]).to_csv(target/(stem+'.csv'),index=False)
            origin = 'analysis/common.py model configuration'
        manifest.append(dict(table=stem, data=stem+'.csv', source=origin))
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'README.md').write_text(
        '# Paper tables\n\nNumbering follows the current main text and SI Appendix. '
        'CSV files retain analysis precision and may include diagnostic columns or additional rows. '
        'Table 1 includes display columns with largest-remainder rounding so each '
        'three-outcome group totals 1.000.\n\n'
        'Rebuild with `python analysis/reproduce_all.py`, or export already computed results with '
        '`python analysis/paper_tables.py`. Table 2 uses the versioned comparison CSV '
        'in `analysis/table_data/`; Table S11 uses the shared model configuration.\n')
    print(f'Exported {len(manifest)} manuscript tables to {target}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'results/figures')
    p.add_argument('--grader',choices=['openai','gemini'],default='openai')
    args=p.parse_args();export(args.source,primary=args.grader=='openai')
