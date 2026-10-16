"""Export numerical results and model information as numbered table CSVs.

Numerical tables retain the analysis outputs in one canonical location.
The model table uses the shared model configuration.
"""
from pathlib import Path
import argparse
import shutil

import pandas as pd
from common import MODEL_ORDER, RUNS

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
    exported = 0
    stems = sorted(set(SOURCES) | {'TableS11_Models'})
    for stem in stems:
        if stem == 'TableS4_IdealListExtent' and not primary: continue
        if stem in SOURCES:
            generated = source / SOURCES[stem]
            destination = target / (stem + '.csv')
            if generated.exists():
                shutil.move(generated, destination)
            elif not destination.exists():
                raise FileNotFoundError(generated)
        else:
            pd.DataFrame([dict(model=RUNS[k]['label'], api_identifier=RUNS[k]['model_id'])
                          for k in MODEL_ORDER]).to_csv(target/(stem+'.csv'),index=False)
        exported += 1
    print(f'Exported {exported} manuscript tables to {target}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'results/figures')
    p.add_argument('--grader',choices=['openai','gemini'],default='openai')
    args=p.parse_args();export(args.source,primary=args.grader=='openai')
