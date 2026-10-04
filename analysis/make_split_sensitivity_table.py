"""Generate SI Table S10 from the saved split-allocation sensitivity analysis.

The table compares both selected rules on the common 1,298 test questions,
using the primary grader. CSV summaries retain both graders. Error percentiles
are recomputed from saved partition results; no model calls or grading occur.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'gemini35flash':'Gemini','sonnet46':'Sonnet','deepseekv32maas':'DeepSeek'}
ALPHAS = (.20,.25,.30,.35,.40)


def summarize_errors(rows):
    errors=[]
    for row in rows:
        n=int(row['common_n_answered'])
        if n:
            error=float(row['common_observed_error'])
            assert math.isclose(error,int(row['common_n_incorrect'])/n)
            errors.append(error)
        else:
            assert row['common_observed_error']==''
    if not errors:
        return {'n_partitions_with_test_answers':0,'median_common_test_error':None,
                'common_test_error_q25':None,'common_test_error_q75':None}
    median=statistics.median(errors)
    q25,_,q75=statistics.quantiles(errors,n=4,method='inclusive') if len(errors)>1 else [median]*3
    return {'n_partitions_with_test_answers':len(errors),'median_common_test_error':median,
            'common_test_error_q25':q25,'common_test_error_q75':q75}


def generate(source, out, primary_grader='openai'):
    if primary_grader not in ('openai','gemini'):
        raise ValueError('Unknown grader: '+primary_grader)
    meta=json.loads((source/'metadata.json').read_text())
    assert meta['additional_splits']==100 and meta['common_test_size']==1298
    assert meta['verification']['original_result_rows_reproduced']==3030
    with (source/'split_summary.csv').open() as stream: rows=list(csv.DictReader(stream))
    lookup={(r['model_key'],float(r['alpha']),r['grader'],r['allocation']):r for r in rows}
    assert len(lookup)==60
    groups={key:[] for key in lookup}
    with (source/'all_results.csv').open() as stream:
        for row in csv.DictReader(stream):
            if int(row['seed'])==meta['seed']:continue
            key=(row['model_key'],float(row['alpha']),row['grader'],row['allocation'])
            groups[key].append(row)
    assert all(len(group)==len({r['seed'] for r in group})==100 for group in groups.values())
    records=[]
    for model in MODELS:
        for alpha in ALPHAS:
            record={'model_key':model,'alpha':alpha,'display_grader':primary_grader}
            for grader in ['openai','gemini']:
                pair=[lookup[model,alpha,grader,a] for a in ['50-50','70-30']]
                assert all(int(r['n_splits'])==100 for r in pair)
                for allocation,r in zip(['50-50','70-30'],pair):
                    group=groups[model,alpha,grader,allocation]
                    assert all(int(x['common_n_test'])==1298 for x in group)
                    summary=summarize_errors(group)
                    assert summary['n_partitions_with_test_answers']==int(r['n_supported'])
                    assert math.isclose(statistics.mean(float(x['common_answer_rate']) for x in group),
                                        float(r['mean_common_answer_rate']),abs_tol=1e-12)
                    record[f'{grader}_{allocation}_n_supported']=int(r['n_supported'])
                    record[f'{grader}_{allocation}_mean_common_answer_rate']=float(r['mean_common_answer_rate'])
                    record.update({f'{grader}_{allocation}_{key}':value for key,value in summary.items()})
            records.append(record)
    out.mkdir(parents=True,exist_ok=True)
    stem='TableS10_SplitSensitivity'
    with (out/(stem+'.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('Generated '+str(out/(stem+'.csv')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/exploratory/threshold_split_sensitivity')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/figures')
    parser.add_argument('--grader',choices=('openai','gemini'),default='openai',
                        help='Grader displayed in Table S10; the CSV retains both graders.')
    args=parser.parse_args();generate(args.source,args.out_dir,args.grader)
