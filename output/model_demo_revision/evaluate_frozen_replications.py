"""Score every predeclared population with the primary frozen model/preprocessing.

No model fitting, coefficient tuning or seed selection occurs in this script.
The primary run's policy metrics must reproduce the main pipeline exactly.
"""
from pathlib import Path
import sys,json,hashlib
import joblib,numpy as np,pandas as pd
import yaml
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/model_demo_revision'
sys.path.insert(0,str(ROOT/'part2/src'))
from _features import add_indicator_cols,apply_imputation_values
from _rule_baseline import apply_rule_baseline
from _review_bootstrap import rank_contracts
rec=json.loads((ROOT/'part2/outputs/model_selection_record.json').read_text())
artifact=ROOT/'part2/models'/rec['refit']['artifact'];model=joblib.load(artifact)
features=rec['selected_model']['num_features']+rec['selected_model']['cat_features']
impute=json.loads((ROOT/'part2/outputs/imputation_values.json').read_text())
policies={'health_rules':'rule_score','commercial_value':'acv','model_score':'model_score','model_x_acv':'model_x_acv'}
rows=[];test_metrics=[];counts=[]
for seed,root in [(137,ROOT)]+[(s,OUT/'replications'/f'seed_{s}') for s in (991,2027)]:
    reference_cfg=yaml.safe_load((ROOT/'config/simulation.yaml').read_text())
    run_cfg=yaml.safe_load((root/'config/simulation.yaml').read_text())
    assert run_cfg['seed']==seed
    run_cfg['seed']=reference_cfg['seed']
    assert run_cfg==reference_cfg, f'Unexpected scenario difference in seed {seed}'
    for source in ['src/generate_data.py','src/ingest_clean.py','src/integrate.py','part2/src/build_features.py','part2/src/make_splits.py']:
        assert (root/source).read_bytes()==(ROOT/source).read_bytes(), (seed,source)
    c=pd.read_csv(root/'part2/outputs/split_cohort_full.csv',low_memory=False)
    for col in ['term_end','pred_date','label_available_at']:c[col]=pd.to_datetime(c[col])
    fit=c[c.split.isin(['train','model_selection'])]
    assert fit.label_available_at.max()<=pd.Timestamp('2024-10-01')
    final=c[c.split.eq('final_test')].copy()
    final=apply_imputation_values(add_indicator_cols(final),impute)
    scores=model.predict_proba(final[features])[:,1];rules=apply_rule_baseline(final)
    for name,s in [('model_score',scores),('health_rules',rules)]:
        test_metrics.append(dict(seed=seed,policy=name,n=len(final),positives=int(final.churned.sum()),prevalence=float(final.churned.mean()),ap=float(average_precision_score(final.churned,s)),roc_auc=float(roc_auc_score(final.churned,s))))
    D=pd.Timestamp('2026-04-01');D90=D+pd.Timedelta(days=90)
    d=c[c.offering_type.eq('subscription') & c.term_end.gt(D) & c.term_end.le(D90)].copy()
    assert d.churned.notna().all()
    if 'decision_known_at_pred_date' in d:d=d[~d.decision_known_at_pred_date.fillna(False).astype(bool)]
    events=pd.read_csv(root/'data/cleaned/contract_events_clean.csv',low_memory=False)
    known=events.loc[events.event_type.isin(['cancellation','renewal','renewal_expansion','renewal_contraction']) & pd.to_datetime(events.event_date).lt(D),'expiring_contract_id'].dropna()
    d=d[~d.contract_id.isin(known)].copy()
    d=apply_imputation_values(add_indicator_cols(d),impute)
    d['model_score']=model.predict_proba(d[features])[:,1]
    d['rule_score']=apply_rule_baseline(d);d['model_x_acv']=d.model_score*d.acv
    n_churn_accounts=d.loc[d.churned.eq(1),'company_id'].nunique()
    cancelled_acv=float(d.loc[d.churned.eq(1),'acv'].sum())
    counts.append(dict(seed=seed,contracts=len(d),accounts=d.company_id.nunique(),cancelled_contracts=int(d.churned.sum()),accounts_with_cancellation=n_churn_accounts,cancelled_acv=cancelled_acv))
    for policy,col in policies.items():
        lead=rank_contracts(d,col).drop_duplicates('company_id').copy()
        lead['policy']=policy;lead['rank']=np.arange(1,len(lead)+1)
        lead.head(100).to_csv(OUT/f'seed_{seed}_{policy}_top100.csv',index=False)
        for cap in [25,30,50,100]:
            top=lead.head(cap);n=int(top.churned.sum());lost=float(top.loc[top.churned.eq(1),'acv'].sum())
            rows.append(dict(seed=seed,policy=policy,capacity=cap,n_accounts_reviewed=len(top),n_lead_churned=n,precision_lead_churned=n/len(top),recall_churned_accounts=n/max(n_churn_accounts,1),cancelled_lead_acv=lost,pct_cancelled_acv_of_pool=lost/max(cancelled_acv,1)))
result=pd.DataFrame(rows);result.to_csv(OUT/'all_seed_policy_results.csv',index=False)
pd.DataFrame(test_metrics).to_csv(OUT/'all_seed_test_metrics.csv',index=False)
pd.DataFrame(counts).to_csv(OUT/'all_seed_pool_counts.csv',index=False)
# Reproduce every primary review-policy point estimate from the active runner.
actual=pd.read_csv(ROOT/'part2/outputs/renewal_review_policy_comparison.csv')
check=result[result.seed.eq(137)].merge(actual,on=['policy','capacity'],suffixes=('_independent','_pipeline'))
assert len(check)==len(actual)==16
for col in ['n_lead_churned','precision_lead_churned','recall_churned_accounts','cancelled_lead_acv','pct_cancelled_acv_of_pool']:
    assert np.allclose(check[col+'_independent'],check[col+'_pipeline'],atol=.00006),col
frozen=json.loads((OUT/'frozen_source_hashes.json').read_text())
for f,h in frozen['files'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
receipt=dict(primary_model=str(artifact.relative_to(ROOT)),model_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),feature_count=len(features),seed_order=[137,991,2027],refitted_on_replication=False,primary_policy_metrics_reproduced=16,source_freeze_verified=True,claim_scope='Recovery of assumed synthetic mechanisms, not PANW validation')
(OUT/'replication_receipt.json').write_text(json.dumps(receipt,indent=2))
print(result[result.capacity.eq(30)].to_string(index=False))
print(pd.DataFrame(test_metrics).to_string(index=False))
