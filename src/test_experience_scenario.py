"""Mechanism invariants for the disclosed synthetic demonstration scenario."""
import sys
from pathlib import Path
import json
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).parent))
from generate_data import (_compute_renewal_probability, _forward_customer_state,
                           _reference_experience_intercept, _experience_interaction_score, rng)
CFG = yaml.safe_load((Path(__file__).parents[1] / 'config/simulation.yaml').read_text())


def state(d=.05, v=.05, service=.05, commercial=.05):
    return dict(deployment_gap=d, value_decline=v, service_problem=service,
                commercial_pressure=commercial, owner_gap_days=0, onboarding_delay_days=0)


def test_joint_failure_adds_more_than_independent_penalties():
    p = CFG['experience_scenario']
    score = lambda s: float(_experience_interaction_score(s, p))
    joint = score(state(.65, .50)) - score(state())
    separate = score(state(.65, .05)) + score(state(.05, .50)) - 2*score(state())
    assert joint > separate + 3


def test_probability_is_stochastic_and_segment_anchored():
    for seg, base in [('smb', .08), ('midmarket', .05), ('enterprise', .025), ('large_enterprise', .01)]:
        healthy = _compute_renewal_probability(state(), seg, CFG, base)
        struggling = _compute_renewal_probability(state(.75, .65, .3), seg, CFG, base)
        assert 0 < healthy < struggling < 1
    assert _compute_renewal_probability(state(.6,.5), 'smb', CFG, .08) > _compute_renewal_probability(state(.6,.5), 'enterprise', CFG, .025)


def test_reference_annual_rates_on_independent_latent_draws():
    rg = rng(54733)
    names = sorted(CFG['deployment_archetypes'])
    for seg, rate in [('smb',.08), ('midmarket',.05), ('enterprise',.025), ('large_enterprise',.01)]:
        rows = [_forward_customer_state(seg, str(a), 365, rg) for a in rg.choice(names, 12000, p=[CFG['deployment_archetypes'][a] for a in names])]
        x = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        for k, sd in [('deployment_gap',.08), ('value_decline',.08), ('service_problem',.04), ('commercial_pressure',.08)]:
            x[k] = np.clip(x[k]+rg.normal(0,sd,len(rows)),0,1)
        intercept = _reference_experience_intercept(seg, rate, json.dumps(CFG['experience_scenario'],sort_keys=True), json.dumps(CFG['deployment_archetypes'],sort_keys=True))
        actual = np.mean(1/(1+np.exp(-(intercept+_experience_interaction_score(x,CFG['experience_scenario'])))))
        assert abs(actual-rate)<.012, (seg,rate,actual)


def test_hidden_states_are_not_model_inputs():
    sys.path.insert(0,str(Path(__file__).parents[1]/'part2/src'))
    from _features import FULL_CONTEXT_NUM, FULL_CONTEXT_CAT
    assert not set(FULL_CONTEXT_NUM + FULL_CONTEXT_CAT).intersection({'deployment_gap','value_decline','service_problem','commercial_pressure','computed_cancel_prob','archetype'})


def test_bootstrap_optimization_preserves_exact_rows_and_multiplicity():
    import pandas as pd
    from pandas.testing import assert_frame_equal
    sys.path.insert(0,str(Path(__file__).parents[1]/'part2/src'))
    from _review_bootstrap import resample_parent_rows
    df = pd.DataFrame({'purchasing_account':['a','a','b','c','d'], 'parent':['p','p','p','q','r'], 'score':[.2,.9,.5,.6,.3]}, index=[2,4,7,10,19])
    ix = {k:g.index.tolist() for k,g in df.groupby('parent',sort=False)}
    rg = np.random.default_rng(42)
    for _ in range(50):
        draws = rg.choice(['p','q','r'],size=5,replace=True)
        pieces=[];counter={}
        for parent in draws:
            occ=counter.get(parent,0);counter[parent]=occ+1
            rows=df.loc[ix[parent]].copy()
            rows['_boot_acct']=rows.purchasing_account.astype(str)+f'__{occ}'
            pieces.append(rows)
        expected=pd.concat(pieces,ignore_index=True)
        assert_frame_equal(resample_parent_rows(df,draws,ix),expected)


def test_review_score_ties_are_input_order_invariant():
    import pandas as pd
    sys.path.insert(0,str(Path(__file__).parents[1]/'part2/src'))
    from _review_bootstrap import rank_contracts
    x=pd.DataFrame({'contract_id':['c3','c1','c2'], 'score':[.5,.5,.5], 'churned':[1,0,1]})
    assert rank_contracts(x,'score').contract_id.tolist()==['c1','c2','c3']
    assert rank_contracts(x.iloc[::-1],'score').contract_id.tolist()==['c1','c2','c3']


def test_current_review_uses_identical_pools_for_all_policies():
    import pandas as pd
    root=Path(__file__).parents[1]
    p=pd.read_csv(root/'part2/outputs/renewal_review_policy_comparison.csv')
    assert set(p.policy)=={'health_rules','commercial_value','model_score','model_x_acv'}
    assert set(p.capacity)=={25,30,50,100}
    assert len(p)==16
    assert p.n_contracts_churned_in_pool.nunique()==1
    assert p.n_accounts_with_any_churn.nunique()==1
    assert p.total_cancelled_acv_observed.nunique()==1


def test_current_review_capacity_counts_distinct_accounts():
    import pandas as pd
    root=Path(__file__).parents[1]
    q=pd.read_csv(root/'part2/outputs/renewal_review_accounts.csv')
    for _,g in q.groupby('policy'):
        assert g.purchasing_account.is_unique
        assert len(g[g.policy_rank.le(30)])==30


def test_current_review_is_not_automatically_outreach_cleared():
    import pandas as pd
    root=Path(__file__).parents[1]
    q=pd.read_csv(root/'part2/outputs/renewal_review_contracts.csv')
    assert q.outreach_clearance_status.str.contains('not_contact_cleared').all()
    for k in ['gate_prior_outreach_history','gate_measurement_completeness','gate_active_escalation','gate_urgent_renewal_flag']:
        assert q[k].eq('pending_manual_check').all()
