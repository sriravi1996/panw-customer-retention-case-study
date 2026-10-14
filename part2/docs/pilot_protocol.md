# Renewal-review feasibility pilot

Status: proposal, not an executed intervention or demonstrated retention effect.
The active business output is the next-90-days review from `build_review_cohort.py`.

## Population and timing

The retrospective demonstration uses review date 2026-04-01 and expiries from
2026-04-02 through 2026-06-30. Under the directly managed case-study assumption,
the common pool contains 1,505 contracts across
1,185 accounts. Original prediction
scores can be 0 to 89 days old at review. Four manual contact checks remain.
Historical outcomes are already known, so a prospective pilot must enroll a
fresh cohort under a frozen model and policy.

Check prior outreach, measurement quality, active escalation and urgent renewal
handling equally before assignment in either arm. All accounts are assumed
managed by the customer team.

## Evaluation unit and allocation

Freeze one lead contract per enrolled purchasing account before randomization;
use that contract for the primary outcome even if other products later change.
Choose the ranking objective in advance: cancellation identification or financial
exposure. Model score and commercial-value ranking answer different objectives.
Thirty reviews per cycle is a capacity assumption, not a proven staffing limit.

Randomize by parent organization where shared customer-success work could affect
related accounts; stratify using available size, expiry and risk information.
Keep related accounts in the same arm. Record both parent and account counts.
Analyze assignment as allocated (intention to treat). Estimate uncertainty by
resampling whole parents separately within each arm, retaining all member
accounts and repeated parent draws. Pre-specify account weighting and the
estimand; clustering and imbalance affect precision. A before/after comparison
alone cannot establish treatment effect.

## Treatment and usual care

Assign a named customer-success manager (CSM). Review adoption, productive use,
open service problems and documented commercial concerns. Confirm that historic
signals still apply at contact. Agree a product-specific recovery milestone,
owner and deadline; record contacts, action delivery, staff time and exceptions.
Usual care continues existing outreach without the experimental ranking and
structured action package. Log usual-care contacts and spillover as well.

For leads expiring within 30 days, use an accelerated track: first contact by
pilot day 3, a day-7 check-in, and the agreed action before expiry minus 7 days.
For other leads, first contact by day 14 and agreed action by day 21–28, with
follow-up at days 30 and 60. Urgent cases without enough runway need manual
escalation before enrollment. Do not delay usual support to protect a trial.

## Outcomes and success criteria

Primary: qualifying recorded cancellation of the frozen lead contract through
expiry plus 90 days, using the same event definition in both arms. No recorded
cancellation is not proof of renewal, intention to renew or revenue saved.

Secondary: documented product recovery, verified successor-contract annualized
value (including contraction), and opening ACV retained. Report cancellations,
verified renewals and unknown outcomes separately. ACV means annual contract
value; do not confuse it with recognized revenue. A score multiplied by ACV is a
priority index, not expected financial loss, because scores are uncalibrated.

Pre-register operational targets at launch: timely contact, milestone delivery,
record completeness, staff hours per case, and absence of harmful escalation or
spillover. A small feasibility pilot should be judged on delivery and learning;
a non-significant cancellation-rate difference is not by itself failure.

## Power and feasibility

Illustrative only: a 15% baseline cancellation rate reduced to 9% (6 percentage
points; 40% relative reduction), two-sided alpha 0.05 and 80% power, requires
approximately 460 independent accounts per arm using a normal approximation.
Re-estimate using the actual enrolled baseline and plausible intervention effect;
allow for parent clustering, attrition and unequal cluster sizes. The historical pool contains 1,185 accounts before
manual checks and parent clustering. That count alone does not guarantee enough
effective prospective enrollment. Enrollment of 100 to 150 per arm could test
feasibility but is not powered for this effect under these assumptions.


Model validation is separate: freeze assumptions, features and model before a
fresh untreated evaluation. The repeatedly inspected final_test results and
April retrospective ranking do not constitute an untouched prospective test.
No treatment effect is simulated in this case study, and real-data performance
is unknown.
