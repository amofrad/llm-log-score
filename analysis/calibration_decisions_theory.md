# Connecting calibration to penalty-based decisions

This note develops Routes 1 and 2 for review. It does not alter the manuscript.
The decision always concerns the original top candidate; improving its
selection policy does not improve candidate generation. The results below
specialize established calibration and cost-sensitive classification ideas.

## 1. The calibration gap has an exact decision-score interpretation

Let A_t indicate answering at threshold t, including A_t=0 for empty reports.
On eligible reports, U is the reported top probability and Z is its correctness.
For positive answer probability, write

\[
c(t)=\Pr(A_t=1),\qquad
r(t)=\mathbb E[1-Z\mid A_t=1],\qquad
e(t)=\mathbb E[1-U\mid A_t=1].
\]

Under +1 for correct, -L for incorrect, and 0 for abstention, the expected score
per question and the corresponding value implied by the reported probabilities
are

\[
V_L(t)=\mathbb E[A_t\{Z-L(1-Z)\}]
      =c(t)\{1-(1+L)r(t)\},
\]
\[
\widetilde V_L(t)=\mathbb E[A_t\{U-L(1-U)\}]
      =c(t)\{1-(1+L)e(t)\}.
\]

Therefore, exactly,

\[
\boxed{\widetilde V_L(t)-V_L(t)
       =(1+L)c(t)\{r(t)-e(t)\}.}
\]

This follows by subtraction; it requires neither calibrated probabilities nor
a normalized complete report. With no answers both scores are zero, and the
conditional error quantities need not be defined. The empirical identity holds
exactly when all terms use the same questions and candidate representation.

At t_L=L/(1+L), the dotted line 1-t_L=1/(1+L) is also the break-even conditional
error rate: V_L(t_L) is positive, zero, or negative according as r(t_L) is below,
equal to, or above that line. A positive calibration gap can coexist with
positive expected score because the original report may imply a larger
positive score. Sample versions are descriptive, with sampling uncertainty.

## 2. Population-optimal decisions based on the reported top probability

Condition on an eligible report and define g(u)=Pr(Z=1 | U=u). Let d(U) in
{0,1} indicate whether to return its top candidate. Conditional expected score
from answering is (1+L)g(U)-L. Thus an optimal decision based on U is

\[
d_L^*(U)=\mathbf 1\{g(U)\ge \tau_L\},\qquad \tau_L=\frac{L}{1+L},
\]

with either action optimal at equality. When g is nondecreasing, the answering
region is an upper interval in U. Write its boundary as

\[
t_g(L)=\inf\{u:g(u)\ge\tau_L\}.
\]

The inequality rule in g is the authoritative definition: at a discontinuous
boundary, the endpoint can require a strict rather than weak inequality in U.
For a continuous g, or the right-continuous fitted step function used below,
the weak raw-score threshold gives the stated rule. An empty acceptance region
means all-abstain. Without monotonicity, the optimal U-based region need not
be one interval; optimizing over raw thresholds is then a restricted problem.

If g_D(u)<=g_S(u) on their relevant common domain and both are nondecreasing,
their boundary thresholds satisfy t_D(L)>=t_S(L), interpreting no acceptance
as +infinity. An average calibration discrepancy does not imply this pointwise
ordering.

For an estimate h of g, let d_h=1{h(U)>=tau_L}. Conditional on eligibility, its
exact score regret is

\[
V_L(d_L^*)-V_L(d_h)
 =(1+L)\mathbb E\left[|g(U)-\tau_L|
                  \mathbf 1\{d_h(U)\ne d_L^*(U)\}\right].
\]

Proof: for each U, an incorrect action sacrifices the absolute expected score
of answering; the optimal action has its sign. In particular, if |h-g|<=epsilon
almost surely, decision disagreement is confined to |g-tau_L|<=epsilon and
regret is at most

\[
(1+L)\epsilon\Pr\{|g(U)-\tau_L|\le\epsilon\}.
\]

Overall per-question regret multiplies this eligible-report expression by the
probability of eligibility. This is a conditional estimation bound, not an
empirical estimate of regret or evidence that a uniform error bound holds.

## 3. Direct threshold optimization and increasing penalties

Let a(t)=Pr(A_t=1,Z=1) and h(t)=Pr(A_t=1,Z=0). On a fixed finite threshold
family that also includes all-abstain,

\[
t^*(L)=\min\arg\max_t\{a(t)-Lh(t)\}.
\]

This selects the smallest threshold among ties, and it is weakly increasing
in L, without assuming monotonicity of g. The same statement holds for
empirical correct and incorrect counts.

Proof: suppose L_2>L_1 but t_2=t^*(L_2)<t_1=t^*(L_1). The additional answers
admitted by t_2 have nonnegative correct and incorrect probabilities da,dh.
Optimality gives da-L_1 dh<=0 and da-L_2 dh>=0. Hence dh=0 and da=0, so t_2
also maximizes at L_1, contradicting the smallest-threshold tie convention.

For empirical selection, distinct observed training scores, zero, and
all-abstain cover all possible training answer sets. Once training data are
fixed, this family is fixed for every L. Reusing the training maximum as a
performance estimate would be optimistic; test questions are required.

## 4. Isotonic calibration and direct empirical optimization coincide

Group training observations at distinct eligible scores u_1<...<u_J. Group j
has n_j observations and k_j correct answers. Let fitted values p_j minimize

\[
\sum_{j=1}^J n_j(k_j/n_j-p_j)^2,
\qquad p_1\le\cdots\le p_J.
\]

Weighted pool-adjacent-violators (PAV) fits constant blocks with probability
equal to each block's pooled correct fraction. At every L>=0, including all
blocks with p_j>=L/(1+L) gives the same training answers as maximizing
correct count - L*incorrect count over thresholds, with ties toward more
answers. When all groups qualify, return threshold zero; if none qualify,
return all-abstain.

Proof: within any fitted block of mean p, every initial segment has correctness
fraction at least p and every terminal segment has fraction at most p. These
are the partial-sum optimality conditions for isotonic regression. A threshold
can only retain a terminal segment of a block. If p<tau_L, every nonempty
terminal segment has negative score, so omitting the block is optimal. If
p>=tau_L, adding any missing initial segment has nonnegative score, so taking
the whole block is at least as good. Thus an optimal threshold lies at a block
boundary and retains exactly the blocks at or above tau_L. Including all
zero-score blocks chooses the smallest threshold among maximizing answer sets.

This result does not assume the true correctness function is monotone. It is
an empirical equivalence; it does not imply calibration or optimality on new
questions. Extending the fit between training scores requires a convention.
Using the value at the largest observed score no greater than U, clipped at
the endpoints, makes the step-calibration rule and the chosen raw threshold
agree on new scores as well. Standard linear interpolation can give different
numerical cutoffs between training scores despite identical training decisions.

The PAV/ROC convex hull relationship is established in
[Fawcett and Niculescu-Mizil (2007)](https://doi.org/10.1007/s10994-007-5011-0).
The value here is a unified, transparent procedure for the paper's rubric,
with held-out evaluation and explicit report-processing assumptions.

## 5. Operational procedure

1. On independent labeled fitting questions, select each original top candidate
   without consulting its label, then record (U,Z).
2. Group tied scores and pool adjacent blocks until correct fractions are
   nondecreasing. Store block starts, counts, and correct fractions.
3. A user supplies L. Retain blocks whose fitted fraction is at least L/(1+L).
   The first retained block gives the reported-probability cutoff.
4. On a new eligible report, return its original top candidate if its reported
   probability meets the cutoff. Otherwise abstain. No new model call or
   answer key is needed at this decision step.

Larger penalties produce weakly higher thresholds from the same fitted map.
The method needs representative labeled fitting data, and it does not certify
a fixed future conditional error rate. Its performance should be measured in
held-out score together with answer rate and conditional error. Learning only
to abstain can improve a negative baseline score, but does not establish useful
answering performance.

A prespecified smooth sensitivity uses monotone
[beta calibration](https://proceedings.mlr.press/v54/kull17a.html). It estimates
the same population correctness map using a restricted smooth family rather
than an empirical step function; it is not generally identical to direct
empirical score maximization.
