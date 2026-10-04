# An error-tolerance-to-threshold rule

For eligible reports, U is the original top-candidate probability and Z is
its graded correctness. Empty reports never answer. Define

\[
R(t)=\Pr(Z=0\mid U\ge t),\qquad C(t)=\Pr(U\ge t),
\]

where R(t) is defined when C(t)>0. The goal is to answer as many questions
as possible subject to a supported upper error tolerance alpha.

Let T contain M finite thresholds fixed without the certification observations.
It may be constructed from an independent development set. At each t, count
N_t certification answers and K_t errors. Define

\[
B_\delta(t)=
\begin{cases}
1,&N_t=0\text{ or }K_t=N_t,\\
\operatorname{Beta}^{-1}(1-\delta/M;K_t+1,N_t-K_t),&\text{otherwise}.
\end{cases}
\]

The rule is

\[
\widehat t(\alpha)=\min\{t\in T:N_t>0,\ B_\delta(t)\le\alpha\}.
\]

If this set is empty, return "no supported threshold" and abstain everywhere.
Do not assign zero conditional error to that outcome.

## Guarantee

Assume independent certification questions from the future question
distribution. Conditional on any independent development data, T is fixed.
For each t with positive answer probability, conditional on N_t the number of
errors is Binomial(N_t,R(t)). The one-sided Clopper-Pearson bound fails with
probability at most delta/M. A union bound over the M thresholds therefore gives

\[
\Pr\{R(t)\le B_\delta(t)\text{ for every }t\in T
                 \text{ with }C(t)>0\}\ge1-\delta.
\]

On this event, every threshold returned by the lookup satisfies
R(\widehat t(\alpha))<=alpha, simultaneously for all alpha in (0,1).
Thus alpha can be chosen after inspecting this same valid bound table without
an additional correction over alpha. The event is conditional on development
data and hence also holds unconditionally. Thresholds with population answer
probability zero cannot have positive certification counts except on a
probability-zero event and are not selected.

Because threshold answer sets are nested, the smallest qualifying threshold
maximizes coverage within the supported candidate family. This does not
establish the population-optimal threshold among all possible cutoffs. No
monotonicity of the true conditional risk R(t) or correctness function is
assumed. As alpha increases, the qualifying set expands, so the selected
threshold weakly decreases, treating no supported threshold as +infinity.

The 95% statement concerns the chance, over certification samples, of issuing
an incorrect population-risk certificate. It does not say every finite set of
future answers has at most alpha errors, or that each individual answer has
error probability at most alpha. It also does not cover post-hoc selection
among models or methods unless that selection is included in the error budget.

## Connection to fitted calibration

A development-only isotonic curve supplies a smaller candidate family: zero
and the raw U values where the fitted correctness curve increases. These are
the finite cutoffs of the earlier penalty-based isotonic rule. Its fitted
probabilities need not be correct for the certification theorem to hold;
they affect which cutoffs are considered, not validity of the independent
binomial bounds. A smaller M reduces the search adjustment, at the cost of
using data for development and potentially omitting useful cutoffs.

The certified condition bounds average error across all answers at t.
It does not require the fitted correctness at every included probability
level to exceed 1-alpha. Consequently, optimizing expected penalty score and
maximizing coverage subject to this risk condition remain different objectives.

This is an application of established exact confidence bounds and simultaneous
risk control, not a new universal algorithm. See
[Angelopoulos et al., Learn then Test](https://arxiv.org/html/2110.01052v4).
