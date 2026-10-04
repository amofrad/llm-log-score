# Guarantee scope for the efficiency comparison

For a fixed rule, let N be the number of selection questions answered and K
the number of returned answers graded incorrect. Under independent sampling
from the same distribution as future questions, K conditional on N is
Binomial(N, r), where r is conditional error among returned answers. Define

    p(alpha) = Pr{Binomial(N, alpha) <= K}, if N > 0;
               1, otherwise.

For the null r > alpha this is a superuniform p-value, including after
averaging over N. Testing p(alpha) <= delta/M is equivalent to comparing
the one-sided Clopper-Pearson upper limit at tail delta/M with alpha.
Dependence among rules does not invalidate Bonferroni, Holm, fixed-sequence,
or the fixed graphical procedure used here.

## Holm and fixed-sequence tests

Holm compares ordered p-values p_(j) against delta/(M-j+1), stopping at the
first failure. It controls the probability of any false rejection for the
fixed alpha. In each preordered fixed-sequence chain, a false rejection
requires rejection of its first true null, whose probability is at most the
chain's budget. Union bounding over the ten chains gives delta. No monotonic
risk assumption is required; a poor ordering affects power, not validity.

## Fixed-chain fallback

Each threshold initially receives delta/M, and a rejection transfers its
available budget down the fixed high-to-low threshold chain. Failure blocks
that transfer. This is a special case of sequential graphical testing.
One can also see validity directly: before the first true null is rejected,
each true null can receive only the budgets of the preceding consecutive
false nulls and its own budget. These disjoint blocks have total budget at
most delta. A union bound over the fixed true nulls controls any false
rejection. Already failed true nulls cannot transfer their budgets. Since
every node retains at least its initial budget when tested, all Bonferroni
rejections are retained.

After any of these tests, choose the least restrictive rejected rule. Except
with probability delta it either abstains everywhere or has r <= alpha.
This assertion is for a fixed alpha and a fixed method/model/grader/grid;
it is not automatically simultaneous over alpha, models, or methods.

## Partial inclusion at tied probabilities

For a fixed (t, gamma), augment each question with an independent
V ~ Uniform(0,1). Answer if U > t or (U = t and V <= gamma), always
abstaining on empty lists. The augmented observations are independent and
identically distributed, so the same exact binomial p-value and confidence
bound apply to INTEGER realized answer/error counts. The risk averages over
fresh independent randomization on future questions.

The 408 rule family is fixed. Bonferroni bounds therefore hold simultaneously
for all rules, and the same valid bounds can be reused across every alpha.
Selection maximizes answer rate among qualifying rules because this family
is totally ordered by inclusion: lower t is less restrictive, and at equal
t, larger gamma is less restrictive. Including gamma=1 recovers every
original deterministic rule, but the larger multiple-testing adjustment can
still reduce support. Improvement is not guaranteed.

For descriptive test summaries one may also compute acceptance weights
1{U>t} + gamma*1{U=t}. Their average is the expected answer rate on the fixed
test questions. The weighted error count divided by the weighted answer
count is a randomization-averaged count ratio, not a binomial observation
and not the expectation of a finite-sample error fraction. Exact test
intervals in this experiment use realized integer counts instead.

## Interpretation

Finite test error above alpha is not itself failure of the population-risk
guarantee. Repeated overlapping partitions describe sensitivity; their
frequency of above-alpha test errors is not a valid estimate of FWER.
These methods extend the empirical comparison using established testing
arguments, not a claim of a new universal risk-control theorem.
