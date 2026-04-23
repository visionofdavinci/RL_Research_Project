# RL Research Project

This contains the following

- Baseline:   Deep Q-Learning with epsilon-greedy strategy (done)
- Approach 3: Scrappy Adversial Training/ Exploration Diversity Curriculum Learning - pole length frequency (implemented)
- Approach 2: Scrappy Adversial Training/ Adaptive Curriculum Learning - pole length performance (implemented)
- Approach 1: Prioritized Experience Replay - sampling strategy for Replay Buffer (not yet implemented)

- Tests for each of the 4 
And: 
- Comparison of the 3 approaches

To run you need to:
- pip install torch
- pip install numpy
- pip install gymnasium
- pip install pandas
- pip install matplotlib 

Statistical tests:

1. Welch's t-test per length 
- unpaired because test_pole_length resets the env without a fixed seed, so runs for baseline and a method aren’t matched one-by-one. That makes the samples independent, not paired.

Why Welch over classic t? 
- With n ≈ 10 per group and potentially different variances across policies (and across pole lengths), the equal-variance assumption is shaky. Welch’s t-test is robust to unequal variances and unequal n, giving more reliable p-values.

2. Mann–Whitney U tests per length
- Mann–Whitney doesn’t assume normality; it tests for stochastic dominance (roughly, a shift in central tendency/median) => distribution robustness

Running both Welch and Mann–Whitney lets us see if conclusions depend on normality assumptions. Agreement between the two boosts confidence.

3. Holm–Bonferroni correction across the 30 per-length tests
- run 30 tests per method (one per pole length). Without correction, we inflate false positives

Why Holm? 
- Holm–Bonferroni controls the family-wise error rate (FWER) like Bonferroni, but it’s uniformly more powerful (fewer false negatives). It sequentially adjusts p-values, stopping when one fails

4. Overall aggregate tests (all lengths and runs flattened)
 - per-length tests ask, “At this length, is the method better?” while the overall test asks, “Across the entire distribution of lengths, is the method better on average?”

Pros: aggregation increases power (more samples) and provides a single, easy-to-interpret result
Cons: it mixes contexts (lengths), so it can hide length-specific wins/losses. 
=> That’s why we keep both per-length and overall results.

5. Effect sizes using Hedges’ g
- p-values tell you if an effect is likely non-zero; they don’t say how big it is. Effect sizes quantify practical significance.

Why Hedges’ g?
- With smallish samples, Hedges’ g applies a small-sample correction to Cohen’s d, giving a less biased estimate of standardized mean difference. This makes comparisons across lengths/policies fairer.