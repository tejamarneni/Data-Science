import numpy as np
from scipy import stats

np.random.seed(42)

# --- Data Generation ---
N_participants = 20
# Weights before program (avg 180 lbs)
weights_before = np.random.normal(loc=180, scale=15, size=N_participants)

# To simulate 'after', we take 'before' and subtract some amount plus random noise.
# On average, they lost about 5 lbs.
weight_change = np.random.normal(loc=-5, scale=2, size=N_participants)
weights_after = weights_before + weight_change

print(f"Mean Before: {np.mean(weights_before):.2f}")
print(f"Mean After:  {np.mean(weights_after):.2f}")
print(f"Mean Difference: {np.mean(weights_after - weights_before):.2f}")


# --- Performing the Test ---
# stats.ttest_rel(a, b) stands for 'related' samples
t_statistic, p_value = stats.ttest_rel(weights_before, weights_after)

print(f"\nResults for Paired t-test:")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4e}")

# --- Interpretation ---
alpha = 0.05
if p_value < alpha:
    print(f"Conclusion: REJECT the null hypothesis.")
    print("There is a significant difference between the before and after weights.")
else:
    print(f"Conclusion: FAIL TO REJECT the null hypothesis.")
    print("The program did not result in a statistically significant weight change.")
