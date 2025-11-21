import numpy as np
from scipy import stats

np.random.seed(42)

# --- Data Generation ---
# Group A: Mean recovery 10 days, std dev 2, N=40
group_a_recovery = np.random.normal(loc=10, scale=2, size=40)
# Group B: Mean recovery 12 days, std dev 3, N=35 (different variance and N)
group_b_recovery = np.random.normal(loc=12, scale=3, size=35)

print(f"Group A Mean: {np.mean(group_a_recovery):.2f} (Variance: {np.var(group_a_recovery):.2f})")
print(f"Group B Mean: {np.mean(group_b_recovery):.2f} (Variance: {np.var(group_b_recovery):.2f})")

# --- Performing the Test ---
# Use equal_var=False for Welch's t-test (recommended if unsure about variances)
t_statistic, p_value = stats.ttest_ind(group_a_recovery, group_b_recovery, equal_var=False)
# Note: If you are CERTAIN variances are equal, use equal_var=True for a standard t-test.

print(f"\nResults for Independent (Welch's) t-test:")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4e}")

# --- Interpretation ---
alpha = 0.05
if p_value < alpha:
    print(f"Conclusion: REJECT the null hypothesis.")
    print("There is a significant difference in recovery times between Drug A and Drug B.")
else:
    print(f"Conclusion: FAIL TO REJECT the null hypothesis.")
    print("No significant difference found between the two drugs.")
