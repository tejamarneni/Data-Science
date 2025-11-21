import numpy as np
from scipy import stats

# Set seed for reproducibility so you get the same results as me
np.random.seed(42)

# --- Data Generation ---
# Generate sample data: 30 students with an average score around 74, std dev of 5
sample_scores = np.random.normal(loc=74, scale=5, size=30)
population_mean_to_test = 70

print(f"Sample Mean: {np.mean(sample_scores):.2f}")

# --- Performing the Test ---
# stats.ttest_1samp(a, popmean)
t_statistic, p_value = stats.ttest_1samp(sample_scores, population_mean_to_test)

print(f"\nResults for One-Sample t-test:")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4e}") # Using scientific notation for very small ps

# --- Interpretation ---
alpha = 0.05
if p_value < alpha:
    print(f"Conclusion: Since p ({p_value:.4f}) < {alpha}, we REJECT the null hypothesis.")
    print("The sample mean is statistically significantly different from the population mean of 70.")
else:
    print(f"Conclusion: Since p ({p_value:.4f}) >= {alpha}, we FAIL TO REJECT the null hypothesis.")
    print("There is no significant difference between the sample mean and population mean.")
