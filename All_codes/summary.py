from Try4 import *
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#from statsmodels.stats.outliers_influence import variance_inflation_factor

X = sm.add_constant(X_train)

# Fit the linear regression model
model = sm.OLS(y_train, X)
results = model.fit()

# Model summary
print(results.summary())


residuals = y_test - y_pred

# Plot residuals
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at 0
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

residuals = y_test - y_pred

# Q-Q plot for residuals
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=20, kde=True)  # kde=True adds a density curve
plt.axvline(x=0, color='red', linestyle='dashed', linewidth=2)  # Reference line at zero
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()