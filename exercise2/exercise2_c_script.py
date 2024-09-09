# Question 1
import wooldridge as woo
import statsmodels.formula.api as smf
import numpy as np

ceosal2 = woo.dataWoo('ceosal2') 
y = ceosal2['salary']
x = ceosal2['ceoten']

# a. Average salary and tenure
avg_sal = np.mean(y)
avg_ten = np.mean(x)

print(round(avg_sal, 2))
print(round(avg_ten, 2))

# b. How many CEO's are in their first year?
ten_year1 = np.sum(x==0)

print(ten_year1) 

# Longest tenure 
ten_max = np.max(x)

print(ten_max)

# c. Estimate the regression
reg = smf.ols('np.log(salary) ~ ceoten', data = ceosal2)
results = reg.fit()
b = results.params

# Print parameter estimates
print(f'b: \n{b}\n')

# Print results using summary:
print(f'results.summary(): \n{results.summary()}\n')

# Print regression table
import pandas as pd

table = pd.DataFrame({'b': round(b, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')

# ------------------------------------------------------------------------
# Question 2
import scipy.stats as stats
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

# a. Generate 500 uniform random variable x
# Lower = loc, Upper = loc + scale 
x = stats.uniform.rvs(loc = 0, scale = 10, size = 500)

# Sample mean and standard deviation of x
mean_x = x.mean()
std_x = x.std()
print(mean_x)
print(std_x)

# b. Generate 500 N(0, 36) variable u
u = stats.norm.rvs(loc = 0, scale = np.sqrt(36), size = 500)
mean_u = u.mean()
std_u = u.std()
print(mean_u)
print(std_u)

# c. Generate y = 1 + 2x +u
y = 1 + 2*x + u

data1 = pd.DataFrame({'y':y, 'x': x})

# Run the regression
reg = smf.ols(formula = 'y~x', data = data1)
results = reg.fit()

# Parameter estimates
b = results.params
print(f'b:\n{b}\n')

# d. OLS residuals
# Two ways to get the residuals 
u_hat1 = results.resid

# Verify properties of residuals (use u_hat1)
sum_u_hat1 = u_hat1.sum()
print(sum_u_hat1)

x_u_hat1 = (x * u_hat1).sum() 
print(x_u_hat1)

# e. Use u 
sum_u = u.sum()
print(sum_u)

x_u = (x*u).sum()
print(x_u)

# f. Repeat the above procedures again


# -------------------------------------------------
# Question 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# a. Import data
data = pd.read_csv('data_exercise_2c.csv')

# b. Transfer DATE to date
data['DATE'] = pd.to_datetime(data['DATE'])

# c. Calculate simple return
SP500_sr = np.diff(data['SP500'])/data['SP500'][:-1]
AAPL_sr = np.diff(data['AAPL_Close'])/data['AAPL_Close'][:-1]
AAPL_Adj_sr = np.diff(data['AAPL_Adj_Close'])/data['AAPL_Adj_Close'][:-1]

data['SP500_sr'] = np.append(np.nan, SP500_sr)
data['AAPL_sr'] = np.append(np.nan, AAPL_sr)
data['AAPL_Adj_sr'] = np.append(np.nan, AAPL_Adj_sr)

# d. Calculate risk premia
data['MKT_rp'] = data['SP500_sr'] - data['RF_%']/100
data['AAPL_rp'] = data['AAPL_sr'] - data['RF_%']/100
data['AAPL_Adj_rp'] = data['AAPL_Adj_sr'] - data['RF_%']/100

# Calculate summary statistics of these risk premia
results_sst = data[['MKT_rp', 'AAPL_rp', 'AAPL_Adj_rp']].describe()
print(results_sst)

# e. Fit CAPM 
data = data.drop(data.index[0])      # remove the row containing NaN

# AAPL_rf
capm = smf.ols('AAPL_rp ~ MKT_rp', data=data)
result_capm = capm.fit()
print(f'result_capm.summary():\n{result_capm.summary()}\n')

# f. AAPL_adj_rf
capm_adj = smf.ols('AAPL_Adj_rp ~ MKT_rp', data=data)
result_capm_adj = capm_adj.fit()
print(f'result_capm_adj.summary():\n{result_capm_adj.summary()}\n')

# g. Plot fitted results
fig = plt.figure()       # initialize figure window
fig.subplots_adjust(hspace=.5, wspace=0.4) # Use this to do some adjustments

# AAPL  
x = data['MKT_rp']       # Market risk premium
y = data['AAPL_rp']      # Risk premium of individual stock 
b = result_capm.params   # Estimated parameters

x_range = np.linspace(data['MKT_rp'].min(), data['MKT_rp'].max(), num = 200)
ax = fig.add_subplot(1, 2, 1)

plt.plot(x, y, color = 'blue', marker = 'o', linestyle = '')
plt.plot(x_range, b[0] + b[1]*x_range, color = 'red',
         linestyle = '--', linewidth = 2, label = 'Est. CAPM')
plt.xlabel('Mkt - Rf')
plt.ylabel('AAPL - Rf')
plt.legend()
ax.set_title('AAPL')        ## Use this to make a title for the plot

# AAPL_adj
x = data['MKT_rp']       # Market risk premium     
y = data['AAPL_Adj_rp']   # Risk premium of individual stock
b = result_capm_adj.params # Estimated parameters

x_range = np.linspace(data['MKT_rp'].min(), data['MKT_rp'].max(), num = 200)
ax = fig.add_subplot(1, 2, 2)

plt.plot(x, y, color = 'blue', marker = 'o', linestyle = '')
plt.plot(x_range, b[0] + b[1]*x_range, color = 'red', linestyle = '--', linewidth = 2, label = 'Est. CAPM')
plt.xlabel('Mkt - Rf')
plt.ylabel('Risk Premium')
ax.set_title('AAPL Adjusted')
plt.legend()

# Plot and save the plot
plt.savefig('capm_plots.png')