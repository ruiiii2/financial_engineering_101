## Question 1
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# a. Import data
data = pd.read_csv('exercise3c.csv')

# b. Transfer DATE to date
data['Date'] = pd.to_datetime(data['Date'])

# Calculate simple return
AAPL_sr = np.diff(data['AAPL_Adj_Close'])/data['AAPL_Adj_Close'][:-1]  

data['AAPL_sr'] = np.append(np.nan, AAPL_sr)

# Calculate excess return
data['AAPL_rp'] = data['AAPL_sr'] - data['RF']

# c. Calculate summary statistics of these risk premia
results_sst = data.loc[:,'AAPL_sr':'AAPL_rp'].describe()
print(results_sst)                                           

# d. Fit the CAPM
data = data.drop(data.index[0])                         ## remove the first row (contain NaN)

# CAPM
capm = smf.ols(formula = 'AAPL_rp ~ Mkt_RF', data = data)
result_capm = capm.fit()
print(f'result_capm.summary():\n{result_capm.summary()}\n')           ## The estimates can be seen in the printed table

# e. Three-factor model 
ff3 = smf.ols(formula = 'AAPL_rp ~ Mkt_RF + SMB + HML', data = data)
result_ff3 = ff3.fit()
print(f'result_ff3.summary():\n{result_ff3.summary()}\n')      

# f. Five-factor model 
ff5 = smf.ols(formula = 'AAPL_rp ~ Mkt_RF + SMB + HML + RMW + CMA', data = data) 
result_ff5 = ff5.fit()
print(f'result_ff5.summary():\n{result_ff5.summary()}\n')             

# g. Six-factor model
ff6 = smf.ols(formula = 'AAPL_rp ~ Mkt_RF + SMB + HML + RMW + CMA + MOM', data = data)
result_ff6 = ff6.fit()
print(f'result_ff6.summary():\n{result_ff6.summary()}\n')

# h. Compare the alpha
capm_alpha = round(result_capm.params['Intercept'], 4)
ff3_alpha = round(result_ff3.params['Intercept'], 4)
ff5_alpha = round(result_ff5.params['Intercept'],4)
ff6_alpha = round(result_ff6.params['Intercept'], 4)

print(capm_alpha, ff3_alpha, ff5_alpha, ff6_alpha)

### ------------------------------------------------------------------------
## Question 2
import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

# Import data WAGE2
wage2 = woo.data('WAGE2')

# a. Run a regression of IQ on educ
reg_IQ = smf.ols(formula = 'IQ ~ educ', data = wage2)
results_IQ = reg_IQ.fit()
delta_educ_hat = results_IQ.params['educ']

print(round(delta_educ_hat, 4))

# b. Run a regression of log(wage) on educ
reg1 = smf.ols(formula = 'np.log(wage) ~ educ', data = wage2)
results1 = reg1.fit()
beta_educ_tilde = results1.params['educ']

print(round(beta_educ_tilde, 4))

# c. Run a regression of log(wage) on educ and IQ
reg = smf.ols(formula = 'np.log(wage) ~ educ + IQ', data = wage2)  
results = reg.fit()
beta_hat = results.params

print(beta_hat)

# d. Verify the result
beta_educ_tilde1 = beta_hat['educ']+beta_hat['IQ']*delta_educ_hat
print(round(beta_educ_tilde1, 4))

# e. Does the result apply to the intercept term? The answer is Yes!
delta_int_hat = results_IQ.params['Intercept']
print(delta_int_hat)

beta_int_tilde = results1.params['Intercept']
print(beta_int_tilde)

beta_int_tilde1 = results.params['Intercept'] + results.params['IQ']*delta_int_hat
print(round(beta_int_tilde1, 4))

### ----------------------------------------------------------
## Question 3
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf

# Redo the OLS estimation
wage1 = woo.data('wage1')

reg = smf.ols(formula='np.log(wage) ~ educ + exper + tenure', data = wage1)
results = reg.fit()
beta_educ_hat = results.params['educ']
beta_educ_hat = round(beta_educ_hat, 4)

print(f'beta_educ_hat: \n{beta_educ_hat}\n')

# The first step: regress educ on exper and tenure with the OLS 
# and save the residuals
reg_educ = smf.ols(formula='educ ~ exper + tenure', data = wage1)
results_educ = reg_educ.fit()
wage1['resid_educ'] = results_educ.resid

# The second step: regress log(wage) on resid_educ with the OLS 
# and obtain the estimated slop parameter
reg1 = smf.ols(formula='np.log(wage) ~ resid_educ', data = wage1)
results1 = reg1.fit()
beta_educ_hat1 = results1.params['resid_educ']
beta_educ_hat1 = round(beta_educ_hat1, 4)

print(f'beta_educ_hat1: \n{beta_educ_hat1}')          

# Is this result applied to the estimated intercept?
# The first step: regress a vector of ones on educ, exper and tenure with the OLS 
# and save the residuals
# note that NO INTERCEPT HERE!
wage1['ones'] = 1                       # add a vector of ones into the data set
reg_int = smf.ols(formula='ones ~ educ + exper + tenure -1', data = wage1)              
results_int = reg_int.fit()                 
wage1['resid_int'] = results_int.resid

# The second step: regress y on resid_educ with the OLS 
# and obtain the estimated slop parameter
reg_int1 = smf.ols(formula='np.log(wage) ~ resid_int', data = wage1)
results_int1 = reg_int1.fit()
beta_int_hat1 = results_int1.params['resid_int']
beta_int_hat1 = round(beta_int_hat1, 4)

# The OLS intercept from original regression
beta_int_hat = results.params['Intercept']
beta_int_hat = round(beta_int_hat, 4)

print(f'beta_int_hat: \n{beta_int_hat}\n')
print(f'beta_int_hat1: \n{beta_int_hat1}')