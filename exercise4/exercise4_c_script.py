## Question 1
## A simulation for examing confidence intervals
import statsmodels.formula.api as smf
import scipy.stats as stats
import pandas as pd
import numpy as np

## a. 
## Set sample size and number of iterations:
n = 100                       ## sample size
r = 50                      ## number of iteration  

## Set true parameters
beta0 = 0.4
beta1 = -1.46
beta2 = 2.5

df = n - 2 - 1
cv_005 = stats.t.ppf(1 - 0.05, df = df)                ## Critical value for 95% CI
cv_001 = stats.t.ppf(1 - 0.01, df = df)            ## Critical value for 99% CI

## Create empty vectors to store the simulation results
beta0_result_005 = np.empty(r)                 
beta1_result_005 = np.empty(r)
beta2_result_005 = np.empty(r)

beta0_result_001 = np.empty(r)
beta1_result_001 = np.empty(r)
beta2_result_001 = np.empty(r)

## Set random seed
np.random.seed(1234567)

## Generate samples of x1 and x2
x1 = stats.uniform.rvs(size = n)
x2 = stats.norm.rvs(loc = 0.78, size = n)

for i in range(r):
    
    u = np.random.normal(size = n)               ## Generate u
    y = beta0 + beta1 * x1 + beta2 * x2 + u    ## Generate y
    datax = pd.DataFrame({'y':y, 'x1':x1, 'x2':x2}) 
    
    reg = smf.ols(formula = 'y ~ x1 + x2', data = datax)
    results = reg.fit()
    beta_hat = results.params
    bse = results.bse
    
    ## Whether the true parameters lie in the 95% CI's?
    beta0_result_005[i] = (beta0 >= (beta_hat['Intercept'] - cv_005*bse['Intercept'])) and (beta0 <= (beta_hat['Intercept'] + cv_005*bse['Intercept']))
    beta1_result_005[i] = (beta1 >= (beta_hat['x1'] - cv_005*bse['x1'])) and (beta1 <= (beta_hat['x1'] + cv_005*bse['x1']))
    beta2_result_005[i] = (beta2 >= (beta_hat['x2'] - cv_005*bse['x2'])) and (beta2 <= (beta_hat['x2'] + cv_005*bse['x2']))
    
    ## Whether the true parameters lie in the 99% CI's?
    beta0_result_001[i] = (beta0 >= (beta_hat['Intercept'] - cv_001*bse['Intercept'])) and (beta0 <= (beta_hat['Intercept'] + cv_001*bse['Intercept']))
    beta1_result_001[i] = (beta1 >= (beta_hat['x1'] - cv_001*bse['x1'])) and (beta1 <= (beta_hat['x1'] + cv_001*bse['x1'])) 
    beta2_result_001[i] = (beta2 >= (beta_hat['x2'] - cv_001*bse['x2'])) and (beta2 <= (beta_hat['x2'] + cv_001*bse['x2']))


## Print out the results
## Alpha = 0.05
print(np.mean(beta0_result_005), np.mean(beta1_result_005), np.mean(beta2_result_005))
## Alpha = 0.01
print(np.mean(beta0_result_001), np.mean(beta1_result_001), np.mean(beta2_result_001))

## b. Repeat the simulations with r = 200
## Set sample size and number of iterations:
n = 100                       ## sample size
r = 200                      ## number of iteration  

## Set true parameters
beta0 = 0.4
beta1 = -1.46
beta2 = 2.5

df = n - 2 - 1
cv_005 = stats.t.ppf(1 - 0.05, df = df)                ## Critical value for 95% CI
cv_001 = stats.t.ppf(1 - 0.01, df = df)            ## Critical value for 99% CI

## Create empty vectors to store the simulation results
beta0_result_005 = np.empty(r)                 
beta1_result_005 = np.empty(r)
beta2_result_005 = np.empty(r)

beta0_result_001 = np.empty(r)
beta1_result_001 = np.empty(r)
beta2_result_001 = np.empty(r)

## Set random seed
np.random.seed(1234567)

## Generate samples of x1 and x2
x1 = stats.uniform.rvs(size = n)
x2 = stats.norm.rvs(loc = 0.78, size = n)

for i in range(r):
    
    u = np.random.normal(size = n)               ## Generate u
    y = beta0 + beta1 * x1 + beta2 * x2 + u    ## Generate y
    datax = pd.DataFrame({'y':y, 'x1':x1, 'x2':x2}) 
    
    reg = smf.ols(formula = 'y ~ x1 + x2', data = datax)
    results = reg.fit()
    beta_hat = results.params
    bse = results.bse
    
    ## Whether the true parameters lie in the 95% CI's?
    beta0_result_005[i] = (beta0 >= (beta_hat['Intercept'] - cv_005*bse['Intercept'])) and (beta0 <= (beta_hat['Intercept'] + cv_005*bse['Intercept']))
    beta1_result_005[i] = (beta1 >= (beta_hat['x1'] - cv_005*bse['x1'])) and (beta1 <= (beta_hat['x1'] + cv_005*bse['x1']))
    beta2_result_005[i] = (beta2 >= (beta_hat['x2'] - cv_005*bse['x2'])) and (beta2 <= (beta_hat['x2'] + cv_005*bse['x2']))
    
    ## Whether the true parameters lie in the 99% CI's?
    beta0_result_001[i] = (beta0 >= (beta_hat['Intercept'] - cv_001*bse['Intercept'])) and (beta0 <= (beta_hat['Intercept'] + cv_001*bse['Intercept']))
    beta1_result_001[i] = (beta1 >= (beta_hat['x1'] - cv_001*bse['x1'])) and (beta1 <= (beta_hat['x1'] + cv_001*bse['x1'])) 
    beta2_result_001[i] = (beta2 >= (beta_hat['x2'] - cv_001*bse['x2'])) and (beta2 <= (beta_hat['x2'] + cv_001*bse['x2']))


## Print out the results
## Alpha = 0.05
print(np.mean(beta0_result_005), np.mean(beta1_result_005), np.mean(beta2_result_005))
## Alpha = 0.01
print(np.mean(beta0_result_001), np.mean(beta1_result_001), np.mean(beta2_result_001))

## c. Repeat the simulations with r = 1000
## Set sample size and number of iterations:
n = 100                       ## sample size
r = 1000                      ## number of iteration  

## Set true parameters
beta0 = 0.4
beta1 = -1.46
beta2 = 2.5

df = n - 2 - 1
cv_005 = stats.t.ppf(1 - 0.05, df = df)                ## Critical value for 95% CI
cv_001 = stats.t.ppf(1 - 0.01, df = df)            ## Critical value for 99% CI

## Create empty vectors to store the simulation results
beta0_result_005 = np.empty(r)                 
beta1_result_005 = np.empty(r)
beta2_result_005 = np.empty(r)

beta0_result_001 = np.empty(r)
beta1_result_001 = np.empty(r)
beta2_result_001 = np.empty(r)

## Set random seed
np.random.seed(1234567)

## Generate samples of x1 and x2
x1 = stats.uniform.rvs(size = n)
x2 = stats.norm.rvs(loc = 0.78, size = n)

for i in range(r):
    
    u = np.random.normal(size = n)               ## Generate u
    y = beta0 + beta1 * x1 + beta2 * x2 + u    ## Generate y
    datax = pd.DataFrame({'y':y, 'x1':x1, 'x2':x2}) 
    
    reg = smf.ols(formula = 'y ~ x1 + x2', data = datax)
    results = reg.fit()
    beta_hat = results.params
    bse = results.bse
    
    ## Whether the true parameters lie in the 95% CI's?
    beta0_result_005[i] = (beta0 >= (beta_hat['Intercept'] - cv_005*bse['Intercept'])) and (beta0 <= (beta_hat['Intercept'] + cv_005*bse['Intercept']))
    beta1_result_005[i] = (beta1 >= (beta_hat['x1'] - cv_005*bse['x1'])) and (beta1 <= (beta_hat['x1'] + cv_005*bse['x1']))
    beta2_result_005[i] = (beta2 >= (beta_hat['x2'] - cv_005*bse['x2'])) and (beta2 <= (beta_hat['x2'] + cv_005*bse['x2']))
    
    ## Whether the true parameters lie in the 99% CI's?
    beta0_result_001[i] = (beta0 >= (beta_hat['Intercept'] - cv_001*bse['Intercept'])) and (beta0 <= (beta_hat['Intercept'] + cv_001*bse['Intercept']))
    beta1_result_001[i] = (beta1 >= (beta_hat['x1'] - cv_001*bse['x1'])) and (beta1 <= (beta_hat['x1'] + cv_001*bse['x1'])) 
    beta2_result_001[i] = (beta2 >= (beta_hat['x2'] - cv_001*bse['x2'])) and (beta2 <= (beta_hat['x2'] + cv_001*bse['x2']))


## Print out the results
## Alpha = 0.05
print(np.mean(beta0_result_005), np.mean(beta1_result_005), np.mean(beta2_result_005))
## Alpha = 0.01
print(np.mean(beta0_result_001), np.mean(beta1_result_001), np.mean(beta2_result_001))

### ------------------------------------------------------------
## Question 2
import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats
import numpy as np

## Import data
vote1 = woo.data('vote1')

## a. Descriptive statistics
vote1[['voteA','lexpendA','lexpendB','prtystrA']].describe()

## b. What is the interpretation of beta1?
## Answer: 
    
## c. In terms of the parameters, state the null hypothesis that a 1% increase in A's expenditure is 
## offset by a 1% increase in B's expenditure
## Answer: 
    
## d. Fit the regression 
reg = smf.ols(formula = 'voteA ~ lexpendA + lexpendB + prtystrA', data = vote1)
results = reg.fit()

print(f'results.summary():\n{results.summary()}\n')

## Answer: 
    
## Test H0: beta1 + beta 2 = 0 in b.
## Use method f_test
hypotheses = ['lexpendA + lexpendB = 0']
ftest = results.f_test(hypotheses)
fstat = ftest.fvalue                  ## Extract the F statistic
fpval = ftest.pvalue                      ## Extract the p-value

print(f'fstat: {round(fstat, 3)}\n')
print(f'fpval: {round(fpval, 3)}\n')
## Answer: 

## e. Estimate a modified model:    
vote1['lB_lA'] = vote1['lexpendB'] - vote1['lexpendA']
reg_mod = smf.ols(formula = 'voteA ~ lB_lA + prtystrA', data = vote1)
results_mod = reg_mod.fit()
print(f'results_mod.summary():\n{results_mod.summary()}\n')

## From the result, p-value of the t statistic (coefficient of lexpendA) is also about 0.32 and we cannnot reject the null
## at 5% significant level. We do not have enough evidence to say that there is no offset effect.

## Verify whether t^2 = F
tvalues = results_mod.tvalues
tvalues_sq = tvalues['lB_lA']**2
print(round(tvalues_sq,4), round(fstat,4))

## Answer: 

## f. Test H0: beta1 + beta2 = 1.5
## Use the method f_test
hypotheses = ['lexpendA + lexpendB = 1.5']
ftest = results.f_test(hypotheses)
fstat = ftest.fvalue
fpval = ftest.pvalue

print(f'fstat: {round(fstat, 3)}\n')
print(f'fpval: {round(fpval, 3)}\n')

## Estimate a modified model
vote1['lA_lB'] = vote1['lexpendA'] - vote1['lexpendB']
reg_mod = smf.ols(formula = 'voteA ~ lA_lB + prtystrA', data = vote1)
results_mod = reg_mod.fit()
print(f'results_mod.summary():\n{results_mod.summary()}\n')

## Manually calculate t statistic
n = vote1.shape[0]
k = 3
df = n - k - 1
b_lexpendA = results_mod.params['lA_lB']
se_lexpendA = results_mod.bse['lexpendA']
tstat = round((b_lexpendA - 1.5) / se_lexpendA, 3)
tpval = 2*stats.t.cdf(-abs(tstat), df = df)

print(f'tstat: {round(tstat, 3)}\n')
print(f'tpval: {round(tpval, 3)}\n')                  ## should be the same value as fpval

## Use method t_test
hypothesis = 'lA_lB = 1.5'
ttest = results_mod.t_test(hypothesis)
tstat = ttest.statistic[0][0] 
tpval = ttest.pvalue

print(f'tstat: {round(tstat, 3)}\n')
print(f'tpval: {np.around(tpval, 3)}\n')                  ## should be the same value as fpval

## Answer:

## -----------------------------------------------------------------------
## Question 3
import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

## Import data
hprice1 = smf.data('hprice1')

## a. Descriptive statistics
hprice1[['price','sqrft','bdrms']].describe()

## b. Fit the regression
reg = smf.ols(formula = 'np.log(price) ~ sqrft + bdrms', data = hprice1)
results = reg.fit()

## Estimate theta1: 150 sqrft bed room added
theta1 = 150*results.params['sqrft'] + results.params['bdrms']
theta1 = round(theta1, 4)
print(f'Percentage change:\n{theta1*100}%\n')

## c. 
## Calculate standard error of theta1 and construct CI's
hprice1['new_var'] = hprice1['sqrft'] - 150*hprice1['bdrms']
reg_mod = smf.ols(formula = 'np.log(price) ~ new_var + bdrms', data = hprice1)
results_mod = reg_mod.fit()

## Now the slope parameters of bdrms is theta1
bse = results_mod.bse['bdrms']
n = hprice1.shape[0]
k = 2
df = n - k - 1
cv_005 = stats.t.ppf(1 - 0.025, df = df)          ## Critical value for 95% CI
cv_001 = stats.t.ppf(1 - 0.005, df = df)          ## Critical value for 99% CI 

ci_005_up = theta1 + cv_005*bse
ci_005_low = theta1 - cv_005*bse
ci_005 = [round(ci_005_low, 3), round(ci_005_up, 3)]
print(f'ci_005:\n{ci_005}\n')                     ## 95% CI for theta1

ci_001_up = theta1 + cv_001*bse  
ci_001_low = theta1 - cv_001*bse
ci_001 = [round(ci_001_low, 3), round(ci_001_up, 3)]
print(f'ci_001:\n{ci_001}\n')                     ## 99% CI for theta1

## Answer: 