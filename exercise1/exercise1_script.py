#李叡_貿四乙_109301060 
# 1. Generate a sequence 0,1,2,3,...,30 with range and assign it to a list
x = range(0,31)
x = list(x)
# a. Extract 0,1,2,..,10
x[0:11]

# b. Extract 21,22,...,30
x[-10:]

# c. Extract 10,12,14,...,24
x[10:25:2]

# d. Extract 21,23,..,29
x[-10:-1:2]

# e. Use list comprehension to create y = [31,32,...,40] to x
y = [i+31 for i in range(10)]
print(y)

# f. Make x be a list with numbers (0,1,...,40) with list y
x.extend(y)
x

# g. remove the number 33 from x
x.remove(33)
x

# h. delete numbers 31,32,...,40 in x
del x[31:]
x

# --------------------------------------------------------------
# 2. Create a dictionary looks like Team, Points, Result
var1 = ['Nuggets', 'Heat']
var2 = [94, 89]
var3 = ['Win', 'Lose']

Point_Board = {'Team': var1, 'Points': var2, 'Result': var3} 

# a. Extract Team and Points
print(Point_Board['Team'])
print(Point_Board['Points'])

# b. Extract Team in the first cell
print(Point_Board['Team'][0])

# c. Extract Points in the second cell
print(Point_Board['Points'][1])

# d. Change Points and Result
Point_Board['Points'] = [108, 111]
Point_Board['Result'] = ['Lose', 'Win']
print(Point_Board['Points'],Point_Board['Result'])

# ---------------------------------------------------------------------
import pandas as pd 
import numpy as np

df1 = pd.read_csv('NASDAQm.csv')
df2 = pd.read_csv('SP500m.csv')

# a. Transfer date
df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])

# b. Calculate log return
logr1 = np.diff(np.log(df1['Adj Close']))
logr2 = np.diff(np.log(df2['Adj Close']))

df1['logr'] = np.append(np.nan, logr1)
df2['logr'] = np.append(np.nan, logr2)

# c. Calculate simple return
sr1 = np.diff(df1['Adj Close'])/df1['Adj Close'][:-1]
sr2 = np.diff(df2['Adj Close'])/df2['Adj Close'][:-1]

df1['sr'] = np.append(np.nan, sr1)
df2['sr'] = np.append(np.nan, sr2)

# d. Use method describe
pd.set_option('display.expand_frame_repr', False) 
result1 = df1.describe()
result2 = df2.describe()

# e. Calculate skewness
import scipy.stats as stats

skew1 = df1.drop(columns = 'Date').apply(stats.skew, axis = 0, nan_policy = 'omit')
skew2 = df2.drop(columns = 'Date').apply(stats.skew, axis = 0, nan_policy = 'omit')

# Calculate kurtosis
kurt1 = df1.drop(columns = 'Date').apply(stats.kurtosis, axis = 0, nan_policy = 'omit')
kurt2 = df2.drop(columns = 'Date').apply(stats.kurtosis, axis = 0, nan_policy = 'omit')

# f. Combine the results, transfer to a "dataframe" with a column name
skew1 = pd.DataFrame(skew1, columns = ['skew'])    # should have a column name
skew2 = pd.DataFrame(skew2, columns = ['skew'])    # should have a column name
kurt1 = pd.DataFrame(kurt1, columns = ['kurt'])    # should have a column name
kurt2 = pd.DataFrame(kurt2, columns = ['kurt'])    # should have a column name

result1 = pd.concat([result1, skew1.transpose(), kurt1.transpose()])   
                    # Use the function 'pd.concat', # need to transpose skew1 and kurt1    
                                                   
result2 = pd.concat([result2, skew2.transpose(), kurt2.transpose()]) 

print(result1)

print(result2)

# g. Output result1 and result2, show correct representation of NaN
result1.to_csv('result1.csv', na_rep = 'NaN') 
result2.to_csv('result2.csv', na_rep = 'NaN')

# h. Calculate correlation matrix
print(df1[['logr','sr']].dropna())          # ignore NaN

print(df2[['logr','sr']].dropna())          # ignore NaN


# --------------------------------------------------------------------
# 4. PLot
# a. Plot probability mass function of a binominal distribution 
# with support n = 10, p = 0.2
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# values for x (all between 0 and 10):
x = np.linspace(0, 10, num=11)

# PMF for all these x's:
fx = stats.binom.pmf(x, 10, 0.2)

# plot:
plt.bar(x, fx, color = '0.6')
plt.xlabel('x')
plt.ylabel('fx')
plt.savefig('figure1.png')

# b. Plot probability density function of a normal distribution 
# with mean = 0.5, std = 1.2. 
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Support of normal density, 100 points, from -4 to 4.5 equally spaced
x_range = np.linspace(-4, 4.5, num = 100)

# PDF for all these x's:
pdf = stats.norm.pdf(x_range, loc = 0.5, scale = 1.2)

# plot:
plt.plot(x_range, pdf, linestyle='-', color = 'black')
plt.xlabel('x')
plt.ylabel('dx')
plt.savefig('figure2.png')

#----------------------------------------------------------------
# 5. Simulation for verifying unbiasedness
import numpy as np
import scipy.stats as stats

# a. Set the random seed:
np.random.seed(123456)

# b. The mean is 0.8333 and variance is 0.6944

# c. Set sample size:
n = 100

# Draw a sample given the population parameters:
sample1 = stats.expon.rvs(scale = 1/1.2, size = n)

# Calculate the sample average:
print(np.mean(sample1), np.var(sample1, ddof = 1))

# d. Draw a different sample and estimate again:
sample2 = stats.expon.rvs(scale = 1/1.2, size = n)
print(np.mean(sample2), np.var(sample2, ddof = 1))

# e. Repeat previous procedures 10,000 times 
# Initialize ybar to an array of length r = 10,000 to later store results:
r = 10000
lambda_mean = np.empty(r)       ## for storing estimate of mean
lambda_var = np.empty(r)        ## for storing estimate of variance

# Repeat 10,000 times:
for j in range(r):
    # draw a sample and store the sample mean in pos. j=0,1,... of ybar:
    sample = stats.expon.rvs(scale = 1/1.2, size = n)
    lambda_mean[j] = np.mean(sample)
    lambda_var[j] = np.var(sample, ddof = 1)    # ddof = 1 means divided by n-1

# f. Calculate means of the 10,000 sample mean and sample variance estimates:
print(np.mean(lambda_mean))
print(np.mean(lambda_var))