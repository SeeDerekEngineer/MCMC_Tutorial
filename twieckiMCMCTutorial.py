#%matplotlib inline

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm


sns.set_style('white')
sns.set_context('talk')

np.random.seed(123)

data = np.random.randn(20)

ax = plt.subplot()
sns.distplot(data, kde=False, ax=ax)
_ = ax.set(title='Histogram of observed data', xlabel='x', ylabel='# observations');
plt.show()

#We can do this because the normal prior for mu is conjugate to the posterior distribution.
def calc_posterior_analytical(data, x, mu_0, sigma_0):
    sigma = 1.
    n = len(data)
    mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
    sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
    return norm(mu_post, np.sqrt(sigma_post)).pdf(x) #The probability density function of the data for
                                                     #a normal distribution of mu equal to mu_post
                                                     #and sigma equal to np.sqrt(sigma_post).
                                                     #For every mu value, x, what is the
                                                     #probability it shows up in the previously defined
                                                     #normal distribution, norm(mu_post, np.sqrt(sigma_post)).

ax = plt.subplot()
x = np.linspace(-1, 1, 500) #500 numbers between -1 and 1, meant to represent the mu values of our data.
posterior_analytical = calc_posterior_analytical(data, x, 0., 1.)
ax.plot(x, posterior_analytical)
ax.set(xlabel='mu', ylabel='belief', title='Analytical posterior');
sns.despine()
plt.show()

ax = plt.subplot()
x = np.linspace(-1, 1, 8) #Only five points will get plotted, with x being the random point, and y being the probability.
posterior_analytical = calc_posterior_analytical(data, x, 0., 1.)
ax.plot(x, posterior_analytical)
ax.set(xlabel='mu', ylabel='belief', title='Analytical posterior');
sns.despine()
plt.show()


#Up to this point, this is calculating the posterior, having seen the data (literally, the data variable)
#taking the prior (assumed to be the Normal Distribution) into account.
#But what if the prior wasn't a Normal Distribution.