import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


def normal_qq(data):
    """Get the Q-Q for the normal distribution."""
    n = data.size
    theor = data.std() * st.norm.ppf(
        np.arange(1, n + 1) / (n + 1)) + data.mean()
    return theor, np.sort(data)


def f_test(numer, denom):
    """Calculate the F test and the corresponding p-value for a
    numerator and denominator."""
    numer = np.array(numer)
    denom = np.array(denom)
    # Calculate F test statistic
    f = np.var(numer, ddof=1) / np.var(denom, ddof=1)
    # Define the degrees of freedom numerator
    dfn = numer.size - 1
    # Define the degrees of freedom denominator
    dfd = denom.size - 1
    # Get the p-value of the F test statistic
    p = 1 - st.f.cdf(f, dfn, dfd)
    return {'statistic': f, 'pvalue': p}


def check_normality(array, rn=5):
    print("""
          The null hypothesis for all these tests that the
          population is drawn from a normal distribution.
          Thus, the p-values should all be greater than 0.05.""", end='\n\n')

    print('Skew =', np.round(st.skew(array), rn))
    print(st.skewtest(array), end='\n\n')

    print('Kurtosis =', np.round(st.kurtosis(array), rn))
    print(st.kurtosistest(array), end='\n\n')

    print('D\'Agostino and Pearson',
          st.normaltest(array), sep='\n', end='\n\n')


# Plotting functions

def scatter_plot(x, y):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5, facecolors='none', edgecolors='#1f77b4')
    plt.grid(alpha=0.5)
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))
    plt.xlabel('Parent score')
    plt.ylabel('Child score')
    plt.gca().set_aspect('equal', adjustable='box')


def plot_normal_qq(data, lim=3.5):
    plt.figure(figsize=(5, 5))
    x_theor, x_sample = normal_qq(data)

    plt.plot([-5, 5], [-5, 5], color='grey')
    plt.scatter(x_theor, x_sample, alpha=0.6,
                facecolors='none', edgecolors='#1f77b4')
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.grid(alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Theoretical')
    plt.ylabel('Observed')
    plt.tight_layout()


def plot_residuals_by_parent(true_bins, resid_means, resid_cis):
    plt.figure(figsize=(5, 4))
    plt.plot(true_bins, resid_means, color='black', linewidth=2)
    plt.errorbar(true_bins, resid_means, yerr=resid_cis.T,
                 color='grey', alpha=1, linewidth=1.4)
    plt.axhline(y=0, color='grey')
    plt.grid(alpha=0.3)
    plt.xlabel('Parent score')
    plt.ylabel('Child residual')
    plt.tight_layout()
