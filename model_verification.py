import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


def normal_qq(data):
    """Get the Q-Q for the normal distribution.
    Returns the theoretical values and the order statistics to be plotted against
    each other.
    For a normal distribution, we expect Phi(x_(k)) (the cdf of the kth order
    statistic) to be approximately k / (n+1).
    Hence, the theoretical z-score should be Phi^-1( k/(n+1) ). That is, the inverse
    cdf of k/(n+1). To convert the theoretical z-score to to to theoretical (actual)
    x-score, we multiply by the population standard deviation and add the population
    mean.

    The data argument must be a numpy array.
    """
    # Get the number of data points
    n = data.size
    # Get the k values (for the order statistics) from 1 through n
    k = np.arange(1, n + 1)
    # Get the population standard deviation
    sigma = data.std()
    # Get the population mean
    mu = data.mean()
    # Calculate the theoretical data values
    theor = sigma * st.norm.ppf(k / (n + 1)) + mu
    # Return the theoretical values, and the order statistics
    return theor, np.sort(data)


def f_test(numer, denom):
    """Calculate the F test and the corresponding p-value for a
    numerator and denominator.

    The numerator and denominator arrays can be numpy arrays or lists."""
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
          The null hypothesis for all of these tests is that 
          the population is drawn from a normal distribution.
          Thus, the p-values should all be greater than 0.05.""", end='\n\n')

    print('Skew =', np.round(st.skew(array), rn))
    print(st.skewtest(array), end='\n\n')

    print('Kurtosis =', np.round(st.kurtosis(array), rn))
    print(st.kurtosistest(array), end='\n\n')

    print('D\'Agostino and Pearson',
          st.normaltest(array), sep='\n', end='\n\n')


# Plotting functions

def scatter_plot(x, y, lim=4):
    """Simple square scatter plot with light grid lines and hollow blue circular
    data points. The limit (lim) argument provides the upper and lower bound of the
    x and y axes for the (square) plot. """
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5, facecolors='none', edgecolors='#1f77b4')
    plt.grid(alpha=0.5)
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.xlabel('Parent score')
    plt.ylabel('Child score')
    # Make the plot square
    plt.gca().set_aspect('equal', adjustable='box')


def plot_normal_qq(data, lim=3.5):
    """Plots the theoretical values (x-axis) against the order statistics (y-axis)
    to see if the points lie on an approximate straight line (with gradient
    population SD and intercept population mean).
    The limit (lim) argument provides the upper and lower bound of the x and y
    axes for the (square) plot."""
    plt.figure(figsize=(5, 5))
    x_theor, x_sample = normal_qq(data)

    plt.plot([-5, 5], [-5, 5], color='grey')
    plt.scatter(x_theor, x_sample, alpha=0.6,
                facecolors='none', edgecolors='#1f77b4')
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.grid(alpha=0.3)
    # Make the plot square
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
