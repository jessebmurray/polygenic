import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# Basic model functions


def pdf(x_i, sigma_i=1, mu=0):
    """Returns the marginal (population) pdf of X_i ~ Normal(mu, sigma_i^2)."""

    return st.norm.pdf(x_i, scale=sigma_i, loc=mu)


def stable_rs(r):
    """Calculates r_s from r under stable population variance, where
    r^2 + r_s^2 = 1"""

    return np.sqrt(1 - np.square(r))


# Conditional descendant distribution parameters

def get_mu_tilda(x_i, r, n):
    """Calculates the conditional descendant normal distribution *expectation*
    for generation-gap n.

    Latex equation:
    tilde{\mu}_{i+n} = r^n X_i
    (See the paper for the derivation.)"""

    return (r**n) * x_i


def get_sigma_tilda(sigma_i, r, rs, n):
    """Calculates the conditional descendant normal distribution *standard deviation*
    (SD) for generation-gap n.

    Latex equation for the variance (square of the SD):
    tilde{\sigma}_{i+n}^2 = [(r^2+r_s^2)^n - r^{2n}] \sigma_i^2
    (See the paper for the derivation.)"""

    # Get the added part of the square root
    add_part = (np.square(r) + np.square(rs)) ** n
    # Get the subtracted part of the square root
    subtract_part = r ** (2*n)

    return sigma_i * np.sqrt(add_part - subtract_part)


# Ancestor and descendant bounds


def get_percentile_bounds(m):
    """Returns an m x 2 percentile-bounds matrix of the lower and upper bounds
    of the percentiles.
    For example: (0, 0.2),.., (0.8, 1). Where there are m equal-size continuous
    percentile sets.

    See the "Percentile transition matrices" section of the paper, where the percentile
    sets are denoted Q_1, Q_2,..., Q_m. Here, row j corresponds to Q_j.

    See the `test_percentile_bounds()` function for an example output."""

    lower_percentiles = np.linspace(start=0, stop=1, num=m, endpoint=False)
    upper_percentiles = lower_percentiles + (1 / m)

    return np.column_stack((lower_percentiles, upper_percentiles))


def get_real_bounds(m, sigma):
    """Returns an m x 2 real-bounds matrix of the lower and upper real-number values.
    Obtains a percentile-bounds matrix and converts to the real line.

    Uses the normal `ppf` or percent point function (inverse of cdf) for the conversion.

    Assumes for simplicity a location (population mean) of zero (standardization).

    Note that percentiles of 0 and 1 are converted to -infinity and +infinity,
    respectively."""

    percentile_bounds = get_percentile_bounds(m=m)

    return st.norm.ppf(q=percentile_bounds, loc=0, scale=sigma)


def expand_real_bounds(real_bounds, x_i, sigma_i, r, rs, n):
    """Converts real bounds into z-scores of the conditional distribution(s) of x_i.
    That is, it converts the score relative to the conditional descendant distribution into
    z-scores relative to the conditional distributions of the ancestor score(s) x_i, which
    can be a scalar or vector.

    This is the same as, in the paper (D_ - mu_tilda) / sigma_tilda in the
    P_n(D, x_i) equation, which is calculated for each D_min and D_max in the real_bounds.

    Note that the real_bounds are generally descendant_bounds, which are real values
    corresponding to percentiles of the marginal (population) descendant distribution.
    That distribution is normal with 0 mean and sigma_n SD.
    We input sigma_i, which is the SD of the ancestor distribution, it is needed to
    calculate sigma_tilda, which is the same for each x_i.

    The conditional means are shaped into three dimensions. We do this because the
    real_bounds is an (m x 2) matrix. The size of x_i will be n_iters in the get_matrix
    function. Then, when we subtract the conditional means from the real_bounds,
    we get an (n_iters x m x 2) array. That is, each 'row' (0th dimension) is a
    conditionalized real_bound.
    """

    # Get the conditional mean (mu_tilda), which has the same shape as x_i
    # (scalar or vector)
    mu_tilda = get_mu_tilda(x_i=x_i, r=r, n=n)
    # Reshape mu_tilda into three dimensions, with as many rows (0th dimension)
    # as needed to fit in the size of mu_tilda (1 if scalar, more if vector)
    mu_tilda = np.reshape(mu_tilda, (-1, 1, 1))

    # Get the conditional SD (sigma_tilda), which is the same for all conditional
    # descendant distributions. That is, sigma_tilda is a scalar.
    sigma_tilda = get_sigma_tilda(sigma_i, r, rs, n)

    # Return the (n_iters x m x 2) array
    return (real_bounds - mu_tilda) / sigma_tilda


# State to set probability


def get_state_set(m_descendant, x_i, r, rs, n, sigma_i):
    """
    Calculates the the state to set probabilities for m_descendant equally spaced
    (by percentile) sets, where a set is referred to in the paper as D.

    This function carries out the P_n(D, x_i) calculation from the paper for each D.

    The input x_i is a vector (of size n_iters) or could even be a scalar. In the
    get_matrix function, x_i is a vector of evenly spaced ancestor states over an
    ancestor bound. In the paper, this is denoted by x_i \in A.

    For an (m_descendant x 2) real-bound matrix, or 1 by 2 element thereof,
    returns the state to set probability.
    Requires the right element of each 1 by 2 a-vector element to be greater than the
    left element (tested elsewhere)."""

    # SD of the marginal (population) descendant distribution
    sigma_n = np.sqrt((np.square(r) + np.square(rs)) ** n) * sigma_i
    # Calculate the real descendant bounds
    descendant_bounds = get_real_bounds(m=m_descendant, sigma=sigma_n)

    # Get the expanded (conditionalized) bounds according to x_i
    expanded_bounds = expand_real_bounds(descendant_bounds, x_i, sigma_i, r, rs, n)

    # Convert z-score to cdf
    expanded_bounds = st.norm.cdf(expanded_bounds)

    # Take the difference (along the last axis, which has size 2)
    # This gets the area/probability between the m lower and upper bounds for each x_i
    probabilities = np.diff(expanded_bounds)
    # Remove the axis of length one (the last one which was collapsed when taking diff)
    probabilities = np.squeeze(probabilities)

    # Return the conditional probabilities scaled by the densities of the x_i
    # The output is transposed so that it is a matrix of shape (m_descendant x n_iters)
    return probabilities.T * pdf(x_i)


# Percentile transition matrix

def trim_real_bounds(real_bounds, trim_score=5):
    """
    Symmetrically trim the ends of a real_bounds matrix to an absolute-value trim_score.
    This is done so that manual integration is possible for the tail bounds (and manual
    integration cannot be accomplished over an infinite length). The approximation works
    because the density at a substantial trim_score (distance) from the mean will be so low,
    that integrating further adds an immaterial amount of area (probability).

    It should be noted that the trim_score is equal to the z-score if and only if the SD of
    the real_bounds is 1. This the case for the ancestor_bounds in the get_matrix function.
    """

    real_bounds[np.isneginf(real_bounds)] = -1 * trim_score
    real_bounds[np.isposinf(real_bounds)] = trim_score

    return real_bounds


def get_x_i_matrix(m_ancestor, trim_score, num_iters, sigma_i):
    """
    Obtain a (m_ancestor x num_iters) matrix, where each row is the vector
    of x_i for each of the m_ancestor real sets (couples).
    """
    # Calculate the bounds for the ancestor states
    ancestor_bounds = get_real_bounds(m=m_ancestor, sigma=sigma_i)
    ancestor_bounds = trim_real_bounds(ancestor_bounds, trim_score=trim_score)

    # Get the line-space from the lower bound to upper bound for each
    # of the m_ancestor couples.
    x_i_matrix = np.linspace(ancestor_bounds[:, 0], ancestor_bounds[:, 1],
                             num=num_iters, axis=1)

    return x_i_matrix


def get_matrix(r, rs, n=1, num_iters=100_000, trim_score=5, m_descendant=5, m_ancestor=5):
    """
    Obtain an (m_descendant x m_ancestor) percentile transition matrix.
    As described in the paper, m_ancestor = m_descendant. However, this function allows
    for the flexibility of different m's, if desired.

    There are num_iters number of iterations over the numerically calculated integrals of
    each entry in the matrix. As there are m_descendant x m_ancestor entries in the matrix,
    that means num_iters x m_descendant x m_ancestor total iterations.
    """

    # Set sigma_i (the marginal ancestor distribution SD) to be equal to one
    sigma_i = 1
    x_i_matrix = get_x_i_matrix(m_ancestor=m_ancestor, trim_score=trim_score,
                                num_iters=num_iters, sigma_i=sigma_i)

    # Initialize the percentile transition matrix
    matrix = np.zeros((m_descendant, m_ancestor))

    # Loop through the ancestor states, filling in the columns of the matrix one by one
    for j in range(m_ancestor):
        # Get the x_i vector for the jth ancestor set (out of m_ancestor)
        # The resultant x_i has size num_iters
        x_i = x_i_matrix[j]

        # Calculate the state to set probabilities: P_n(D, x_i) in the paper
        # for each of the (m_descendant) descendant states.
        state_set = get_state_set(m_descendant=m_descendant, x_i=x_i, sigma_i=sigma_i,
                                  r=r, rs=rs, n=n)

        # Numerical integration of the probabilities to obtain the total probability/area
        # within each element of column j of the percentile transition matrix
        matrix[:, j] = np.trapz(state_set, x_i)

    # End for loop

    # Because we want to obtain the probability for each ancestor state (rather than
    # the overall probability), we normalize to the probability of an ancestor state.
    # This is the same as doing: matrix /= matrix.sum(axis=0)
    ancestor_state_probability = 1 / m_ancestor
    matrix /= ancestor_state_probability

    return matrix


# Plotting functions


def plot_ax(ax, matrix, i=0, j=0, title=None, title_loc='left', x_label=True, child=False):
    """Plots a percentile transition matrix on an axis."""

    from matplotlib.ticker import PercentFormatter
    ancestors = ['Parent', 'Grandparent', 'Great-Grandparent', 'Great-Great-Grandparent',
                 'Great$^3$-Grandparent', 'Great$^4$-Grandparent']
    # ancestors = ['Generation {}'.format(i) for i in range(1, 20)]

    if title:
        ax.set_title(title, fontsize=17, loc=title_loc)
    if matrix.shape[1] == 5:
        step_labels = ['Bottom', 'Second', 'Third', 'Fourth', 'Top']
        if x_label:
            ax.set_xlabel('{}\'s Quintile'.format(ancestors[i]), fontsize=15)
        else:
            if j >= 4:
                ax.set_xlabel('{}\'s Quintile'.format(ancestors[i]), fontsize=15)
            else:
                ax.set_xlabel(' '.format(ancestors[i]), fontsize=15)
        # ax.set_xlabel("Generation {} Quintile".format(i+1), fontsize=15)
        if j % 2 == 0:
            if child:
                ax.set_ylabel('Cumulative Probability of Child\'s Quintile', fontsize=15)
            else:
                ax.set_ylabel('Cumulative Probability of Descendant\'s Quintile', fontsize=15)
    else:
        step_labels = list(range(1, matrix.shape[1] + 1))

    pal = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5'][::-1]

    ax.set_ylim(0, 1)
    values_sum_list = [1] * matrix.shape[1]
    for j in range(len(matrix) - 1, -1, -1):
        if len(matrix) <= 5:
            ax.bar(step_labels, [- value for value in matrix[j]],
                   bottom=values_sum_list, color=pal[j])
        else:
            ax.bar(step_labels, [- value for value in matrix[j]],
                   bottom=values_sum_list)

        for a, b, c in zip(step_labels, values_sum_list, [value for value in matrix[j]]):
            if c >= 0.01:
                num = (b - c / 2) - 0.018
                color = 'w'
                if j >= 2:
                    color = 'k'
                round_str = "{:0.0%}"
                if i > 3:
                    round_str = "{:0.1%}"
                ax.text(a, num, ' ' + round_str.format(c),
                        va='bottom', ha='center', color=color, size=13, alpha=0.8)

        for k in range(len(values_sum_list)):
            values_sum_list[k] -= matrix[j][k]

    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(step_labels, Fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(1))


def plot_matrix(matrix, n=1, child=True, legend=True):
    """Plots a figure with only one percentile transition matrix."""
    fig, axes = plt.subplots(1, 1, figsize=(13 * 0.95 * 0.75, 8 / 0.95 * 0.75))
    plot_ax(ax=axes, matrix=matrix, i=n-1, child=child)
    term = 'Descendant'
    if matrix.shape[1] == 5:
        if n == 1:
            term = 'Child'
        if legend:
            legend = ['{} in the\nTop Quintile'.format(term), 'Fourth Quintile',
                      'Third Quintile', 'Second Quintile', 'Bottom Quintile']
            fig.legend(legend, bbox_to_anchor=(1, 0.977), loc="upper left", fontsize=15)
    plt.tight_layout()


def get_rv_rsv(mv):
    """Get the corresponding r vector and rs vector from a mobility vector"""
    rv = 1 / np.sqrt(mv**2 + 1)
    rsv = stable_rs(rv)
    return rv, rsv


def report_mobility(mv, rv, rsv, i):
    """Give a 'report' of the mobility, and corresponding regression and residual
    coefficients.
    mv is short for mobility_vector (a vector of mobility values)"""

    # If the mobility value is an integer, display it as an integer
    if mv[i] % 1 == 0:
        return "$m$ = {:.0f}, $r$ = {:.3f}, $r_s$ = {:.3f}".format(mv[i], rv[i], rsv[i])
    # Otherwise, display the first decimal of the mobility value
    else:
        return "$m$ = {:.1f}, $r$ = {:.3f}, $r_s$ = {:.3f}".format(mv[i], rv[i], rsv[i])


# Functions for handling (Pearson) data


def get_percentiles(vector):
    """Convert an vector of data into percentiles"""
    return st.rankdata(vector) / vector.size


def get_matrix_data(x, y, m_ancestor=5, m_descendant=5, return_raw=False):
    """Obtains the observed percentile transition matrix from data.
    x is the ancestor values and y is the descendant values (typically parent-child
    parallel vectors).

    If return_raw = True, then the counts are returned, rather than the proportions.
    To estimate the probability (and obtain an estimated percentile transition matrix)
    it is necessary that return_raw = False.
    """

    # Create (representing percentiles) of the data
    bins_ancestor = np.linspace(0, 1, m_ancestor, endpoint=False)
    bins_descendant = np.linspace(0, 1, m_descendant, endpoint=False)

    # Obtain the bin for each data-point based on its percentile
    xb = np.digitize(get_percentiles(x), bins_ancestor)
    yb = np.digitize(get_percentiles(y), bins_descendant)

    # Initialize the percentile transition matrix
    matrix = np.zeros((m_ancestor, m_descendant))

    # Loop through the ancestor bins
    for i in range(m_ancestor):
        # Get the descendants of this ancestor bin
        desc = xb[yb == i+1]

        # Loop through the descendant bins
        for j in range(m_descendant):

            if return_raw:
                # Get the total number of descendants in the
                # ancestor bin, descendant bin pair
                matrix[j, i] = np.sum(desc == j+1)
            else:
                # Get the proportion of descendants in the
                # ancestor bin, descendant bin pair (approximates probability)
                matrix[j, i] = np.mean(desc == j+1)

    # End of for loop

    return matrix


# Testing functions


def test_percentile_bounds():
    expected = np.array([[0., 0.2],
                        [0.2, 0.4],
                        [0.4, 0.6],
                        [0.6, 0.8],
                        [0.8, 1.]])
    assert np.allclose(get_percentile_bounds(m=5).ravel(), expected.ravel())


def test_expanded_real_bounds():
    """Test that this gives the correct shape. (Further tests can be added"""
    x_i_trial = np.array([1, 2, 3])
    rb = np.array([[-5., -0.84162123],
                   [-0.84162123, -0.2533471],
                   [-0.2533471, 0.2533471],
                   [0.2533471, 0.84162123],
                   [0.84162123, 5.]])
    expand_real_bounds(real_bounds=rb, x_i=x_i_trial, sigma_i=1, r=0.5, rs=0.9, n=1)
    assert (rb - np.reshape([1, 2, 3], (-1, 1, 1))).shape == (3, 5, 2)


def test_trim_real_bounds():
    rb = get_real_bounds(m=5, sigma=1)
    trim_score = 4
    rb[0, 0] = -1 * trim_score
    rb[-1, -1] = trim_score
    assert (rb == trim_real_bounds(rb, trim_score)).all()


def test_functions():
    test_percentile_bounds()
    test_expanded_real_bounds()
    test_trim_real_bounds()


test_functions()
print('Tests passed')
