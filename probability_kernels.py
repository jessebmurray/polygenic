import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# Basic model functions

def pdf(x_i, sigma_i=1, mu=0):
    """Returns the pdf of X_i ~ Normal(mu, sigma_i^2)."""
    return st.norm.pdf(x_i, scale=sigma_i, loc=mu)


def stable_rs(r):
    """Calculates r_s from r under stable population variance."""
    return np.sqrt(1 - np.square(r))


# Conditional descendent distribution

def mu_tilda(x_i, r, n):
    return (r**n)*x_i


def sigma_tilda(sigma_i, r, rs, n):
    return sigma_i * np.sqrt(
            (np.square(r)+np.square(rs))**n - r**(2*n))


# A (or D) vector


def get_a_pers(n_subs):
    return np.transpose(
        np.array([np.linspace(0, 1, n_subs, endpoint=False),
                     np.linspace(0, 1, n_subs, endpoint=False)+1/n_subs]))


def convert_a_per(a_vec_per, sigma):
    """Converts an a_vector of percentiles to the real line."""
    return st.norm.ppf(a_vec_per, scale=sigma)


def normalize_a(a_vec, x_i, sigma_i, r, rs, n):
    """Takes in a converted a_vector (on the real line)."""
    return (a_vec - np.reshape(mu_tilda(x_i, r, n), (-1, 1, 1))
                       ) / sigma_tilda(sigma_i, r, rs, n)


# State to set probability


def state_set(a_vec, x_i, sigma_i, r, rs, n):
    """For an n_subs by 2 a-vector, or 1 by 2 element thereof,
    returns the state to set probability.
    Requires the right element of each 1 by 2 a-vector element to be greater than the
    left element (tested elsewhere)."""
    return np.diff(
        st.norm.cdf( # taking the transpose doesn't slow it down
        normalize_a(a_vec, x_i, sigma_i, r, rs, n))).transpose()[0] * pdf(x_i)


# Percentile transition matrix


def get_matrix(r, rs, n=1, num_iters=100_000, low_round=5, n_subs0=5, n_subs1=5, sigma_i=1,
               return_raw=False):
    sigma_n = np.sqrt((np.square(r) + np.square(rs))**n)*sigma_i

    a_vec = convert_a_per(get_a_pers(n_subs=n_subs1), sigma=sigma_n)
    # assert (np.diff(a_vec) > 0).all()

    d_vec = convert_a_per(get_a_pers(n_subs=n_subs0), sigma=sigma_i)
    d_vec[np.isneginf(d_vec)] = -1 * low_round
    d_vec[np.isposinf(d_vec)] = low_round
    d_lines = np.linspace(d_vec[:, 0], d_vec[:, 1], num=num_iters, axis=1)
    # assert d_lines.shape == (d_vec.shape[0], num_iters)

    matrix = np.zeros((a_vec.shape[0], d_vec.shape[0]))

    for j in range(d_vec.shape[0]):
        matrix[:, j] = np.trapz(state_set(
                        a_vec, x_i=d_lines[j], sigma_i=sigma_i, r=r, rs=rs, n=n),
                                d_lines[j])
    # assert np.allclose(matrix.sum(axis=0), matrix.sum(axis=1))  # fails for low `num_iters`
    matrix /= matrix.sum(axis=0)
    return matrix


# Plot a matrix


def plot_ax(ax, matrix, i=0, j=0, title=None, title_loc='left', x_label=True, child=False):
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
            ax.bar(step_labels, [- value for value in matrix[j]], bottom=values_sum_list, color=pal[j])
        else:
            ax.bar(step_labels, [- value for value in matrix[j]], bottom=values_sum_list)

        for a, b, c in zip(step_labels, values_sum_list, [value for value in matrix[j]]):
            if c >= 0.01:
                num = (b - c / 2) - 0.018
                color = 'w'
                if j >= 2:
                    color = 'k'
                round_str = "{:0.0%}"
                if i > 3:
                    round_str = "{:0.1%}"
                ax.text(a, num, ' ' + round_str.format(c), va='bottom', ha='center', color=color, size=13, alpha=0.8)

        for k in range(len(values_sum_list)):
            values_sum_list[k] -= matrix[j][k]

    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(step_labels, Fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(1))


# Plot a figure with only one matrix


def plot_matrix(matrix, n=1, child=True, legend=True):
    fig, axes = plt.subplots(1, 1, figsize=(13 * 0.95 * 0.75, 8 / 0.95 * 0.75))
    plot_ax(ax=axes, matrix=matrix, i=n-1, child=child)
    term = 'Descendant'
    if matrix.shape[1] == 5:
        if n == 1:
            term = 'Child'
        if legend:
            legend = ['{} in the\nTop Quintile'.format(term), 'Fourth Quintile', 'Third Quintile', 'Second Quintile',
                  'Bottom Quintile']
            fig.legend(legend, bbox_to_anchor=(1, 0.977), loc="upper left", fontsize=15)
    plt.tight_layout()


# Handling data


def get_percentiles(array):
    """Convert data to percentiles"""
    return st.rankdata(array) / array.size


def get_matrix_data(x, y, n_subs0=5, n_subs1=5, return_raw=False):
    """Obtains the observed percentile transition matrix from data"""
    bins0 = np.linspace(0, 1, n_subs0, endpoint=False)
    bins1 = np.linspace(0, 1, n_subs1, endpoint=False)

    xb = np.digitize(get_percentiles(x), bins0)
    yb = np.digitize(get_percentiles(y), bins1)

    matrix = np.zeros((n_subs0, n_subs1))

    for i in range(n_subs0):
        desc = xb[yb == i + 1]
        for j in range(n_subs1):
            if return_raw:
                matrix[j, i] = np.sum(desc == j + 1)
            else:
                matrix[j, i] = np.mean(desc == j + 1)

    return matrix


def report_mobility(mv, rv, rsv, i):
    if mv[i] % 1 == 0:
        return "$m$ = {:.0f}, $r$ = {:.3f}, $r_s$ = {:.3f}".format(mv[i], rv[i], rsv[i])
    else:
        return "$m$ = {:.1f}, $r$ = {:.3f}, $r_s$ = {:.3f}".format(mv[i], rv[i], rsv[i])
