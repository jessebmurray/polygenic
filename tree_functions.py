import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.ticker import PercentFormatter

# from scipy.special import erf

# Global variables that would be used in main
number_of_iterations = 100
z_range = 8
rMain = 0.9
rsMain = 0.9
mean_gen = 0
sd_gen = 1
k_val = -2
percent_step = 0.33

# Global variables that are used in the module to fix issues that result from the base 2 arithmetic in python
ROUND_NUMBER = 6


# BASE DISTRIBUTION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def f_norm(x, mean, sd):
    return (1 / (sd * ((2 * np.pi) ** 0.5))) * np.exp(-1 * ((((x - mean) / sd) ** 2) / 2))


def normal_distribution(number_of_steps, z_score_range, mean=0, sd=1, above_k_v=None, below_k_v=None):
    # This creates a normal distribution with a certain number of steps. The motivation for using number of steps is
    # that it is related to the number of operations. It can make all y values 0, for x below k_v
    two_d_distribution = list()
    bound = z_score_range * sd
    number = number_of_steps
    increment = bound / number
    if above_k_v is not None:
        above_n = ((0.5 * z_score_range * sd) + above_k_v - mean) / increment
    else:
        above_n = 0
    if below_k_v is not None:
        below_n = ((0.5 * z_score_range * sd) + below_k_v - mean) / increment
    else:
        below_n = 0
    x_start = mean - (bound / 2)
    x = round(x_start, ROUND_NUMBER)
    for num in range(number + 1):
        x_y_pair = [0, 0]
        go_ahead = True
        if above_k_v is not None:
            if num < above_n:
                x_y_pair = [x, 0]
                go_ahead = False
        if below_k_v is not None:
            if num > below_n:
                x_y_pair = [x, 0]
                go_ahead = False
        if go_ahead:
            x_y_pair = [x, f_norm(x, mean, sd)]
        two_d_distribution.append(x_y_pair)
        x = round(x_start + (increment * (num + 1)), ROUND_NUMBER)
    two_d_distribution[0] += [['increment', increment], ['number', number], ['bound', bound], ['mean', mean],
                              ['sd', sd]]
    return two_d_distribution


def one_offspring_distribution(parent_distribution, index_num, r, r_s, above_k_v=None, below_k_v=None):
    # we need a function that takes a value from the parent distribution and multiplies every value in the offspring
    # distribution by that value, essentially scaling it by that value.
    # Also the x values in the offspring distribution need to be shifted by r * z_p
    parent_mean = parent_distribution[0][5][1]
    shift = r * (parent_distribution[index_num][0] - parent_mean)
    offspring_mean = parent_mean + shift
    parent_sd = parent_distribution[0][6][1]
    offspring_sd = r_s * parent_sd
    scale_factor = parent_distribution[index_num][1]
    number = parent_distribution[0][3][1]
    z_score_range = parent_distribution[0][4][1] / offspring_sd
    offspring_distribution = normal_distribution(number, z_score_range, offspring_mean, offspring_sd, above_k_v,
                                                 below_k_v)
    for row in offspring_distribution:
        row[1] *= scale_factor
    offspring_distribution[0] += [['parent mean', parent_mean]]
    return offspring_distribution


def offspring_distributions(parent_distribution, r, r_s, above_k_v_p=None, below_k_v_p=None,
                            above_k_v_o=None, below_k_v_o=None):
    parent_increment = parent_distribution[0][2][1]
    parent_bound = parent_distribution[0][4][1]
    parent_mean = parent_distribution[0][5][1]

    if above_k_v_p is not None:
        above_num = (above_k_v_p - parent_mean + (parent_bound * 0.5)) / parent_increment
    else:
        above_num = 0
    if below_k_v_p is not None:
        below_num = (below_k_v_p - parent_mean + (parent_bound * 0.5)) / parent_increment
    else:
        below_num = 0

    all_offspring_distributions = list()
    for index in range(len(parent_distribution)):
        go_ahead = True
        if above_k_v_p is not None:
            if index < above_num:
                go_ahead = False
        if below_k_v_p is not None:
            if index > below_num:
                go_ahead = False
        if go_ahead:
            all_offspring_distributions.append(one_offspring_distribution(parent_distribution, index, r, r_s,
                                                                          above_k_v_o, below_k_v_o))
    # add the parent area to the top of offspring distributions
    parent_area = area_under_one_distribution(parent_distribution)
    all_offspring_distributions[0][0] += [['parent area', parent_area]]
    return all_offspring_distributions


# SUPERIMPOSED DISTRIBUTION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def superimposed_offspring_distribution(distributions):
    set_of_x = set()
    for distribution in distributions:
        set_of_x.update([row[0] for row in distribution])
    list_of_x = sorted(list(set_of_x))
    superimposed_distribution = [[x, 0] for x in list_of_x]
    for superimposed_row in superimposed_distribution:
        value = superimposed_row[0]
        for distribution in distributions:
            for row in distribution:
                if row[0] == value:
                    superimposed_row[1] += row[1]
    # the below increment is wrong in certain cases, but it doesn't matter because it is never used
    increment = round(superimposed_distribution[1][0] - superimposed_distribution[0][0], ROUND_NUMBER)

    parent_increment = distributions[0][0][2][1]
    parent_mean = distributions[0][0][7][1]
    parent_area = distributions[0][0][8][1]

    # Jan 2020
    parent_number = distributions[0][0][3][1]
    parent_bound = distributions[0][0][4][1]

    superimposed_distribution[0] += [['increment', increment], ['parent increment', parent_increment],
                                     ['parent mean', parent_mean], ['parent area', parent_area],  # Jan 2020
                                     ['parent number', parent_number], ['parent bound', parent_bound]]
    return superimposed_distribution


def normalized_superimposed_distribution_to_parent_increment(superimposed_distribution):
    parent_increment = superimposed_distribution[0][3][1]
    parent_mean = superimposed_distribution[0][4][1]
    smallest_x = superimposed_distribution[0][0]
    n = int(abs(smallest_x - parent_mean) / parent_increment)
    par_inc_norm_superimposed_distribution = list()
    for num in range((2 * n) + 1):
        x_value_prev = round((num - n - 1) * parent_increment, ROUND_NUMBER)
        x_value = round((num - n) * parent_increment, ROUND_NUMBER)
        par_inc_norm_superimposed_distribution.append([x_value, 0])
        for row in superimposed_distribution:
            if x_value_prev < row[0] <= x_value:
                par_inc_norm_superimposed_distribution[num][1] += row[1]
            # ideally we'd like to stop this loop once row[0] is greater than the x_value
    par_inc_norm_superimposed_distribution[0] += [['increment', parent_increment]]
    par_inc_norm_superimposed_distribution[0] += superimposed_distribution[0][3:]
    return par_inc_norm_superimposed_distribution


def final_superimposed_distribution_all_not_area_adj(parent_distribution, r, r_s):
    offspring_distributions_all = offspring_distributions(parent_distribution, r, r_s)
    super_offspring_distribution_all = superimposed_offspring_distribution(offspring_distributions_all)
    par_inc_super_offspring_distribution_all = \
        normalized_superimposed_distribution_to_parent_increment(super_offspring_distribution_all)
    return par_inc_super_offspring_distribution_all


def normalized_superimposed_distribution_to_parent_area(superimposed_distribution, area_scale_factor):
    par_area_norm_superimposed_distribution = \
        [[row[0], row[1] / area_scale_factor] for row in superimposed_distribution]
    par_area_norm_superimposed_distribution[0] += superimposed_distribution[0][2:]
    return par_area_norm_superimposed_distribution


def final_superimposed_distribution(parent_distribution, r, r_s, above_k_v_p=None, below_k_v_p=None,
                                    above_k_v_o=None, below_k_v_o=None):
    offspring_distributions_ = offspring_distributions(parent_distribution, r, r_s, above_k_v_p, below_k_v_p,
                                                       above_k_v_o, below_k_v_o)
    super_offspring_distribution = superimposed_offspring_distribution(offspring_distributions_)
    par_inc_super_offspring_distribution = \
        normalized_superimposed_distribution_to_parent_increment(super_offspring_distribution)
    par_inc_super_offspring_distribution_all = \
        final_superimposed_distribution_all_not_area_adj(parent_distribution, r, r_s)
    parent_area_factor = area_scale_factor_entire(par_inc_super_offspring_distribution_all)
    par_area_super_offspring_distribution = \
        normalized_superimposed_distribution_to_parent_area(par_inc_super_offspring_distribution, parent_area_factor)
    return par_area_super_offspring_distribution


def final_superimposed_distribution_all_area_adj(parent_distribution, r, r_s):
    par_inc_super_offspring_distribution_all = \
        final_superimposed_distribution_all_not_area_adj(parent_distribution, r, r_s)
    parent_area_factor = area_scale_factor_entire(par_inc_super_offspring_distribution_all)
    par_area_super_offspring_distribution_all = \
        normalized_superimposed_distribution_to_parent_area(par_inc_super_offspring_distribution_all,
                                                            parent_area_factor)
    return par_area_super_offspring_distribution_all


# AREA FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def area_scale_factor_entire(entire_superimposed_distribution):
    parent_area = entire_superimposed_distribution[0][5][1]
    superimposed_distribution_area = area_under_one_distribution(entire_superimposed_distribution)
    return superimposed_distribution_area / parent_area


def area_under_one_distribution(one_distribution):
    increment = one_distribution[0][2][1]
    return increment * (sum(row[1] for row in one_distribution))


def area_under_distributions(distributions):
    area = 0
    for distribution in distributions:
        area += area_under_one_distribution(distribution)
    return area


# CONVERSION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def z_score_to_index(z_score, number_of_steps, z_score_range):
    z_to_index_conversion = number_of_steps / z_score_range
    z_to_travel = z_score + (z_score_range / 2)
    return int((z_to_travel * z_to_index_conversion) + 0.5)


def percentile_to_value(percentile, distribution, distribution_sd=None):
    if distribution[0][6][0] == 'sd':  # if the distribution (parent) already has a given SD, use it
        return (distribution[0][6][1] * st.norm.ppf(percentile)) + distribution[0][5][1]
    else:  # if not, and we're not passing in a pre-calculated SD, then go ahead and calculate it
        if distribution_sd is not None:
            standard_dev = distribution_sd
        else:
            standard_dev = st_dev_of_distribution(distribution)  # calculate it based on the distribution
        return (standard_dev * st.norm.ppf(percentile)) + distribution[len(distribution) // 2][0]  # sd * z + mean


def z_score_to_value(sd, distribution, distribution_sd=None):  # should really be called z-score to value, but okay
    if distribution[0][6][0] == 'sd':  # if the distribution (parent) already has a given SD, use it
        return (distribution[0][6][1] * sd) + distribution[0][5][1]
    else:  # if not, and we're not passing in a pre-calculated SD, then go ahead and calculate it
        if distribution_sd is not None:
            standard_dev = distribution_sd
        else:
            standard_dev = st_dev_of_distribution(distribution)  # calculate it based on the distribution
        return (standard_dev * sd) + distribution[len(distribution) // 2][0]


def st_dev_of_distribution(distribution):
    mid_index = (len(distribution) - 1) // 2
    mean = distribution[mid_index][0]

    weights = [value[1] for value in distribution]
    x = [value[0] for value in distribution]

    sum_of_sq = 0
    for i in range(len(weights)):
        sum_of_sq += weights[i] * ((x[i] - mean) ** 2)

    n = sum(weights)
    st_dev = (sum_of_sq / n) ** 0.5

    return st_dev


# PROPORTION GENERAL FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_ks(parent_distribution, r, r_s, above_k_p=None, below_k_p=None, above_k_o=None, below_k_o=None,
               offspring_distribution=None, offspring_sd=None, assume_same_sd=True, conv_type='percentile'):
    k_list = [above_k_p, below_k_p, above_k_o, below_k_o]
    if assume_same_sd is False:  # if the parent and offspring generations are not assumed to
        # have same SD
        if offspring_distribution is None:  # if there is no provided off. dist. then create it
            offspring_distribution = \
                final_superimposed_distribution_all_area_adj(parent_distribution, r, r_s)
        if offspring_sd is None:  # if there is no provided
            standard_dev = st_dev_of_distribution(offspring_distribution)
        else:
            standard_dev = offspring_sd

    for i in range(len(k_list)):  # Convert the k_list to value
        if assume_same_sd is True:  # Assume offspring sd = parent sd
            if k_list[i] is not None:
                if conv_type == 'percentile':
                    k_list[i] = percentile_to_value(k_list[i], parent_distribution)
                elif conv_type == 'z_score':
                    k_list[i] = z_score_to_value(k_list[i], parent_distribution)
        elif assume_same_sd is False:
            if k_list[i] is not None:
                if i <= 1:  # for parent k, just use the sd listed in the parent distribution for the conversion
                    if conv_type == 'percentile':
                        k_list[i] = percentile_to_value(k_list[i], parent_distribution)
                    elif conv_type == 'z_score':
                        k_list[i] = z_score_to_value(k_list[i], parent_distribution)
                elif i >= 2:  # offspring k conversions
                    if conv_type == 'percentile':
                        # noinspection PyUnboundLocalVariable
                        k_list[i] = percentile_to_value(k_list[i], offspring_distribution, standard_dev)
                    elif conv_type == 'sd':
                        k_list[i] = z_score_to_value(k_list[i], offspring_distribution, standard_dev)
    return k_list


def select_over_all(parent_distribution, r, r_s, above_k_p=None, below_k_p=None, above_k_o=None, below_k_o=None,
                    area_all_distributions=None):
    select_distributions = offspring_distributions(parent_distribution, r, r_s, above_k_p, below_k_p, above_k_o,
                                                   below_k_o)
    area_select_distributions = area_under_distributions(select_distributions)
    return area_select_distributions / area_all_distributions


# PROPORTION VALUE FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def proportion_attributable_value(parent_distribution, r, r_s, above_k_p=None, below_k_p=None, above_k_o=None,
                                  below_k_o=None, area_all_distributions=None):
    if area_all_distributions is None:
        all_distributions = offspring_distributions(parent_distribution, r, r_s, above_k_v_o=above_k_o,
                                                    below_k_v_o=below_k_o)
        area_all_distributions = area_under_distributions(all_distributions)
    return select_over_all(parent_distribution, r, r_s, above_k_p, below_k_p, above_k_o, below_k_o,
                           area_all_distributions)


def proportion_destined_value(parent_distribution, r, r_s, above_k_p=None, below_k_p=None, above_k_o=None,
                              below_k_o=None, area_all_distributions=None):
    if area_all_distributions is None:
        all_distributions = offspring_distributions(parent_distribution, r, r_s, above_k_v_p=above_k_p,
                                                    below_k_v_p=below_k_p)
        area_all_distributions = area_under_distributions(all_distributions)
    return select_over_all(parent_distribution, r, r_s, above_k_p, below_k_p, above_k_o, below_k_o,
                           area_all_distributions)


# MISC. k PROPORTION DESTINED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def proportion_attributable_z_score(parent_distribution, r, r_s, above_k_p=None, below_k_p=None,
                                    above_k_o=None, below_k_o=None, area_all_distributions=None,
                                    offspring_distribution=None, offspring_sd=None, assume_same_sd=False):
    k_list = convert_ks(parent_distribution=parent_distribution, r=r, r_s=r_s, above_k_p=above_k_p,
                        below_k_p=below_k_p, above_k_o=above_k_o, below_k_o=below_k_o,
                        offspring_distribution=offspring_distribution, offspring_sd=offspring_sd,
                        assume_same_sd=assume_same_sd, conv_type='sd')

    return proportion_attributable_value(parent_distribution, r, r_s, k_list[0], k_list[1], k_list[2], k_list[3],
                                         area_all_distributions)


def proportion_attributable_percentile(parent_distribution, r, r_s, above_k_p=None, below_k_p=None,
                                       above_k_o=None, below_k_o=None, area_all_distributions=None,
                                       offspring_distribution=None, offspring_sd=None, assume_same_sd=False):
    k_list = convert_ks(parent_distribution=parent_distribution, r=r, r_s=r_s, above_k_p=above_k_p,
                        below_k_p=below_k_p, above_k_o=above_k_o, below_k_o=below_k_o,
                        offspring_distribution=offspring_distribution, offspring_sd=offspring_sd,
                        assume_same_sd=assume_same_sd, conv_type='percentile')

    return proportion_attributable_value(parent_distribution, r, r_s, k_list[0], k_list[1], k_list[2], k_list[3],
                                         area_all_distributions)


def proportion_destined_z_score(parent_distribution, r, r_s, above_k_p=None, below_k_p=None,
                                above_k_o=None, below_k_o=None, area_all_distributions=None,
                                offspring_distribution=None, offspring_sd=None, assume_same_sd=False):
    k_list = convert_ks(parent_distribution=parent_distribution, r=r, r_s=r_s, above_k_p=above_k_p,
                        below_k_p=below_k_p, above_k_o=above_k_o, below_k_o=below_k_o,
                        offspring_distribution=offspring_distribution, offspring_sd=offspring_sd,
                        assume_same_sd=assume_same_sd, conv_type='sd')

    return proportion_destined_value(parent_distribution, r, r_s, k_list[0], k_list[1], k_list[2], k_list[3],
                                     area_all_distributions)


def proportion_destined_percentile(parent_distribution, r, r_s, above_k_p=None, below_k_p=None, above_k_o=None,
                                   below_k_o=None, area_all_distributions=None, offspring_distribution=None,
                                   offspring_sd=None, assume_same_sd=False):
    k_list = convert_ks(parent_distribution=parent_distribution, r=r, r_s=r_s, above_k_p=above_k_p,
                        below_k_p=below_k_p, above_k_o=above_k_o, below_k_o=below_k_o,
                        offspring_distribution=offspring_distribution, offspring_sd=offspring_sd,
                        assume_same_sd=assume_same_sd, conv_type='percentile')

    return proportion_destined_value(parent_distribution, r, r_s, k_list[0], k_list[1], k_list[2], k_list[3],
                                     area_all_distributions)


# STEP PROPORTION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def step_proportion_attributable_percentile(parent_distribution, r, r_s, percentile_step,
                                            offspring_distribution=None, assume_same_sd=False):
    # offspring zones are in the first column of every row and the percent of offspring attributable to each parent
    # zone is in the second column of every row
    stepwise_percentile_list = list()
    above_k_o = 1 - percentile_step
    below_k_o = 1

    # Start Mar 13
    if offspring_distribution is None:
        offspring_distribution = final_superimposed_distribution_all_area_adj(parent_distribution, r, r_s)
    offspring_sd = st_dev_of_distribution(offspring_distribution)
    # End Mar 13

    while below_k_o > 0.5:
        step_list_offspring = [[above_k_o, below_k_o]]
        above_k_p = 1 - percentile_step
        below_k_p = 1
        step_list_offspring_parents = list()

        # Start. Mar 13
        above_k_o_v = percentile_to_value(above_k_o, offspring_distribution, offspring_sd)
        below_k_o_v = percentile_to_value(below_k_o, offspring_distribution, offspring_sd)
        # End. Was fixed.

        all_distributions = offspring_distributions(parent_distribution, r, r_s,
                                                    above_k_v_o=above_k_o_v, below_k_v_o=below_k_o_v)
        area_all_distributions = area_under_distributions(all_distributions)

        while below_k_p > 0.001:
            step_list_parent = [[above_k_p, below_k_p], proportion_attributable_percentile(
                parent_distribution, r, r_s, above_k_p, below_k_p, above_k_o, below_k_o,
                area_all_distributions, offspring_distribution, offspring_sd, assume_same_sd=assume_same_sd)]

            step_list_offspring_parents.append(step_list_parent)
            above_k_p = round(above_k_p - percentile_step, ROUND_NUMBER)
            below_k_p = round(below_k_p - percentile_step, ROUND_NUMBER)

        step_list_offspring.append(step_list_offspring_parents)
        stepwise_percentile_list.append(step_list_offspring)
        above_k_o = round(above_k_o - percentile_step, ROUND_NUMBER)
        below_k_o = round(below_k_o - percentile_step, ROUND_NUMBER)
    return stepwise_percentile_list


def step_proportion_destined_percentile(parent_distribution, r, r_s, percentile_step,
                                        offspring_distribution=None, assume_same_sd=False):
    # parent zones are in the first column of every row and the percent of the parent zone's offspring that are destined
    # to each offspring zone in the second column of every row
    stepwise_percentile_list = list()
    above_k_p = 1 - percentile_step
    below_k_p = 1

    # Start Mar 13
    if offspring_distribution is None:
        offspring_distribution = final_superimposed_distribution_all_area_adj(parent_distribution, r, r_s)
    offspring_sd = st_dev_of_distribution(offspring_distribution)
    # End Mar 13

    while below_k_p > 0.5:
        step_list_parent = [[above_k_p, below_k_p]]
        above_k_o = 1 - percentile_step
        below_k_o = 1
        step_list_parents_offspring = list()

        # SD approved
        above_k_p_v = percentile_to_value(above_k_p, parent_distribution)
        below_k_p_v = percentile_to_value(below_k_p, parent_distribution)
        # SD approved

        all_distributions = offspring_distributions(parent_distribution, r, r_s, above_k_v_p=above_k_p_v,
                                                    below_k_v_p=below_k_p_v)
        area_all_distributions = area_under_distributions(all_distributions)

        while below_k_o > 0.001:
            step_list_offspring = [[above_k_o, below_k_o], proportion_destined_percentile(
                parent_distribution, r, r_s, above_k_p, below_k_p, above_k_o, below_k_o,
                area_all_distributions, offspring_distribution, offspring_sd, assume_same_sd=assume_same_sd)]

            step_list_parents_offspring.append(step_list_offspring)
            above_k_o = round(above_k_o - percentile_step, ROUND_NUMBER)
            below_k_o = round(below_k_o - percentile_step, ROUND_NUMBER)

        step_list_parent.append(step_list_parents_offspring)
        stepwise_percentile_list.append(step_list_parent)
        above_k_p = round(above_k_p - percentile_step, ROUND_NUMBER)
        below_k_p = round(below_k_p - percentile_step, ROUND_NUMBER)
    return stepwise_percentile_list


def step_tree_question_z_score(parent_distribution, r, r_s, z_score_increment, z_score_bound,
                               offspring_distribution=None, assume_same_sd=False):
    z_score = - z_score_bound / 2  # initialize the z_score at the lower end of the bound
    proportion_list = list()

    if offspring_distribution is None:
        offspring_distribution = final_superimposed_distribution_all_area_adj(parent_distribution, r, r_s)
    offspring_sd = st_dev_of_distribution(offspring_distribution)

    while z_score <= z_score_bound / 2:
        proportion = proportion_attributable_z_score(parent_distribution=parent_distribution, r=r, r_s=r_s,
                                                     below_k_p=z_score, above_k_o=z_score,
                                                     offspring_distribution=offspring_distribution,
                                                     offspring_sd=offspring_sd, assume_same_sd=assume_same_sd)
        proportion_list.append(proportion)
        z_score += z_score_increment
    return proportion_list


# REPRODUCING FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def final_super_to_parent(final_super_distribution):
    super_max_index = len(final_super_distribution) - 1
    super_parent_max_index = final_super_distribution[0][6][1]

    final = list()

    # If it's bigger than the parent, make it only as big as the bound
    if super_max_index > super_parent_max_index:
        super_start_index = (super_max_index - super_parent_max_index) // 2
        super_end_index = super_start_index + super_parent_max_index + 1

    # If's it's equal to or smaller than the parent, make it as big as it is already
    else:
        super_start_index = 0
        super_end_index = len(final_super_distribution)

    for row_num in range(super_start_index, super_end_index):
        final_row = list()
        for column_num in range(2):
            final_row.append(final_super_distribution[row_num][column_num])
        final.append(final_row)

    mid_index = (len(final) - 1) // 2

    increment = final_super_distribution[0][2][1]
    number = len(final) - 1
    bound = final[-1][0] - final[0][0]
    mean = final[mid_index][0]
    st_dev = st_dev_of_distribution(final_super_distribution)

    final[0] += [['increment', increment], ['number', number], ['bound', bound], ['mean', mean], ['sd', st_dev]]

    return final


# PLOT FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_distributions(distributions):
    all_x = list()
    all_y = list()
    for distribution in distributions:
        x = list()
        y = list()
        for row in distribution:
            x.append(row[0])
            y.append(row[1])
        all_x.append(x)
        all_y.append(y)
    for dist_num in range(len(all_x)):
        plt.plot(all_x[dist_num], all_y[dist_num])


def plot_distribution(distribution, label=None):
    x = [row[0] for row in distribution]
    y = [row[1] for row in distribution]
    if label is not None:
        plt.plot(x, y, label=label)
    else:
        plt.plot(x, y, label=label)


def plot_generations_sd(generations):
    sd_list = list()
    for generation in generations:
        sd_list.append(st_dev_of_distribution(generation))

    x_label_list = list(range(len(generations)))
    x_generation_labels = [str(value + 1) for value in x_label_list]
    plt.xlabel('generation')
    plt.ylabel('standard deviation')
    plt.xticks(x_label_list, x_generation_labels)
    plt.plot(sd_list, '-o')


def bar_graph_step(step_list):
    num_groups = len(step_list[0][1])
    percent_group_values = list()
    for i in range(len(step_list[0][1])):
        values_list = [row[1][i][1] for row in step_list]
        percent_group_values.append(values_list)
    one_or_zero = num_groups % 2
    for num in range(len(percent_group_values[0]) - one_or_zero - 1, -1, -1):
        extra_values = list()
        for row in percent_group_values:
            extra_values.append(row[num])
        extra_values.reverse()
        for i in range(len(percent_group_values)):
            percent_group_values[i].append(extra_values[i])

    plot_mobility(percent_group_values)


def plot_mobility(data):
    pal = ['xkcd:light navy blue', 'xkcd:windows blue', 'xkcd:turquoise blue', 'xkcd:carolina blue', 'xkcd:light blue']

    if len(data) == 5:
        step_labels = ['Parent in\nBottom Quintile', 'Second Quintile', 'Third Quintile', 'Fourth Quintile',
                       'Top Quintile']
    else:
        step_labels = list(range(1, len(data) + 1))

    plt.figure()
    plt.xlabel('Parent\'s Quintile')
    plt.ylabel('Likelihood of Offspring in each Quintile')
    plt.ylim(0, 1)
    values_sum_list = [0] * len(data)
    for j in range(len(data)):
        if len(data) <= 5:
            plt.bar(step_labels, data[j], bottom=values_sum_list, color=pal[j])
        else:
            plt.bar(step_labels, data[j], bottom=values_sum_list)

        for a, b, c in zip(step_labels, values_sum_list, data[j]):
            num = (b + c / 2) - 0.02
            plt.text(a, num, ' ' + "{:0.0%}".format(c), va='bottom', ha='center', color='w', size=15)

        for i in range(len(values_sum_list)):
            values_sum_list[i] += data[j][i]

    legend = ['Offspring in\nBottom Quintile', 'Second Quintile', 'Third Quintile', 'Fourth Quintile', 'Top Quintile']
    plt.legend(legend, bbox_to_anchor=(1, 1), loc="upper left", fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()


# NOT USED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def z_erf(num):
#     return erf(num / (2 ** 0.5))