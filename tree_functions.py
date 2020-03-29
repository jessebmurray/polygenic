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
percent_step = 0.2

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

    # set the mean of the individual distribution
    shift = r * (parent_distribution[index_num][0] - parent_mean)
    offspring_mean = parent_mean + shift

    # set the st dev of the individual distribution
    parent_sd = parent_distribution[0][6][1]
    offspring_sd = r_s * parent_sd

    # set the scale factor, i.e., f(x_p)
    scale_factor = parent_distribution[index_num][1]

    # set the number of iterations for the individual distribution to be equal to that of the parent generation
    number = parent_distribution[0][3][1]

    # Here we range across the same bound as the parent distribution, even if that means
    # traversing 30 standard deviations, or 2. This reduces accuracy for r_s, especially when r_s << 1. Better
    # to traverse the same number of sds as the parent generation. However, the increase in computational complexity
    # is crazy because the normal dist function scales the increment too. So
    z_score_range = parent_distribution[0][4][1] / offspring_sd
    # Here we divide the parent bound by its standard deviation to get the parent z_score range, which we would use,
    # except it increases the computational complexity by a lot. We can deal with a less than an ideal response to
    # varying r_s, which we don't actually vary that much
    # z_score_range = parent_distribution[0][4][1] / parent_sd  # gives parent z_score range

    offspring_distribution = normal_distribution(number, z_score_range, offspring_mean, offspring_sd, above_k_v,
                                                 below_k_v)

    # scale all the y values by the scale factor
    for row in offspring_distribution:
        row[1] *= scale_factor

    # add the parent mean for bookkeeping
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

# def normalized_superimposed_distribution_to_parent_increment(superimposed_distribution):
#     parent_increment = superimposed_distribution[0][3][1]
#     parent_mean = superimposed_distribution[0][4][1]
#     smallest_x = superimposed_distribution[0][0]
#     n = int(abs(smallest_x - parent_mean) / parent_increment)
#     par_inc_norm_superimposed_distribution = list()
#     for num in range((2 * n) + 1):
#         x_value_prev = round((num - n - 1) * parent_increment, ROUND_NUMBER)
#         x_value = round((num - n) * parent_increment, ROUND_NUMBER)
#         par_inc_norm_superimposed_distribution.append([x_value, 0])
#         for row in superimposed_distribution:
#             if x_value_prev < row[0] <= x_value:
#                 par_inc_norm_superimposed_distribution[num][1] += row[1]
#             # ideally we'd like to stop this loop once row[0] is greater than the x_value
#     par_inc_norm_superimposed_distribution[0] += [['increment', parent_increment]]
#     par_inc_norm_superimposed_distribution[0] += superimposed_distribution[0][3:]
#     return par_inc_norm_superimposed_distribution


def normalized_superimposed_distribution_to_parent_increment(pre_dist, gen_num=0):
    norm_dist = list()

    par_inc = pre_dist[0][3][1]
    lowest_pre_x = pre_dist[0][0]
    # Initialize the increment location
    loc = int(lowest_pre_x / par_inc) * par_inc
    if gen_num == 1:
        loc = round(loc - par_inc, ROUND_NUMBER)

    pre_index = 0
    norm_index = 0
    # Loop through the distribution and bin to the increment location
    norm_dist.append([loc, 0])
    while pre_index < len(pre_dist):
        if gen_num == 0:  # shift to the right
            if pre_dist[pre_index][0] <= loc:
                norm_dist[norm_index][1] += pre_dist[pre_index][1]
                pre_index += 1
            else:
                norm_index += 1
                loc = round(loc + par_inc, ROUND_NUMBER)
                norm_dist.append([loc, 0])
        elif gen_num == 1:  # shift to the left
            if loc <= pre_dist[pre_index][0] <= round(loc + par_inc, ROUND_NUMBER):
                norm_dist[norm_index][1] += pre_dist[pre_index][1]
                pre_index += 1
            else:
                norm_index += 1
                loc = round(loc + par_inc, ROUND_NUMBER)
                norm_dist.append([loc, 0])
    norm_dist[0] += [['increment', par_inc]]
    norm_dist[0] += pre_dist[0][3:]

    return norm_dist


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


def final_superimposed_distribution_all_area_adj(parent_distribution, r, r_s):
    par_inc_super_offspring_distribution_all = \
        final_superimposed_distribution_all_not_area_adj(parent_distribution, r, r_s)
    parent_area_factor = area_scale_factor_entire(par_inc_super_offspring_distribution_all)
    par_area_super_offspring_distribution_all = \
        normalized_superimposed_distribution_to_parent_area(par_inc_super_offspring_distribution_all,
                                                            parent_area_factor)
    return par_area_super_offspring_distribution_all


# AREA FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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


def z_score_to_value(sd, distribution, distribution_sd=None):
    if distribution[0][6][0] == 'sd':  # if the distribution (parent) already has a given SD, use it
        return (distribution[0][6][1] * sd) + distribution[0][5][1]
    else:  # if not, and we're not passing in a pre-calculated SD, then go ahead and calculate it
        if distribution_sd is not None:
            standard_dev = distribution_sd
        else:
            standard_dev = st_dev_of_distribution(distribution)  # calculate it based on the distribution
        return (standard_dev * sd) + distribution[len(distribution) // 2][0]


def st_dev_of_distribution(distribution):
    mean = true_mean(distribution)

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
        if offspring_sd is None:  # if there is no provided offspring sd
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
                    elif conv_type == 'z_score':
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
                        assume_same_sd=assume_same_sd, conv_type='z_score')

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
                        assume_same_sd=assume_same_sd, conv_type='z_score')

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


def percentile_to_value(percentile, distribution, distribution_sd=None):
    if distribution[0][6][0] == 'sd':  # if the distribution (parent) already has a given SD, use it
        return (distribution[0][6][1] * st.norm.ppf(percentile)) + distribution[0][5][1]  # sd * z + mean
    else:  # if not, and we're not passing in a pre-calculated SD, then go ahead and calculate it
        if distribution_sd is not None:
            standard_dev = distribution_sd
        else:
            standard_dev = st_dev_of_distribution(distribution)  # calculate it based on the distribution
        return (standard_dev * st.norm.ppf(percentile)) + true_mean(distribution)  # sd * z + mean
        # return (standard_dev * st.norm.ppf(percentile)) + distribution[len(distribution) // 2][0]  # sd * z + mean


def step_proportion_destined_percentile(parent_distribution, r, r_s, percentile_step,
                                        offspring_distribution=None, assume_same_sd=False):
    # Very slow
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


def final_superimposed_distribution(parent_distribution, r, r_s, above_k_v_p=None, below_k_v_p=None,
                                    above_k_v_o=None, below_k_v_o=None, parent_area_factor=None):
    offspring_distributions_ = offspring_distributions(parent_distribution, r, r_s, above_k_v_p, below_k_v_p,
                                                       above_k_v_o, below_k_v_o)
    super_offspring_distribution = superimposed_offspring_distribution(offspring_distributions_)

    # Normalize to the parent increment
    par_inc_super_offspring_distribution = \
        normalized_superimposed_distribution_to_parent_increment(super_offspring_distribution)

    # We need the entire offspring distribution to get the area scale factor reliably
    if parent_area_factor is None:
        par_inc_super_offspring_distribution_all = \
            final_superimposed_distribution_all_not_area_adj(parent_distribution, r, r_s)
        parent_area_factor = area_scale_factor_entire(par_inc_super_offspring_distribution_all)
    else:
        parent_area_factor = parent_area_factor

    # Just divides all the 'y's in the distribution by the area scale factor
    par_area_super_offspring_distribution = \
        normalized_superimposed_distribution_to_parent_area(par_inc_super_offspring_distribution, parent_area_factor)

    return par_area_super_offspring_distribution


def area_scale_factor_entire(entire_superimposed_distribution):
    parent_area = entire_superimposed_distribution[0][5][1]
    superimposed_distribution_area = area_under_one_distribution(entire_superimposed_distribution)
    return superimposed_distribution_area / parent_area


# not used
def gen_conv(percentile, gen_sd):
    z_score = st.norm.ppf(percentile)
    return gen_sd * z_score


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


# TRUE MEAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def true_mean(dist):
    # gives the probability weighted mean/also median for any
    max_ = 0
    max_index = 0
    for i in range(len(dist)):
        if dist[i][1] > max_:
            max_ = dist[i][1]
            max_index = i
    return dist[max_index][0]


def true_mean_index(dist):
    # gives the index of the probability weighted mean/also median for any
    max_ = 0
    max_index = 0
    for i in range(len(dist)):
        if dist[i][1] > max_:
            max_ = dist[i][1]
            max_index = i
    return max_index


# REPRODUCING FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def final_super_to_parent(final_super_distribution, population_mean=None):
    # Very fast
    super_max_index = len(final_super_distribution) - 1
    super_parent_max_index = final_super_distribution[0][6][1]  # (parent number gives the max index)

    final = list()

    # If it's bigger than the parent, make it only as big as the bound
    if super_max_index > super_parent_max_index:
        # super_start_index = (super_max_index - super_parent_max_index) // 2
        # super_end_index = super_start_index + super_parent_max_index + 1

        # Mar 27
        mean_index = true_mean_index(final_super_distribution)
        start_index = mean_index - (super_parent_max_index // 2)
        end_index = start_index + super_parent_max_index + 1
        # Mar 27

    # If's it's equal to or smaller than the parent, make it as big as it is already
    else:
        # super_start_index = 0
        # super_end_index = len(final_super_distribution)
        start_index = 0
        end_index = len(final_super_distribution)

    # Simply copy it with the new lengths

    for row_num in range(start_index, end_index):
        final_row = list()
        for column_num in range(2):
            final_row.append(final_super_distribution[row_num][column_num])
        final.append(final_row)

    # for row_num in range(super_start_index, super_end_index):
    #     final_row = list()
    #     for column_num in range(2):
    #         final_row.append(final_super_distribution[row_num][column_num])
    #     final.append(final_row)

    mid_index = (len(final) - 1) // 2

    increment = final_super_distribution[0][2][1]
    number = len(final) - 1
    bound = final[-1][0] - final[0][0]
    if population_mean is None:
        mean = final[mid_index][0]
        mean_name = 'mean'
    else:
        mean = population_mean
        mean_name = 'population mean'
        final[1] += ['parent_mean', true_mean(final_super_distribution)]
    st_dev = st_dev_of_distribution(final_super_distribution)

    final[0] += [['increment', increment], ['number', number], ['bound', bound], [mean_name, mean], ['sd', st_dev]]

    return final


# PLOT FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plt_dists(distributions, linestyle=None):
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
        plt.plot(all_x[dist_num], all_y[dist_num], linewidth=0.8, linestyle=linestyle)


def plt_dist(distribution, label=None, color=None, linestyle=None, alpha=None, lw=None):
    x = [row[0] for row in distribution]
    y = [row[1] for row in distribution]
    plt.xlabel('Phenotype SDS')
    plt.ylabel('Proportion')
    plt.plot(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha, lw=lw)


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


def plot_mobility(data, pal=None):
    plt.figure(figsize=(13, 8))
    if pal is None:
        # pal = ['xkcd:light navy blue', 'xkcd:windows blue', '#1CA3DE', 'xkcd:carolina blue', 'xkcd:light blue']
        pal = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5']
        pal.reverse()

    if len(data) == 5:
        step_labels = ['Parent in\nBottom Quintile', 'Second Quintile', 'Third Quintile', 'Fourth Quintile',
                       'Top Quintile']
    else:
        step_labels = list(range(1, len(data) + 1))

    plt.xlabel('Parent\'s Quintile', fontsize=15)
    plt.ylabel('Likelihood of Offspring in each Quintile', fontsize=15)
    plt.ylim(0, 1)
    values_sum_list = [1] * len(data)
    for j in range(len(data) - 1, -1, -1):
        if len(data) <= 5:
            plt.bar(step_labels, [- value for value in data[j]], bottom=values_sum_list, color=pal[j])
        else:
            plt.bar(step_labels, [- value for value in data[j]], bottom=values_sum_list)

        for a, b, c in zip(step_labels, values_sum_list, [- value for value in data[j]]):
            num = (b + c / 2) - 0.018
            # plt.text(a, num, ' ' + "{:0.0%}".format(- c), va='bottom', ha='center', color='k', size=15, alpha=0.7)
            if j >= 2:
                plt.text(a, num, ' ' + "{:0.0%}".format(- c), va='bottom', ha='center', color='k', size=15, alpha=0.8)
            else:
                plt.text(a, num, ' ' + "{:0.0%}".format(- c), va='bottom', ha='center', color='w', size=15, alpha=0.8)

        for i in range(len(values_sum_list)):
            values_sum_list[i] -= data[j][i]

    legend = ['Offspring in\nTop Quintile', 'Fourth Quintile', 'Third Quintile', 'Second Quintile', 'Bottom Quintile']
    plt.legend(legend, bbox_to_anchor=(1, 1), loc="upper left", fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))


# MULTI GENERATIONAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def initialize_dscts(gen_0_par, r, r_s, percentile_step=0.2):
    below_k_p = 1
    above_k_p = 1 - percentile_step

    par_inc_super_offspring_distribution_all = \
        final_superimposed_distribution_all_not_area_adj(gen_0_par, r, r_s)
    parent_area_factor = area_scale_factor_entire(par_inc_super_offspring_distribution_all)

    gen_1s = list()
    while below_k_p > 0.5:
        above_k_p_v = percentile_to_value(above_k_p, gen_0_par)
        below_k_p_v = percentile_to_value(below_k_p, gen_0_par)

        gen_1s.append(final_superimposed_distribution(gen_0_par, r, r_s,
                                                      above_k_v_p=above_k_p_v,
                                                      below_k_v_p=below_k_p_v,
                                                      parent_area_factor=parent_area_factor))
        # st.norm.ppf(gen_cov(above_k_p)) (the old way)

        above_k_p = round(above_k_p - percentile_step, ROUND_NUMBER)
        below_k_p = round(below_k_p - percentile_step, ROUND_NUMBER)

    return gen_1s


def initialize_dsct(gen_0_par, r, r_s, above_perc, below_perc):
    par_inc_super_offspring_distribution_all = \
        final_superimposed_distribution_all_not_area_adj(gen_0_par, r, r_s)
    parent_area_factor = area_scale_factor_entire(par_inc_super_offspring_distribution_all)

    above_k_p_v = percentile_to_value(above_perc, gen_0_par)
    below_k_p_v = percentile_to_value(below_perc, gen_0_par)

    gen_1_dsct = final_superimposed_distribution(gen_0_par, r, r_s, above_k_v_p=above_k_p_v, below_k_v_p=below_k_p_v,
                                                 parent_area_factor=parent_area_factor)

    return gen_1_dsct


def dsct_destined(gen_par, gen_dsct_par, perc=0.2):
    total_area = area_under_one_distribution(gen_dsct_par)
    above_k_o = 1 - perc
    below_k_o = 1

    areas = list()

    latest_below_index = len(gen_dsct_par) - 1

    while below_k_o > 0.001:

        above_k_v_o = percentile_to_value(above_k_o, gen_par)

        # set the indices
        above_index = nearest_index(above_k_v_o, gen_dsct_par)
        below_index = latest_below_index

        area = area_btw(gen_dsct_par, above_index, below_index)
        area /= total_area
        areas.append([[above_k_o, below_k_o], area])

        above_k_o = round(above_k_o - perc, ROUND_NUMBER)
        below_k_o = round(below_k_o - perc, ROUND_NUMBER)

        # update the latest below index to be the same as the previous above index
        # this way we don't ever ever have to calculate below values, cause they're
        # the same as the previous step's above value
        latest_below_index = above_index - 1

    return areas


def nearest_index(value, dist):
    value_index = 0
    for i in range(len(dist)):
        if dist[i][0] <= value:
            value_index = i
    return value_index


def area_btw(dist_par, above_index, below_index):
    # this function assumes a constant increment for a parent distribution
    increment = dist_par[0][2][1]
    area = 0
    for i in range(above_index, below_index + 1):
        area += dist_par[i][1]
    area *= increment
    return area


def multi_1st(gen_0_par, r, r_s, perc=0.2, population_mean=0):
    gen_1 = final_superimposed_distribution_all_area_adj(gen_0_par, r, r_s)
    gen_1_par = final_super_to_parent(gen_1)

    gen_1_dscts = initialize_dscts(gen_0_par, r, r_s, perc)
    gen_1_dscts_par = list()
    for row in gen_1_dscts:
        gen_1_dscts_par.append(final_super_to_parent(row, population_mean=population_mean))

    return gen_1_par, gen_1_dscts_par


def one_1st(gen_0_par, r, r_s, above_perc, below_perc, population_mean=0):
    gen_1 = final_superimposed_distribution_all_area_adj(gen_0_par, r, r_s)
    gen_1_par = final_super_to_parent(gen_1)

    gen_1_dsct = initialize_dsct(gen_0_par, r, r_s, above_perc, below_perc)
    gen_1_dsct_par = final_super_to_parent(gen_1_dsct, population_mean=population_mean)

    return gen_1_par, gen_1_dsct_par


def destined_dsct_n_gen(gen_1_par, gen_1_dsct_par, r, r_s, n_gen, perc, population_mean):
    destined_matrices = list()
    gen_dsct_pars = list()
    gen_pars = list()

    # Append generation 1 and the descendant
    gen_pars.append(gen_1_par)
    gen_dsct_pars.append(gen_1_dsct_par)

    # Initialize the destined matrix
    destined_matrix = dsct_destined(gen_1_par, gen_1_dsct_par, perc=perc)
    destined_matrices.append(destined_matrix)

    # Initialize the latest generation
    latest_gen_par = gen_1_par
    latest_gen_dsct_par = gen_1_dsct_par

    # Create the distributions and their matrices
    for i in range(2, n_gen + 1):
        # Calculate and append the new total parent generation
        gen_i = final_superimposed_distribution_all_area_adj(latest_gen_par, r, r_s)
        gen_i_par = final_super_to_parent(gen_i)
        gen_pars.append(gen_i_par)

        # Calculate and append the new descendant parent generation
        gen_i_dsct = final_superimposed_distribution_all_area_adj(latest_gen_dsct_par, r, r_s)
        gen_i_dsct_par = final_super_to_parent(gen_i_dsct, population_mean=population_mean)
        gen_dsct_pars.append(gen_i_dsct_par)

        # Create and append the new destined matrix
        destined_matrix = dsct_destined(gen_i_par, gen_i_dsct_par, perc=perc)
        destined_matrices.append(destined_matrix)

        # Update the latest generation
        latest_gen_par = gen_i_par
        latest_gen_dsct_par = gen_i_dsct_par

        # Print update
        print(destined_matrices, gen_dsct_pars, gen_pars)
        print('\n\n\n\n\n\n\n\n\n\n\n\n')

    return destined_matrices, gen_dsct_pars, gen_pars


def transpose_matrix(matrix):
    num_gens = len(matrix)
    num_dscts = len(matrix[0])

    transposed = list()
    for dsct_num in range(num_dscts - 1, -1, -1):
        sub_transposed = list()
        for gen_num in range(num_gens):
            sub_transposed.append(matrix[gen_num][dsct_num][1])
        transposed.append(sub_transposed)
    return transposed


def get_colors():
    file_colors = open('tree_colors.txt', 'r')
    colors_list = list()
    for line in file_colors:
        colors_list.append(line.strip())
    file_colors.close()
#     colors_list.reverse()
    return colors_list


def plot_multi_mobility(data, pal=None):
    n_dscts = len(data)  # 10
    n_gens = len(data[0])  # 5

    plt.figure(figsize=(13, 8))
    if pal is None:
        # pal = ['xkcd:light navy blue', 'xkcd:windows blue', '#1CA3DE', 'xkcd:carolina blue', 'xkcd:light blue']
        pal = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5']
        pal.reverse()
        if n_dscts == 10:
            pal = get_colors()

    step_labels = list(range(1, n_gens + 1))

    plt.xlabel('Generation', fontsize=15)
    if n_dscts == 10:
        group_name = 'Decile'
    elif n_dscts == 5:
        group_name = 'Quintile'
    else:
        group_name = 'Group'

    plt.ylabel('Likelihood of Descendants in each ' + group_name, fontsize=15)
    plt.ylim(0, 1)
    values_sum_list = [1] * n_gens

    for j in range(len(data) - 1, -1, -1):
        plt.bar(step_labels, [- value for value in data[j]], bottom=values_sum_list, color=pal[j])

        for a, b, c in zip(step_labels, values_sum_list, [- value for value in data[j]]):
            num = (b + c / 2) - 0.018
            # plt.text(a, num, ' ' + "{:0.0%}".format(- c), va='bottom', ha='center', color='k', size=15, alpha=0.7)

            horz_align = 'center'
            #             if abs(c) <= 0.01:
            #                 horz_align = 'left'
            #             elif abs(c) <= 0.03:
            #                 horz_align = 'right'
            if abs(c) >= 0.01:
                if 7 >= j >= 2:
                    plt.text(a, num, ' ' + "{:0.0%}".format(- c), va='bottom', ha=horz_align, color='k', size=15,
                             alpha=0.8)
                else:
                    plt.text(a, num, ' ' + "{:0.0%}".format(- c), va='bottom', ha=horz_align, color='w', size=15,
                             alpha=0.8)

        for i in range(len(values_sum_list)):
            values_sum_list[i] -= data[j][i]

    if n_dscts == 5:
        legend = ['Descendants in\nTop Quintile', 'Fourth Quintile', 'Third Quintile', 'Second Quintile',
                  'Bottom Quintile']
    elif n_dscts == 10:
        legend = ['Descendants in\nTop Decile', 'Ninth Decile', 'Eighth Decile', 'Seventh Decile',
                  'Sixth Decile', 'Fifth Decile', 'Fourth Decile', 'Third Decile', 'Second Decile', 'Bottom Decile']
    else:
        legend = ['None']

    plt.legend(legend, bbox_to_anchor=(1, 1), loc="upper left", fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(list(range(1, n_gens + 1)))
#     plt.xticks([1,2,3,4,5], ['Son/Daughter', 'Grand', 'Great', 'Great-great', 'Great-great-great'])


# NOT USED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def z_erf(num):
#     return erf(num / (2 ** 0.5))
