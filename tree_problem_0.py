import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
# from scipy.special import erf


# Global variables that are just used in main
number_of_iterations = 100
z_range = 8
r = 0.9
r_s = 0.9
mean_gen = 0
sd_gen = 1
k_val = -2
percent_step = 0.33


# Global variables that are used other than just in main (should really be a constant)
round_number = 6


def main():

    # DONE
    parent_distribution = normal_distribution(number_of_iterations, z_range, mean_gen, sd_gen)

    # print(proportion_destined_value(parent_distribution, r, r_s, above_k_p=value, below_k_o=value))
    # print(step_proportion_destined_percentile(parent_distribution, r, r_s, percent_step))
    #
    # # print(proportion_destined_value(parent_distribution, r, r_s, below_k_p=k_val, below_k_o=k_val))
    # print(step_proportion_attributable_percentile(parent_distribution, r, r_s, percent_step))

    # DONE
    offspring_distributions_ = offspring_distributions(parent_distribution, r, r_s)
    plot_distribution(parent_distribution)
    plot_distributions(offspring_distributions_)
    plot_distribution(final_superimposed_distribution_all_area_adj(parent_distribution, r, r_s))

    # plot_distribution(final_superimposed_distribution(parent_distribution, r, r_s, below_k_v_p=k_val))
    # plot_distribution(final_superimposed_distribution(parent_distribution, r, r_s, below_k_v_p=k_val,
    #                                                   above_k_v_o=k_val))

    # plt.show()

    # plt.show()
    # offspring_distribution = final_superimposed_distribution_all_area_adj(parent_distribution, r, r_s)
    # plot_distribution(parent_distribution)
    # plot_distribution(offspring_distribution)
    # plt.show()
    # print(parent_distribution)

    # print(proportion_attributable_value(parent_distribution, r, r_s, below_k_p=k_val, above_k_o=k_val))


# FUNDAMENTAL STRUCTURE FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def f_norm(x, mean, sd):
    return (1 / (sd * ((2 * np.pi) ** 0.5))) * np.exp(-1 * ((((x - mean) / sd) ** 2) / 2))


def normal_distribution(number_of_steps, z_score_range, mean=0, sd=1, above_k_value=None, below_k_value=None):
    # This creates a normal distribution with a certain number of steps. The motivation for using number of steps is
    # that it is related to the number of operations. It can make all y values 0, for x below k_value
    two_d_distribution = []
    bound = z_score_range * sd
    number = number_of_steps
    increment = bound / number
    if above_k_value is not None:
        above_n = ((0.5 * z_score_range * sd) + above_k_value - mean) / increment
    else:
        above_n = 0
    if below_k_value is not None:
        below_n = ((0.5 * z_score_range * sd) + below_k_value - mean) / increment
    else:
        below_n = 0
    x_start = mean - (bound / 2)
    x = round(x_start, round_number)
    for num in range(number + 1):
        x_y_pair = [0, 0]
        go_ahead = True
        if above_k_value is not None:
            if num < above_n:
                x_y_pair = [x, 0]
                go_ahead = False
        if below_k_value is not None:
            if num > below_n:
                x_y_pair = [x, 0]
                go_ahead = False
        if go_ahead:
            x_y_pair = [x, f_norm(x, mean, sd)]
        two_d_distribution.append(x_y_pair)
        x = round(x_start + (increment * (num + 1)), round_number)
    two_d_distribution[0] += [['increment', increment], ['number', number], ['bound', bound], ['mean', mean],
                              ['sd', sd]]
    return two_d_distribution


def one_offspring_distribution(par_distribution, index_num, reg_coefficient, sd_reg_coefficient, above_k_value=None,
                               below_k_value=None):
    # we need a function that takes a value from the parent distribution and multiplies every value in the offspring
    # distribution by that value, essentially scaling it by that value.
    # Also the x values in the offspring distribution need to be shifted by r * z_p
    parent_mean = par_distribution[0][5][1]
    shift = reg_coefficient * (par_distribution[index_num][0] - parent_mean)
    offspring_mean = parent_mean + shift
    parent_sd = par_distribution[0][6][1]
    offspring_sd = sd_reg_coefficient * parent_sd
    scale_factor = par_distribution[index_num][1]
    number = par_distribution[0][3][1]
    z_score_range = par_distribution[0][4][1] / offspring_sd
    offspring_distribution = normal_distribution(number, z_score_range, offspring_mean, offspring_sd, above_k_value,
                                                 below_k_value)
    for row in offspring_distribution:
        row[1] *= scale_factor
    offspring_distribution[0] += [['parent mean', parent_mean]]
    return offspring_distribution


def offspring_distributions(par_distribution, reg_coefficient, sd_reg_coefficient, above_k_v_p=None, below_k_v_p=None,
                            above_k_v_o=None, below_k_v_o=None):
    parent_increment = par_distribution[0][2][1]
    parent_bound = par_distribution[0][4][1]
    parent_mean = par_distribution[0][5][1]

    if above_k_v_p is not None:
        above_num = (above_k_v_p - parent_mean + (parent_bound * 0.5)) / parent_increment
    else:
        above_num = 0
    if below_k_v_p is not None:
        below_num = (below_k_v_p - parent_mean + (parent_bound * 0.5)) / parent_increment
    else:
        below_num = 0

    all_offspring_distributions = []
    for index in range(len(par_distribution)):
        go_ahead = True
        if above_k_v_p is not None:
            if index < above_num:
                go_ahead = False
        if below_k_v_p is not None:
            if index > below_num:
                go_ahead = False
        if go_ahead:
            all_offspring_distributions.append(one_offspring_distribution(par_distribution, index, reg_coefficient,
                                                                          sd_reg_coefficient, above_k_v_o, below_k_v_o))
    # add the parent area to the top of offspring distributions
    parent_area = area_under_one_distribution(par_distribution)
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
    # the below increment is wrong in a lot of cases
    increment = round(superimposed_distribution[1][0] - superimposed_distribution[0][0], round_number)

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
    par_inc_norm_superimposed_distribution = []
    for num in range((2 * n) + 1):
        x_value_prev = round((num - n - 1) * parent_increment, round_number)
        x_value = round((num - n) * parent_increment, round_number)
        par_inc_norm_superimposed_distribution.append([x_value, 0])
        for row in superimposed_distribution:
            if x_value_prev < row[0] <= x_value:
                par_inc_norm_superimposed_distribution[num][1] += row[1]
            # ideally we'd like to stop this loop to stop once row[0] is greater than the x_value
    par_inc_norm_superimposed_distribution[0] += [['increment', parent_increment]]
    par_inc_norm_superimposed_distribution[0] += superimposed_distribution[0][3:]
    return par_inc_norm_superimposed_distribution


def final_superimposed_distribution_all_not_area_adj(parent_distribution, reg_coefficient, sd_reg_coefficient):
    offspring_distributions_all = offspring_distributions(parent_distribution, reg_coefficient, sd_reg_coefficient)
    super_offspring_distribution_all = superimposed_offspring_distribution(offspring_distributions_all)
    par_inc_super_offspring_distribution_all = \
        normalized_superimposed_distribution_to_parent_increment(super_offspring_distribution_all)
    return par_inc_super_offspring_distribution_all


def normalized_superimposed_distribution_to_parent_area(superimposed_distribution, area_scale_factor):
    par_area_norm_superimposed_distribution = \
        [[row[0], row[1] / area_scale_factor] for row in superimposed_distribution]
    par_area_norm_superimposed_distribution[0] += superimposed_distribution[0][2:]
    return par_area_norm_superimposed_distribution


def final_superimposed_distribution(parent_distribution, reg_coefficient, sd_reg_coefficient, above_k_v_p=None,
                                    below_k_v_p=None, above_k_v_o=None, below_k_v_o=None):
    offspring_distributions_ = offspring_distributions(parent_distribution, reg_coefficient, sd_reg_coefficient,
                                                       above_k_v_p, below_k_v_p, above_k_v_o, below_k_v_o)
    super_offspring_distribution = superimposed_offspring_distribution(offspring_distributions_)
    par_inc_super_offspring_distribution = \
        normalized_superimposed_distribution_to_parent_increment(super_offspring_distribution)
    par_inc_super_offspring_distribution_all = \
        final_superimposed_distribution_all_not_area_adj(parent_distribution, reg_coefficient, sd_reg_coefficient)
    parent_area_factor = area_scale_factor_entire(par_inc_super_offspring_distribution_all)
    par_area_super_offspring_distribution = \
        normalized_superimposed_distribution_to_parent_area(par_inc_super_offspring_distribution, parent_area_factor)
    return par_area_super_offspring_distribution


def final_superimposed_distribution_all_area_adj(parent_distribution, reg_coefficient, sd_reg_coefficient):
    par_inc_super_offspring_distribution_all = \
        final_superimposed_distribution_all_not_area_adj(parent_distribution, reg_coefficient, sd_reg_coefficient)
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
def percentile_to_value(percentile, parent_distribution):
    return (parent_distribution[0][6][1] * st.norm.ppf(percentile)) + parent_distribution[0][5][1]


def sd_to_value(sd, parent_distribution):
    return (parent_distribution[0][6][1] * sd) + parent_distribution[0][5][1]


def z_score_to_index(z_score, number_of_steps, z_score_range):
    z_to_index_conversion = number_of_steps / z_score_range
    z_to_travel = z_score + (z_score_range / 2)
    return int((z_to_travel * z_to_index_conversion) + 0.5)


# PLOTTING FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_distributions(distributions):
    all_x = []
    all_y = []
    for distribution in distributions:
        x = []
        y = []
        for row in distribution:
            x.append(row[0])
            y.append(row[1])
        all_x.append(x)
        all_y.append(y)
    for dist_num in range(len(all_x)):
        plt.plot(all_x[dist_num], all_y[dist_num])
    plt.show()


def plot_distribution(distribution):
    x = [row[0] for row in distribution]
    y = [row[1] for row in distribution]
    plt.plot(x, y)


def plot_generations_sd(generations):
    sd_list = []
    for generation in generations:
        sd_list.append(st_dev_of_distribution(generation))
    plt.plot(sd_list, '-o')


# PROPORTION ATTRIBUTABLE FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def proportion_attributable_value(parent_distribution, r_mean, r_sd, above_k_p=None, below_k_p=None, above_k_o=None,
                                  below_k_o=None, area_all_distributions=None):
    if area_all_distributions is None:
        all_distributions = offspring_distributions(parent_distribution, r_mean, r_sd, above_k_v_o=above_k_o,
                                                    below_k_v_o=below_k_o)
        area_all_distributions = area_under_distributions(all_distributions)
    # select_distributions = offspring_distributions(parent_distribution, r_mean, r_sd, above_k_p, below_k_p, above_k_o,
    #                                                below_k_o)
    # area_select_distributions = area_under_distributions(select_distributions)
    # return area_select_distributions / area_all_distributions
    return select_over_all(parent_distribution, r_mean, r_sd, above_k_p, below_k_p, above_k_o, below_k_o,
                           area_all_distributions)


def proportion_attributable_standard_deviation(parent_distribution, r_mean, r_sd, above_k_p=None, below_k_p=None,
                                               above_k_o=None, below_k_o=None, area_all_distributions=None):
    k_list = [above_k_p, below_k_p, above_k_o, below_k_o]
    for i in range(len(k_list)):
        if k_list[i] is not None:
            k_list[i] = sd_to_value(k_list[i], parent_distribution)
    return proportion_attributable_value(parent_distribution, r_mean, r_sd, k_list[0], k_list[1], k_list[2], k_list[3],
                                         area_all_distributions)


def proportion_attributable_percentile(parent_distribution, r_mean, r_sd, above_k_p=None, below_k_p=None,
                                       above_k_o=None, below_k_o=None, area_all_distributions=None):
    k_list = [above_k_p, below_k_p, above_k_o, below_k_o]
    for i in range(len(k_list)):
        if k_list[i] is not None:
            k_list[i] = percentile_to_value(k_list[i], parent_distribution)
    return proportion_attributable_value(parent_distribution, r_mean, r_sd, k_list[0], k_list[1], k_list[2], k_list[3],
                                         area_all_distributions)


def step_proportion_attributable_percentile(parent_distribution, reg_coefficient, sd_reg_coefficient, percentile_step):
    # offspring zones are in the first column of every row and the percent of offspring attributable to each parent
    # zone is in the second column of every row
    stepwise_percentile_list = []
    above_k_o = 1 - percentile_step
    below_k_o = 1
    while below_k_o > 0.5:
        step_list_offspring = [[above_k_o, below_k_o]]
        above_k_p = 1 - percentile_step
        below_k_p = 1
        step_list_offspring_parents = []
        above_k_o_v = percentile_to_value(above_k_o, parent_distribution)
        below_k_o_v = percentile_to_value(below_k_o, parent_distribution)
        # different
        all_distributions = offspring_distributions(parent_distribution, reg_coefficient, sd_reg_coefficient,
                                                    above_k_v_o=above_k_o_v, below_k_v_o=below_k_o_v)
        area_all_distributions = area_under_distributions(all_distributions)
        while below_k_p > 0.001:
            # different
            step_list_parent = [[above_k_p, below_k_p], proportion_attributable_percentile(
                parent_distribution, reg_coefficient, sd_reg_coefficient, above_k_p, below_k_p, above_k_o, below_k_o,
                area_all_distributions)]

            step_list_offspring_parents.append(step_list_parent)
            above_k_p = round(above_k_p - percentile_step, round_number)
            below_k_p = round(below_k_p - percentile_step, round_number)
        step_list_offspring.append(step_list_offspring_parents)
        stepwise_percentile_list.append(step_list_offspring)
        above_k_o = round(above_k_o - percentile_step, round_number)
        below_k_o = round(below_k_o - percentile_step, round_number)
    return stepwise_percentile_list


def select_over_all(parent_distribution, r_mean, r_sd, above_k_p=None, below_k_p=None, above_k_o=None, below_k_o=None,
                    area_all_distributions=None):
    select_distributions = offspring_distributions(parent_distribution, r_mean, r_sd, above_k_p, below_k_p, above_k_o,
                                                   below_k_o)
    area_select_distributions = area_under_distributions(select_distributions)
    return area_select_distributions / area_all_distributions


def step_tree_question_z_score(parent_distribution, r_mean, r_sd, z_score_increment, z_score_bound):
    z_score = - z_score_bound / 2

    proportion_list = []
    while z_score <= z_score_bound / 2:
        proportion = proportion_attributable_standard_deviation(parent_distribution, r_mean, r_sd,
                                                                below_k_p=z_score, above_k_o=z_score)
        proportion_list.append(proportion)
        z_score += z_score_increment
    return proportion_list


# PROPORTION DESTINED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def proportion_destined_value(parent_distribution, r_mean, r_sd, above_k_p=None, below_k_p=None, above_k_o=None,
                              below_k_o=None, area_all_distributions=None):
    if area_all_distributions is None:
        all_distributions = offspring_distributions(parent_distribution, r_mean, r_sd, above_k_v_p=above_k_p,
                                                    below_k_v_p=below_k_p)
        area_all_distributions = area_under_distributions(all_distributions)
    return select_over_all(parent_distribution, r_mean, r_sd, above_k_p, below_k_p, above_k_o, below_k_o,
                           area_all_distributions)


def proportion_destined_percentile(parent_distribution, r_mean, r_sd, above_k_p=None, below_k_p=None,
                                   above_k_o=None, below_k_o=None, area_all_distributions=None):
    k_list = [above_k_p, below_k_p, above_k_o, below_k_o]
    for i in range(len(k_list)):
        if k_list[i] is not None:
            k_list[i] = percentile_to_value(k_list[i], parent_distribution)
    return proportion_destined_value(parent_distribution, r_mean, r_sd, k_list[0], k_list[1], k_list[2], k_list[3],
                                     area_all_distributions)


def step_proportion_destined_percentile(parent_distribution, reg_coefficient, sd_reg_coefficient, percentile_step):
    # parent zones are in the first column of every row and the percent of the parent zone's offspring that are destined
    # to each offspring zone in the second column of every row
    stepwise_percentile_list = []
    above_k_p = 1 - percentile_step
    below_k_p = 1
    while below_k_p > 0.5:
        step_list_parent = [[above_k_p, below_k_p]]
        above_k_o = 1 - percentile_step
        below_k_o = 1
        step_list_parents_offspring = []
        above_k_p_v = percentile_to_value(above_k_p, parent_distribution)
        below_k_p_v = percentile_to_value(below_k_p, parent_distribution)
        all_distributions = offspring_distributions(parent_distribution, reg_coefficient, sd_reg_coefficient,
                                                    above_k_v_p=above_k_p_v, below_k_v_p=below_k_p_v)
        area_all_distributions = area_under_distributions(all_distributions)

        while below_k_o > 0.001:
            step_list_offspring = [[above_k_o, below_k_o], proportion_destined_percentile(
                parent_distribution, reg_coefficient, sd_reg_coefficient, above_k_p, below_k_p, above_k_o, below_k_o,
                area_all_distributions)]

            step_list_parents_offspring.append(step_list_offspring)
            above_k_o = round(above_k_o - percentile_step, round_number)
            below_k_o = round(below_k_o - percentile_step, round_number)
        step_list_parent.append(step_list_parents_offspring)

        stepwise_percentile_list.append(step_list_parent)
        above_k_p = round(above_k_p - percentile_step, round_number)
        below_k_p = round(below_k_p - percentile_step, round_number)
    return stepwise_percentile_list


# REPRODUCING FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def final_super_to_parent(final_super_distribution):
    super_max_index = len(final_super_distribution) - 1
    super_parent_max_index = final_super_distribution[0][6][1]

    final = []

    # If it's bigger than the parent, make it only as big as the bound
    if super_max_index > super_parent_max_index:
        super_start_index = (super_max_index - super_parent_max_index) // 2
        super_end_index = super_start_index + super_parent_max_index + 1

    # If's it's equal to or smaller than the parent, make it as big as it is already
    else:
        super_start_index = 0
        super_end_index = len(final_super_distribution)

    for row_num in range(super_start_index, super_end_index):
        final_row = []
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


# NOT USED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def z_erf(num):
#     return erf(num / (2 ** 0.5))


# main()

# keep in mind that we have a wrong increment in the superimposed distribution function
# 0.25 -> 5, 0.5 -> 3, 0.75 -> ~7
