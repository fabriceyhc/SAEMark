#encoding:utf-8

# this script is used to validate the sentences

import nltk
import numpy as np
import statsmodels.api as sm
from fcs import calc_ratio
from utils import generate_precise_number

def get_T_statistic(precise_numbers, ratios):
    X = ratios  # independent variable
    y = precise_numbers  # dependent variable

    # add the constant term (intercept)
    X = sm.add_constant(X)

    # fit the linear regression model
    model = sm.OLS(y, X).fit()

    # get the t statistic and p value of β
    beta_t_statistic = model.tvalues[1]  # the second element corresponds to the t statistic of the intercept term
    beta_p_value = model.pvalues[1]      # the second element corresponds to the p value of the intercept term

    return beta_t_statistic, beta_p_value

def validate_roc_en(
    sentence,
    public_keys,
    device,
    intervals,
    judge_model=None,
    judge_tokenizer=None,
    path_to_params=None,
    feature_mask=None,
    target_layer=20,
):
    max_ind = -1
    ratios = []
    all_precise_numbers = [list() for _ in range(len(public_keys))]
    sentences = nltk.sent_tokenize(sentence)
    for s in sentences:
        # for each sentence, generate the pseudo code
        if max_ind != -1:
            for i, public_key in enumerate(public_keys):
                precise_number = generate_precise_number(public_key, str(max_ind), intervals[i][0], intervals[i][1])
                all_precise_numbers[i].append(precise_number)
        else:
            for i, public_key in enumerate(public_keys):
                precise_number = generate_precise_number(public_key, '', intervals[i][0], intervals[i][1])
                all_precise_numbers[i].append(precise_number)
        s = [s]
        ans = calc_ratio(judge_model, judge_tokenizer, s, path_to_params, device=device, feature_mask=feature_mask, target_layer=target_layer)
        max_ind, ratio = ans[0][0], ans[0][1]
        ratios.append(ratio)

    res = list()

    # calculate every t-value of the series
    for i, precise_numbers in enumerate(all_precise_numbers):
        try:
            t_value, p_value = get_T_statistic(np.array(precise_numbers), np.array(ratios))
            if t_value > 0:
                res.append([i, float(p_value / 2)])
        except: continue
        
    if len(res) == 0:
            return 1
    else:
        return res[0][1]
                
def validate_roc_code(
    sentence, 
    public_keys, 
    device, 
    intervals, 
    judge_model=None,
    judge_tokenizer=None,
    path_to_params=None,
    feature_mask=None
):
    max_ind = -1
    ratios = []
    all_precise_numbers = [list() for _ in range(len(public_keys))]
    sentences = sentence.split('\n')
    sentences = [(s + '\n') for s in sentences if len(s) != 0]
    for s in sentences:
        # for each sentence, generate the pseudo code
        if max_ind != -1:
            for i, public_key in enumerate(public_keys):
                precise_number = generate_precise_number(public_key, str(max_ind), intervals[i][0], intervals[i][1])
                all_precise_numbers[i].append(precise_number)
        else:
            for i, public_key in enumerate(public_keys):
                precise_number = generate_precise_number(public_key, '', intervals[i][0], intervals[i][1])
                all_precise_numbers[i].append(precise_number)
        s = [s]
        ans = calc_ratio(judge_model, judge_tokenizer, s, path_to_params, device=device, feature_mask=feature_mask, target_layer=target_layer)
        max_ind, ratio = ans[0][0], ans[0][1]
        ratios.append(ratio)

    res = list()

    # calculate every t-value of the series
    for i, precise_numbers in enumerate(all_precise_numbers):
        try:
            t_value, p_value = get_T_statistic(np.array(precise_numbers), np.array(ratios))
            if t_value > 0:
                res.append([i, float(p_value / 2)])
        except:
            continue

    if len(res) == 0:
            return 1
    else:
        return res[0][1]
                
def validate_roc_zh(
    sentence, 
    public_keys, 
    device, 
    intervals, 
    judge_model=None,
    judge_tokenizer=None,
    path_to_params=None,
    feature_mask=None
):
    max_ind = -1
    ratios = []
    all_precise_numbers = [list() for _ in range(len(public_keys))]
    sentences = sentence.split('。')
    sentences = [(s + '。') for s in sentences if len(s) != 0]
    for s in sentences:
        # for each sentence, generate the pseudo code
        if max_ind != -1:
            for i, public_key in enumerate(public_keys):
                precise_number = generate_precise_number(public_key, str(max_ind), intervals[i][0], intervals[i][1])
                all_precise_numbers[i].append(precise_number)
        else:
            for i, public_key in enumerate(public_keys):
                precise_number = generate_precise_number(public_key, '', intervals[i][0], intervals[i][1])
                all_precise_numbers[i].append(precise_number)
        s = [s]
        ans = calc_ratio(judge_model, judge_tokenizer, s, path_to_params, device=device, feature_mask=feature_mask, target_layer=target_layer)
        max_ind, ratio = ans[0][0], ans[0][1]
        ratios.append(ratio)

    res = list()

    # calculate every t-value of the series
    for i, precise_numbers in enumerate(all_precise_numbers):
        try:
            t_value, p_value = get_T_statistic(np.array(precise_numbers), np.array(ratios))
            if t_value > 0:
                res.append([i, float(p_value / 2)])
        except: continue

    if len(res) == 0:
            return 1
    else:
        return res[0][1]