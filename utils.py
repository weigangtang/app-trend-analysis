import numpy as np

import glob

import pymannkendall as mk
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


def extract_keywords(fpath_frame):
    elems = fpath_frame.split('{}')

    n = len(elems) - 1
    input_ = ['*'] * n
    file_list = glob.glob(fpath_frame.format(*input_))

    keywords_list = []
    for file in file_list:
        file = file.replace(elems[0], '').replace(elems[-1], '')
        for elem in elems[1:-1]:
            file = file.replace(elem, '|', 1)  # only replace first occurance
        keywords = file.split('|')
        keywords_list.append(keywords)
    return keywords_list


def simple_linear_regression(x, y):
    idx = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[idx], y[idx]
    x = x.reshape(-1, 1)

    model = LinearRegression().fit(x, y)
    slp, intp = model.coef_[0], model.intercept_
    rsq = model.score(x, y)
    return slp, intp, rsq


def digitize_pvalue(pvalue, sig_level=[0.01, 0.05, 0.1]):
    return len(sig_level) - np.digitize(pvalue, sig_level)


def run_mktest(t, y, method='original'):

    assert len(t) == len(y)

    indx = ~np.isnan(t) & ~np.isnan(y)
    t, y = t[indx], y[indx]

    n = np.sum(indx)

    if method == 'original':
        mkout = mk.original_test(y)
    if method == 'yue':
        mkout = mk.yue_wang_modification_test(y)
    if method == 'rao':
        mkout = mk.hamed_rao_modification_test(y)
    if method == 'prewhiten':
        mkout = mk.pre_whitening_modification_test(y)
    if method == 'trendfree':
        mkout = mk.trend_free_pre_whitening_modification_test(y)

    slp, intp, pvalue = mkout.slope, mkout.intercept, mkout.p
    intp -= slp * t[0]
    pvalue_d = digitize_pvalue(pvalue) * np.sign(slp)
    return slp, intp, pvalue, pvalue_d, n


def variance_s(x):
    n = len(x)
    tp = np.array([np.sum(x == xi) for xi in np.unique(x)])
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
    return var_s

def get_pair_slope(y):

    n = len(y)
    x = np.arange(n)

    xmat = x[:, np.newaxis] - x[np.newaxis, :]  # i - j
    ymat = y[:, np.newaxis] - y[np.newaxis, :]  # vi - vj

    tril_idx = np.tril_indices(n, k=-1)  # lower triangle index without diagonal
    xarr = xmat[tril_idx]
    yarr = ymat[tril_idx]

    slps = yarr / xarr
    return slps

def sens_slope_lub(y, alpha=0.05):

    y2 = y[~np.isnan(y)]

    n = len(y2)
    k = n * (n - 1) / 2  # number of pairs

    var_s = variance_s(y2)

    c = norm.ppf(1 - alpha / 2) * np.sqrt(var_s)
    idx_lo = np.round((k - c) / 2).astype(int)
    idx_up = np.round((k + c) / 2 + 1).astype(int)

    slps = get_pair_slope(y)
    slps = slps[~np.isnan(slps)]

    slp = np.median(slps)
    slp_lo, slp_up = np.sort(slps)[[idx_lo, idx_up]]

    return slp, slp_lo, slp_up
    