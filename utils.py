import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

import ulmo
import pymannkendall as mk


# from myfunc.py
# ---------------------------------------------------------------------------
def movfunc(tensor, k, axis=-1, gfunc=np.mean):

    dims = list(tensor.shape)
    slcs = [slice(0, i) for i in dims]

    n = dims[axis] - k + 1  # new dim for axis of moving window

    tensor_list = []
    for i in range(k):
        slcs[axis] = slice(i, i + n)
        tensor_list.append(tensor[slcs])  # using tuple avoid warning message

    tensor_sm = gfunc(np.stack(tensor_list), axis=0)
    return tensor_sm


# from hydownload.py
# ---------------------------------------------------------------------------
def importusgssite(siteno):
    sitename = {}
    sitename = ulmo.usgs.nwis.get_site_data(
        siteno, service="daily", period="all", methods='all')
    sitename = pd.DataFrame(sitename['00060:00003']['values'])
    sitename['dates'] = pd.to_datetime(pd.Series(sitename['datetime']))
    sitename.set_index(['dates'], inplace=True)
    sitename['Q'] = sitename['value'].astype(float)
    sitename['qcode'] = sitename['qualifiers']
    sitename = sitename.drop(['datetime', 'qualifiers', 'value'], axis=1)
    sitename = sitename.replace('-999999', np.nan)
    return sitename


# from hyclean.py
# ---------------------------------------------------------------------------
def find_gaps(arr):
    arr = np.concatenate((np.array([1]), arr, np.array([1])))
    d = np.diff(np.isnan(arr).astype(int))
    sidx = np.where(d == 1)[0]
    eidx = np.where(d == -1)[0]
    outarr = np.vstack([sidx, eidx]).transpose()
    return outarr
# extract gaps: [arr[item[0]:item[1]] for item in outarr]
# get length of gaps: np.diff(outarr, axis=1)


# shorten: remove NaNs at HEAD and TAIL or not
def quick_interp(arr, method='linear', shorten=False):
    sidx, eidx = 0, len(arr) - 1
    while np.isnan(arr[sidx]):
        sidx += 1
    while np.isnan(arr[eidx]):
        eidx -= 1

    x = np.arange(eidx - sidx + 1)
    y = arr[sidx:eidx + 1]

    interp_func = interp1d(x[~np.isnan(y)], y[~np.isnan(y)], kind=method)
    arr[sidx:eidx + 1] = interp_func(x)

    if shorten:
        arr = arr[sidx:eidx + 1]

    return arr


# mglen: max gap length
def fill_small_gaps(arr, mglen=10):
    gap_idx = find_gaps(arr)
    out_arr = quick_interp(arr)
    for [sidx, eidx] in gap_idx:
        if eidx - sidx > mglen:
            out_arr[sidx:eidx] = np.nan
    return out_arr


def shorten_time_series(ts, winsize=1, valid_days_thr=1):

    assert winsize >= valid_days_thr

    n = len(ts)
    mat = np.zeros([n + winsize - 1, winsize]).astype(bool)
    for i in range(winsize):
        mat[i:n + i, i] = ~np.isnan(ts.values)

    y1 = np.sum(mat[winsize - 1:, :], axis=1)
    y2 = np.sum(mat[:n, :], axis=1)

    sidx, eidx = 0, n - 1
    while (np.isnan(ts.values[sidx])) | (y1[sidx] < valid_days_thr):
        sidx += 1
    while (np.isnan(ts.values[eidx])) | (y2[eidx] < valid_days_thr):
        eidx -= 1
    return ts[sidx:eidx + 1]


# thr: tolerable difference between time steps, in timedelta type
# for example, thr = pd.Timedelta(10, 's') or thr = np.timedelta64(1, 'D')
def convert_time_series(tf, ts, thr):
    ts = ts.sort_index()
    n_step = len(tf)

    brk_ticks = list(range(0, n_step, 200))
    rem = n_step - brk_ticks[-1]
    if rem < 100:
        brk_ticks[-1] += rem
    else:
        brk_ticks = brk_ticks + [n_step]

    out_ts = pd.Series(np.zeros(n_step) * np.nan, index=tf)
    for sidx, eidx in zip(brk_ticks[:-1], brk_ticks[1:]):
        tf2 = tf[sidx:eidx]
        ts2 = ts[(ts.index >= tf2[0]) & (ts.index <= tf2[-1])]

        xx, yy = np.meshgrid(ts2.index, tf2)
        d = np.abs(xx - yy, dtype='timedelta64[s]')

        vmat = np.vstack([ts2.values] * len(tf2))
        vmat[d >= thr] = np.nan
        out_ts[sidx:eidx] = np.nanmean(vmat, axis=1)

    return out_ts

# from hyanalysis.py
# ---------------------------------------------------------------------------
def splityear(ts, brk_date='01-01'):

    years = np.unique(ts.index.year)
    sdate = str(years[0]) + '-' + brk_date
    edate = str(years[-1] + 1) + '-' + brk_date
    tf = pd.date_range(sdate, edate, freq='D')[:-1]

    thr = np.timedelta64(1, 'D')
    ts2 = convert_time_series(tf, ts, thr)

    leap_day_index = (ts2.index.month == 2) & (ts2.index.day == 29)
    ts2 = ts2.drop(ts2.index[leap_day_index])

    assert len(ts2) % 365 == 0
    hys = ts2.values.reshape([-1, 365])

    col_name = ts2.index[:365].strftime('%m-%d')
    df_hys = pd.DataFrame(hys, index=years, columns=col_name)
    return df_hys


def calculate_exceeding_jd(hys, perc):
    hys_cum = np.cumsum(hys, axis=1)
    hys_thr = hys_cum[:, -1] * perc / 100
    exceeding_jd = np.argmax(hys_cum > hys_thr[:, np.newaxis], axis=1)
    return exceeding_jd


def calculate_hydrometric(df_hys, name='mean'):

    hys = df_hys.values
    midx = np.array([item[:2] for item in df_hys.columns])

    if name == 'mean':
        return np.mean(hys, axis=1)
    elif name == 'median':
        return np.median(hys, axis=1)
    elif name == 'std':
        return np.std(hys, axis=1)
    elif name == 'skew':
        hys_mean = calculate_hydrometric(df_hys, 'mean') + 1e-16
        hys_median = calculate_hydrometric(df_hys, 'median') + 1e-16
        return hys_median / hys_mean
    elif name == 'range':
        return np.percentile(hys, 90, axis=1) - np.percentile(hys, 10, axis=1)
    elif name == 'max':
        return np.max(hys, axis=1)
    elif name == 'min':
        return np.min(hys, axis=1)
    elif name == '5p':
        return np.percentile(hys, 5, axis=1)
    elif name == '10p':
        return np.percentile(hys, 10, axis=1)
    elif name == '25p':
        return np.percentile(hys, 25, axis=1)
    elif name == '75p':
        return np.percentile(hys, 75, axis=1)
    elif name == '90p':
        return np.percentile(hys, 90, axis=1)
    elif name == '95p':
        return np.percentile(hys, 95, axis=1)
    elif name == 'n0':
        return np.sum(hys == 0, axis=1)
    elif name == 'min_jd':
        return np.argmin(hys, axis=1)
    elif name == 'max_jd':
        return np.argmax(hys, axis=1)
    elif name == 'cen_jd':
        return hys.dot(np.arange(365)) / np.sum(hys, axis=1)
    elif name == 'sum':
        return np.sum(hys, axis=1)

    elif name == '25p_jd':
        return calculate_exceeding_jd(hys, 25)
    elif name == '50p_jd':
        return calculate_exceeding_jd(hys, 50)
    elif name == '75p_jd':
        return calculate_exceeding_jd(hys, 75)

    elif name == 'jan':
        return np.mean(hys[:, midx == '01'], axis=1)
    elif name == 'feb':
        return np.mean(hys[:, midx == '02'], axis=1)
    elif name == 'mar':
        return np.mean(hys[:, midx == '03'], axis=1)
    elif name == 'apr':
        return np.mean(hys[:, midx == '04'], axis=1)
    elif name == 'may':
        return np.mean(hys[:, midx == '05'], axis=1)
    elif name == 'jun':
        return np.mean(hys[:, midx == '06'], axis=1)
    elif name == 'jul':
        return np.mean(hys[:, midx == '07'], axis=1)
    elif name == 'aug':
        return np.mean(hys[:, midx == '08'], axis=1)
    elif name == 'sep':
        return np.mean(hys[:, midx == '09'], axis=1)
    elif name == 'oct':
        return np.mean(hys[:, midx == '10'], axis=1)
    elif name == 'nov':
        return np.mean(hys[:, midx == '11'], axis=1)
    elif name == 'dec':
        return np.mean(hys[:, midx == '12'], axis=1)

    elif name in [
        'jan_p', 'feb_p', 'mar_p', 'apr_p', 'may_p', 'jun_p',
        'jul_p', 'aug_p', 'sep_p', 'oct_p', 'nov_p', 'dec_p'
    ]:
        hys_sum = calculate_hydrometric(df_hys, 'sum') + 1e-16
        if name == 'jan_p':
            return np.sum(hys[:, midx == '01'], axis=1) / hys_sum
        elif name == 'feb_p':
            return np.sum(hys[:, midx == '02'], axis=1) / hys_sum
        elif name == 'mar_p':
            return np.sum(hys[:, midx == '03'], axis=1) / hys_sum
        elif name == 'apr_p':
            return np.sum(hys[:, midx == '04'], axis=1) / hys_sum
        elif name == 'may_p':
            return np.sum(hys[:, midx == '05'], axis=1) / hys_sum
        elif name == 'jun_p':
            return np.sum(hys[:, midx == '06'], axis=1) / hys_sum
        elif name == 'jul_p':
            return np.sum(hys[:, midx == '07'], axis=1) / hys_sum
        elif name == 'aug_p':
            return np.sum(hys[:, midx == '08'], axis=1) / hys_sum
        elif name == 'sep_p':
            return np.sum(hys[:, midx == '09'], axis=1) / hys_sum
        elif name == 'oct_p':
            return np.sum(hys[:, midx == '10'], axis=1) / hys_sum
        elif name == 'nov_p':
            return np.sum(hys[:, midx == '11'], axis=1) / hys_sum
        elif name == 'dec_p':
            return np.sum(hys[:, midx == '12'], axis=1) / hys_sum

    elif name in ['max7d', 'min7d', 'max7d_jd', 'min7d_jd']:
        hys_ma = movfunc(hys, k=7)
        if name == 'max7d':
            return np.max(hys_ma, axis=1)
        elif name == 'min7d':
            return np.min(hys_ma, axis=1)
        elif name == 'max7d_jd':
            return np.argmax(hys_ma, axis=1)
        elif name == 'min7d_jd':
            return np.argmin(hys_ma, axis=1)

    elif name == 'si':
        hys_sum = calculate_hydrometric(df_hys, 'sum')
        hys_mean = calculate_hydrometric(df_hys, 'mean')
        hys_var = np.sum(np.abs(hys - hys_mean[:, np.newaxis]), axis=1)
        return hys_var / (hys_sum + 1e-16)


# ---------------------------------------------------------------------------
# trend analysis tools (write exclusively for this project)
# ---------------------------------------------------------------------------
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

    tril_idx = np.tril_indices(n, k=-1)  # lower triangle without diagonal
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
