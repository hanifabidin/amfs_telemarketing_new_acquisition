import datetime
import pandas as pd
import numpy as np
import numba
from dateutil.relativedelta import relativedelta


def to_numeric(df, npint_type, *feature):
    cols = []
    for col in feature:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            cols.append(col)
        except KeyError as err:
            print('DataFrame doesnt have column: {}'.format(err))
    if npint_type is not None:
        df.dropna(subset=cols, how='any', inplace=True)
        for col in cols:
            df[col] = df[col].astype(npint_type)


def fill_numeric(df, fill_value, astype, *feature):
    for col in feature:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(fill_value)
            df[col] = df[col].astype(astype)
        except KeyError as err:
            print('DataFrame doesnt have column: {}'.format(err))

def next_snapshot(snapshot, in_format='%Y%m', out_format='%Y%m'):
    current_date = datetime.datetime.strptime(snapshot, in_format)
    next_month = current_date + relativedelta(months=1)
    return next_month.strftime(out_format)

def next_month(snapshot, in_format='%Y%m', out_format='%y%m'):
    current_date = datetime.datetime.strptime(snapshot, in_format)
    next_month = current_date + relativedelta(months=1)
    return next_month.strftime(out_format)

def next_month_my(snapshot, in_format='%Y%m', out_format='%m%y'):
    current_date = datetime.datetime.strptime(snapshot, in_format)
    next_month = current_date + relativedelta(months=1)
    return next_month.strftime(out_format)


def last_month(snapshot, in_format='%Y%m', out_format='%y%m'):
    current_date = datetime.datetime.strptime(snapshot, in_format)
    next_month = current_date - relativedelta(months=1)
    return next_month.strftime(out_format)


def last_day(snapshot, in_format='%Y%m', out_format='%Y-%m-%d'):
    any_day = datetime.datetime.strptime(snapshot, in_format)
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    last_day = next_month - datetime.timedelta(days=next_month.day)
    return last_day.strftime(out_format)


def to_format(snapshot, in_format='%Y%m', out_format='%y%m'):
    return datetime.datetime.strptime(snapshot, in_format).strftime(out_format)


def month_range(from_date, to_date=None, period=None, offset=1, in_format='%Y%m', out_format='%y%m'):
    result = []

    from_dt = datetime.datetime.strptime(from_date, in_format)
    if to_date is not None:
        to_dt = datetime.datetime.strptime(to_date, in_format)
        delta = (from_dt - to_dt).days / 30
    elif period is not None:
        delta = period
    else:
        return result

    assert delta > 0, 'from_date must be greater than to_date in reversed range'

    k1 = offset
    while k1 <= delta:
        past = from_dt - relativedelta(months=k1)
        result.append(past.strftime(out_format))
        k1 += 1
    return result


def month_range_Ym(from_date, to_date=None, period=None, offset=1, in_format='%Y%m', out_format='%Y%m'):
    result = []

    from_dt = datetime.datetime.strptime(from_date, in_format)
    if to_date is not None:
        to_dt = datetime.datetime.strptime(to_date, in_format)
        delta = (from_dt - to_dt).days / 30
    elif period is not None:
        delta = period
    else:
        return result

    assert delta > 0, 'from_date must be greater than to_date in reversed range'

    k1 = offset
    while k1 <= delta:
        past = from_dt - relativedelta(months=k1)
        result.append(past.strftime(out_format))
        k1 += 1
    return result



# these are used for modelling
@numba.vectorize
def _process_currency(x):
    x = x / 1000
    if x / 1000 == 0:
        return x
    else:
        flg = 1 if x > 0 else -1
        x = abs(x)
        ds = int(np.log10(x))
        factor = pow(10, ds - 2)

        return flg * round(x / factor) * factor if factor != 0 else factor


@numba.vectorize
def _rounding(x):
    return round(x, 3)


def pre_process_numeric(df, feature, currency=False):
    func = _process_currency if currency else _rounding

    if feature not in df:
        # print('{} not in data-set'.format(feature))
        return pd.Series([feature, np.nan], index=['old', 'new'])
    new_col = feature + '_new'
    # df[new_col] = df[feature].apply(func)
    df[new_col] = func(df[feature].values)
    # df.drop(feature, axis=1, inplace=True)

    if not currency:
        df[new_col] = df[new_col].astype(np.float32)

    return pd.Series([feature, new_col], index=['old', 'new'])


def gen_hour_dummies(prefix, start=8, end=17):
    df = pd.DataFrame(range(start, end + 1), columns=[prefix])
    df[prefix] = df[prefix].astype(np.uint8)
    dummies = pd.get_dummies(df[prefix], prefix=prefix)
    return pd.concat([df, dummies], axis=1)

