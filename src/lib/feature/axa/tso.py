import os
import re

import numpy as np
import pandas as pd

from lib import obj, util


class TSO(obj.Feature):

    def __init__(self, data_path, out_path, sep, snapshot):
        obj.Feature.__init__(self, data_path, 'data/03_features/AXA_Features', '\t')
        self.snapshot = snapshot
        self.nextsnapshot = util.next_snapshot(snapshot)
        self.nextmonth = util.next_month(snapshot)
        self.tso_path = os.path.join(self.root,  'data/01_raw/data_from_AMFS/{0}/TSO_{1}.txt'.format(self.snapshot, util.to_format(snapshot)))
        self.tso_nextmonth_path = os.path.join(self.root, 'data/01_raw/data_from_AMFS/{0}/TSO_{1}.txt'.format(self.nextsnapshot, self.nextmonth))
        if not os.path.isfile(self.tso_nextmonth_path):
            self.tso_nextmonth_path = os.path.join(self.root, 'data/01_raw/data_from_AMFS/{0}/ACTIVE_TSO_{1}.txt'.format(self.snapshot, self.nextmonth))
            self.feature_path = os.path.join(self.root, self.out_path, 'TSO_Features_{0}.csv')

    @property
    def create(self):

        tso = pd.read_csv(self.tso_path, sep=self.sep)
        tso_next = pd.read_csv(self.tso_nextmonth_path, sep=self.sep)

        # merge with active tso
        tso = tso.rename(columns={'TSO_CATEGORY': 'TSO_CATEGORY_old'})
        tso = tso.merge(tso_next[['CALL_AGENT_ID', 'TSO_CATEGORY']], on='CALL_AGENT_ID', how='left')
        tso['TSO_CATEGORY'] = tso['TSO_CATEGORY'].fillna(tso['TSO_CATEGORY_old'])

        tso_new = tso_next[tso_next['TSO_LOS'] == 1].reset_index(drop=True)

        # generate performance_cols
        pattern = r'^TSO_.*_LM[1-3]*$'
        performance_cols = filter(lambda x: re.search(pattern, x), tso.columns.values)
        #if len(performance_cols) != 156:
        #    raise ValueError('Missing or Redundancy in performance columns!')
        for col in performance_cols:
            tso_new[col] = 0
        tso_new['TSO_LOS'] = 0

        tso = pd.concat([tso, tso_new], axis=0)

        # convert to numeric
        # make sure the numerical feature only contain numerical values
        util.to_numeric(tso, None, 'TSO_LOS', 'TSO_AGE', 'TSO_DEPENDANT')

        # impute the performance related feature with 0
        for col in performance_cols:
            tso[col] = tso[col].fillna(0)

        categorical_features = {
            'TSO_LOCATION': ['01.BBD', '02.AXTO'],
            'TSO_GENDER': ['01.FEMALE', '02.MALE'],
            'TSO_CATEGORY': ['01.TOP GUN', '02.STRIKER', '03.PLAYMAKER', '04.DEFENDER'],
            'TSO_MARITAL_STATUS': ['01.SIN', '02.MAR'],
            'TSO_EDUCATION': ['01.S2', '02.S1', '03.AKADEMI', '04.SLTA']
        }
        for feature, categories in categorical_features.iteritems():
            tso.loc[~tso[feature].isin(categories), feature] = np.nan

        # impute the empty values
        dict_imp = {
            'TSO_LOCATION': 'MODE',
            'TSO_GENDER': 'MODE',
            'TSO_CATEGORY': 'MODE',
            'TSO_MARITAL_STATUS': 'MODE',
            'TSO_EDUCATION': 'MODE',
            'TSO_LOS': 'MEDIAN',
            'TSO_AGE': 'MEDIAN',
            'TSO_DEPENDANT': 'MEDIAN'
        }
        for feature, method in dict_imp.iteritems():
            if method == 'MODE':
                v = tso[feature].value_counts(dropna=True).index[0]
            if method == 'MEDIAN':
                v = tso[feature].dropna().median()
            tso[feature] = tso[feature].fillna(v)
            print('Imputed missing values in feature %s with value %s' % (feature, unicode(str(v), 'utf8')))

        # make sure there is no null values any more
        print(np.sum([int(tso[col].isnull().values.any()) for col in tso.columns.values]))

        # numerical feature
        feature_columns = ['CALL_AGENT_ID', 'active_flag']

        posfixes = ['LM', 'LM2', 'LM3']
        # col_pattern = '^{0}_(CC|SA)_{1}$'
        # just keep Saving Account
        col_pattern = '^{0}_SA_{1}$'
        col_prefixes = [
            'TSO_DB',
            'TSO_DB_CONTACTED',
            'TSO_CALL',
            'TSO_CALL_CONTACTED',
        ]
        for col_prefix in col_prefixes:
            for posfix in posfixes:
                feature_col = '_'.join([col_prefix, posfix])
                feature_col_pattern = col_pattern.format(col_prefix, posfix)
                cols = filter(lambda x: re.search(feature_col_pattern, x), tso.columns.values)
                if len(cols) != 1:
                    raise ValueError('Error for col: {}'.format(feature_col))
                feature_columns.append(feature_col)
                tso[feature_col] = tso[cols].sum(axis=1)

        products = ['MHL', 'MJK', 'MPK', 'MSP', 'MHP', 'MFC', 'MHS', 'MIR', 'MPKS', 'TOTAL']
        # col_pattern = '^TSO_{0}_(CC|SA)_{1}_{2}$'
        # just keep Saving Account
        col_pattern = '^TSO_{0}_SA_{1}_{2}$'
        for product in products:
            for posfix in posfixes:
                methods = ['APE', 'CASES']
                for method in methods:
                    feature_col = '_'.join(['TSO', product, method, posfix])
                    feature_col_pattern = col_pattern.format(product, method, posfix)
                    cols = filter(lambda x: re.search(feature_col_pattern, x), tso.columns.values)
                    if len(cols) != 1:
                        raise ValueError('Error for col: {}'.format(feature_col))
                    feature_columns.append(feature_col)
                    tso[feature_col] = tso[cols].sum(axis=1)

        other_products = ['MFC', 'MIR', 'MPKS']
        for posfix in posfixes:
            for method in ['APE', 'CASES']:
                feature_col_pattern = '_'.join(['TSO', '{}', method, posfix])
                feature_col = feature_col_pattern.format('OTHER')
                cols = [feature_col_pattern.format(other_product) for other_product in other_products]
                [feature_columns.remove(col) for col in cols]
                feature_columns.append(feature_col)
                tso[feature_col] = tso[cols].sum(axis=1)

        def __func1(row):
            return row[1] * 1.0 / row[0] if row[0] > 0 else 0

        # some feature relate to tso_db_[*] and tso_call_[*]
        for posfix in posfixes:
            feature_col = 'TSO_CONTACT_RATE_BY_DB_{}'.format(posfix)
            tso_db = 'TSO_DB_{}'.format(posfix)
            tso_db_contacted = 'TSO_DB_CONTACTED_{}'.format(posfix)
            feature_columns.append(feature_col)
            tso[feature_col] = tso[[tso_db, tso_db_contacted]].apply(__func1, axis=1)

            feature_col = 'TSO_CONTACT_RATE_BY_CALL_{}'.format(posfix)
            tso_call = 'TSO_CALL_{}'.format(posfix)
            tso_call_contacted = 'TSO_CALL_CONTACTED_{}'.format(posfix)
            feature_columns.append(feature_col)
            tso[feature_col] = tso[[tso_call, tso_call_contacted]].apply(__func1, axis=1)

            # create some new feature based on db and calls
            feature_col = 'average_calls_{}'.format(posfix)
            feature_columns.append(feature_col)
            tso[feature_col] = tso[[tso_db, tso_call]].apply(__func1, axis=1)

            feature_col = 'average_ape_{}'.format(posfix)
            tso_total_ape = 'TSO_TOTAL_APE_{}'.format(posfix)
            tso_total_cases = 'TSO_TOTAL_CASES_{}'.format(posfix)
            feature_columns.append(feature_col)
            tso[feature_col] = tso[[tso_total_cases, tso_total_ape]].apply(__func1, axis=1)

        # categorical feature
        # get dummies
        for feature, categories in categorical_features.iteritems():
            print('Dummy feature: {}'.format(feature))
            print('The values to dummy', categories)
            for category in categories:
                feature_col = '_'.join([feature, category])
                feature_columns.append(feature_col)
                tso[feature_col] = tso[feature].apply(lambda x: 1 if x == category else 0)
            tso.drop(feature, axis=1, inplace=True)

        # add active flag
        tso['active_flag'] = tso['TSO_STATUS'].apply(lambda x: 1 if x == 'ACTIVE' else 0)
        # for new TSO, the active flag is 0
        tso.loc[tso['TSO_LOS'] == 0, 'active_flag'] = 0

        # create more tso feature here
        tso['sum_TSO_DB'] = tso[['TSO_DB_LM', 'TSO_DB_LM2', 'TSO_DB_LM3']].sum(axis=1)
        tso['sum_TSO_DB_contacted'] = tso[['TSO_DB_CONTACTED_{}'.format(posfix) for posfix in posfixes]].sum(axis=1)
        tso['sum_TSO_CALL'] = tso[['TSO_CALL_LM', 'TSO_CALL_LM2', 'TSO_CALL_LM3']].sum(axis=1)
        tso['sum_TSO_CALL_contacted'] = tso[['TSO_CALL_CONTACTED_{}'.format(posfix) for posfix in posfixes]].sum(axis=1)

        tso['rate_TSO_DB_contacted'] = np.round((tso['sum_TSO_DB_contacted'] / tso['sum_TSO_DB']) * 100, 2)
        tso['rate_TSO_CALL_contacted'] = np.round((tso['sum_TSO_CALL_contacted'] / tso['sum_TSO_CALL']) * 100, 2)

        # SUM & AVG APE of all products
        tso['sum_TSO_APE_3m'] = tso[['TSO_TOTAL_APE_LM', 'TSO_TOTAL_APE_LM2', 'TSO_TOTAL_APE_LM3']].sum(axis=1)

        def __func2(row):
            if row[1] < 3:
                return row[0] / row[1]
            else:
                return row[0] / 3

        tso['avg_TSO_APE_3m'] = tso[['sum_TSO_APE_3m', 'TSO_LOS']].apply(__func2, axis=1)
        # SUM & AVG APE of MJK
        tso['sum_TSO_MJK_MHL_APE_3m'] = tso[['TSO_MJK_APE_LM3', 'TSO_MJK_APE_LM2', 'TSO_MJK_APE_LM',
                                             'TSO_MHL_APE_LM3', 'TSO_MHL_APE_LM2', 'TSO_MHL_APE_LM']].sum(axis=1)

        tso['avg_TSO_MJK_MHL_APE_3m'] = tso[['sum_TSO_MJK_MHL_APE_3m', 'TSO_LOS']].apply(__func2, axis=1)
        # SUM & AVG APE of all products
        tso['sum_TSO_CASES_3m'] = tso['TSO_TOTAL_CASES_LM'] + tso['TSO_TOTAL_CASES_LM2'] + tso['TSO_TOTAL_CASES_LM3']

        tso['avg_TSO_CASES_3m'] = tso[['sum_TSO_CASES_3m', 'TSO_LOS']].apply(__func2, axis=1)
        # SUM & AVG APE of MJK
        tso['sum_TSO_MJK_MHL_CASES_3m'] = tso[['TSO_MJK_CASES_LM', 'TSO_MJK_CASES_LM2', 'TSO_MJK_CASES_LM3',
                                               'TSO_MHL_CASES_LM', 'TSO_MHL_CASES_LM2', 'TSO_MHL_CASES_LM3']].sum(axis=1)

        tso['avg_TSO_MJK_MHL_CASES_3m'] = tso[['sum_TSO_MJK_MHL_CASES_3m', 'TSO_LOS']].apply(__func2, axis=1)

        feature_columns += ['TSO_LOS', 'TSO_AGE', 'TSO_DEPENDANT',
                            'sum_TSO_DB', 'sum_TSO_DB_contacted', 'sum_TSO_CALL', 'sum_TSO_CALL_contacted',
                            'rate_TSO_DB_contacted', 'rate_TSO_CALL_contacted', 'sum_TSO_APE_3m', 'avg_TSO_APE_3m',
                            'sum_TSO_MJK_MHL_APE_3m', 'avg_TSO_MJK_MHL_APE_3m', 'sum_TSO_CASES_3m', 'avg_TSO_CASES_3m',
                            'sum_TSO_MJK_MHL_CASES_3m', 'avg_TSO_MJK_MHL_CASES_3m']

        tso = tso[feature_columns].fillna(0)
        tso.to_csv(self.feature_path.format(util.to_format(self.snapshot)), index=False)
