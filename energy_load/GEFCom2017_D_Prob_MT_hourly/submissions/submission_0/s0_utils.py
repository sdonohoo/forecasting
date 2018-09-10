from dateutil.relativedelta import relativedelta
from statsmodels.regression.quantile_regression import QuantReg
import pandas as pd
from functools import reduce
import copy
from sklearn.ensemble import RandomForestRegressor


def get_month_day_range(date):
    # Replace the date in the original timestamp with day 1
    first_day = date + relativedelta(day=1)
    # Replace the date in the original timestamp with day 1
    # Add a month to get to the first day of the next month
    # Subtract one day to get the last day of the current month
    last_day = date + relativedelta(day=1, months=1, days=-1, hours=23)
    return first_day, last_day


def split_train_validation(df, fct_horizon, datetime_col):
    i_round = 0
    for fct, horizon in fct_horizon:
        i_round += 1
        train = df.loc[df[datetime_col] < fct, ].copy()
        validation = df.loc[(df[datetime_col] >= horizon[0]) & (df[datetime_col] <= horizon[1]), ].copy()

        yield i_round, train, validation


def train_single_quantile(train_df_single, q, config):
    target_col = config['target_col']
    feature_cols = config['feature_cols']
    quant_reg_max_iter = config['quant_reg_max_iter']
    model = QuantReg(train_df_single[target_col], train_df_single[feature_cols])
    model_fit = model.fit(q=q, max_iter=quant_reg_max_iter)

    return model_fit


def train_single_group(train_df_single, group_name, config):
    quantiles = config['quantiles']
    models_all = [train_single_quantile(train_df_single.copy(), q, config) for q in quantiles]

    models_dict = {}
    for q, m in zip(quantiles, models_all):
        models_dict['Q' + str(int(q*100))] = m

    return group_name, models_dict


def train_single_group_pnt(train_df_single, config):

    model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    model_fit = model.fit(train_df_single[config['feature_cols']],
                          train_df_single[config['target_col']])

    return model_fit


def train_pnt(train_df, config):

    models_all = train_df.groupby(config['group_cols']).\
        apply(lambda g: train_single_group_pnt(g, config))

    return models_all


def predict_single_group_pnt(test_df_single, model, config):
    feature_cols = config['feature_cols']
    grain_cols = config['grain_cols']
    datetime_col = config['datetime_col']
    grain_datetime_cols = grain_cols[:]
    grain_datetime_cols.append(datetime_col)
    output = test_df_single[grain_datetime_cols].copy()

    output['predict'] = model.predict(test_df_single[feature_cols])

    return output


def predict_pnt(test_df, models_all, config):
    group_cols = config['group_cols']
    group_names = models_all.index.values

    predictions_all = []

    for g in group_names:
        if not isinstance(g, list):
            g = [g]
        group_masks = []
        for i in range(len(group_cols)):
            group_masks.append(test_df[group_cols[i]] == g[i])

        group_mask_final = reduce(lambda x, y: x & y, group_masks)

        predictions_all.append(predict_single_group_pnt(
            test_df.loc[group_mask_final, ], models_all.loc[g, ][0], config))

    predictions_final = pd.concat(predictions_all)

    return predictions_final


def score_single_group(test_df_single, models_dict, config):
    feature_cols = config['feature_cols']
    grain_cols = config['grain_cols']
    datetime_col = config['datetime_col']
    grain_datetime_cols = grain_cols[:]
    grain_datetime_cols.append(datetime_col)
    output = test_df_single[grain_datetime_cols].copy()
    for q, m in models_dict.items():
        output[q] = m.predict(test_df_single[feature_cols])

    return output


def train(train_df, parallel, config):
    group_cols = config['group_cols']
    train_df_grouped = train_df.groupby(group_cols)

    models_all = parallel\
        (delayed(train_single_group)(group, name, copy.deepcopy(config)) for name, group in train_df_grouped)

    models_all_dict = {}
    for k, v in models_all:
        models_all_dict[k] = v

    return models_all_dict


def predict(test_df, models_all, config):
    group_cols = config['group_cols']
    group_names = models_all.keys()

    predictions_all = []

    for g in group_names:
        group_masks = []
        for i in range(len(group_cols)):
            group_masks.append(test_df[group_cols[i]] == g[i])

        group_mask_final = reduce(lambda x, y: x & y, group_masks)

        predictions_all.append(score_single_group(test_df.loc[group_mask_final, ], models_all[g], config))

    predictions_final = pd.concat(predictions_all)

    return predictions_final
