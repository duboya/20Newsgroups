import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV


def opt_gbdt(x_train, y_train, x_test, y_test):
    params = {'objective': ['multi:softprob'],
              'n_estimators': [100, 300, 500, 1000],
              # 'max_depth': range(3,10,2),
              # 'min_child_weight': range(1,6,2),
              # 'gamma': [0, 0.001],
              # 'reg_alpha': [0,1]
              }

    # Initialize the model
    xgc = xgb.XGBClassifier()

    # Apply GridSearch to the model
    grid = model_selection.GridSearchCV(xgc, params)
    grid.fit(x_train, y_train)

    xgc_best = grid.best_estimator_
    print(grid.best_estimator_)

    print("Accuracy is %0.3f " % grid.score(x_test, y_test))
    print("Best_score:{}".format(grid.best_score_))
    y_hat = grid.predict(x_test)
    print(metrics.classification_report(y_test, y_hat))
    cross = pd.crosstab(y_hat, y_test)
    print(cross)
    print(y_hat)


def opt_gbm(x_train, y_train, x_test, y_test):
    params = {'objective': ['multi:softprob'],
              'n_estimators': [100, 300, 500, 1000],
              # 'max_depth': range(3,10,2),
              # 'min_child_weight': range(1,6,2),
              # 'gamma': [0, 0.001],
              # 'reg_alpha': [0,1]
              }

    # Initialize the model
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=50,
                                  learning_rate=0.1, n_estimators=43, max_depth=6,
                                  metric='rmse', bagging_fraction=0.8, feature_fraction=0.8)
    params_test1 = {
        'max_depth': range(3, 8, 2),
        # 'num_leaves': range(50, 170, 30)
    }
    gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5,
                            verbose=1, n_jobs=4)
    gsearch1.fit(x_train, y_train)
    print('gsearch1.best_params_:{}'.format(gsearch1.best_params_))
    print('gsearch1.best_score_:{}'.format(gsearch1.best_score_))

    lgb_best = gsearch1.best_estimator_
    pred_test = lgb_best.predict(x_test)
    print(metrics.classification_report(y_test, pred_test))
    return pred_test


def opt_lgb(x_train, y_train, x_val, y_val, x_test, y_test):
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 20,
        'metric': 'multi_error',
        # 'num_leaves': 300,
        # 'min_data_in_leaf': 100,
        'learning_rate': 0.01,
        # 'feature_fraction': 0.8,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        # 'lambda_l1': 0.4,
        # 'lambda_l2': 0.5,
        # 'min_gain_to_split': 0.2,
        # 'verbose': 5,
        # 'is_unbalance': True
    }

    # train
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=200)

    print('Start predicting...')

    preds = gbm.predict(x_test, num_iteration=gbm.best_iteration)  # 输出的是概率结果
    # output results
    y_preds = []
    for pred in preds:
        y_preds.append(int(np.argmax(pred)))
    print(metrics.classification_report(y_test, y_preds))
    # output feature importance
    importance = gbm.feature_importance()
    names = gbm.feature_name()
    with open('./feature_importance.txt', 'w+') as file:
        for index, im in enumerate(importance):
            string = names[index] + ', ' + str(im) + '\n'
            file.write(string)
    return y_preds


def opt_lgb_v2(x_train, y_train, x_test, y_test):
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 20,
        'metric': 'multi_error',
        'num_leaves': 300,
        'min_data_in_leaf': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.4,
        'lambda_l2': 0.5,
        'min_gain_to_split': 0.2,
        'verbose': 5,
        'is_unbalance': True
    }

    # train
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=200)

    print('Start predicting...')

    preds = gbm.predict(x_test, num_iteration=gbm.best_iteration)  # 输出的是概率结果

    for pred in preds:
        result = int(np.argmax(pred))

    importance = gbm.feature_importance()
    names = gbm.feature_name()
    with open('./feature_importance.txt', 'w+') as file:
        for index, im in enumerate(importance):
            string = names[index] + ', ' + str(im) + '\n'
            file.write(string)
    return preds
