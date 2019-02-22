import pandas as pd
import numpy as np
import joblib, re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn import manifold, decomposition, ensemble
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from boruta import BorutaPy
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import lightgbm as lgb
import featuretools as ft
import optuna
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


# -------------------
# 前処理
# -------------------

def item_encoding(event_df):
    event_df['Tシャツフラグ'] = event_df['商品名'].str.match('.*(Ｔシャツ|Tシャツ|Tｼｬﾂ|TEE|Tee|T).*')
    event_df['タオルフラグ'] = event_df['商品名'].str.match('.*(タオル|ﾀｵﾙ).*')
    event_df['ライトフラグ'] = event_df['商品名'].str.match('.*(ライト|ﾗｲﾄ|LIGHT|Light).*')
    event_df['バンダナフラグ'] = event_df['商品名'].str.match('.*(バンダナ|ﾊﾞﾝﾀﾞﾅ).*')
    event_df['キーホルダーフラグ'] = event_df['商品名'].str.match('.*(キーホルダー|ｷｰﾎﾙﾀﾞｰ).*')
    event_df['バッグフラグ'] = event_df['商品名'].str.match('.*(バッグ|ﾊﾞｯｸﾞ|バック|ﾊﾞｯｸ|BAG|bag).*')
    event_df['写真集フラグ'] = event_df['商品名'].str.match('.*(写真集).*')
    event_df['カレンダーフラグ'] = event_df['商品名'].str.match('.*(カレンダー|ｶﾚﾝﾀﾞｰ).*')
    event_df['パーカーフラグ'] = event_df['商品名'].str.match('.*(パーカー|ﾊﾟｰｶｰ|HOODIE|ニット).*')
    event_df['CAPフラグ'] = event_df['商品名'].str.match('.*(CAP|cap|Cap|キャップ|ｷｬｯﾌﾟ).*')
    event_df['限定フラグ'] = event_df['商品名'].str.match('.*(Bigeast限定).*')
    event_df['子供用フラグ'] = event_df['商品名'].str.match('.*(キッズ|Kids).*')
    event_df['MA-1フラグ'] = event_df['商品名'].str.match('.*(ＭＡ-1|MA-1).*')
    event_df['その他フラグ'] = event_df['商品名'].str.match('(?!.*(Ｔシャツ|Tシャツ|Tｼｬﾂ|TEE|Tee|'+
                        'タオル|ﾀｵﾙ|ライト|ﾗｲﾄ|LIGHT|Light|バンダナ|ﾊﾞﾝﾀﾞﾅ|キーホルダー|ｷｰﾎﾙﾀﾞｰ|'+
                        'バッグ|ﾊﾞｯｸﾞ|バック|ﾊﾞｯｸ|BAG|bag|写真集|カレンダー|ｶﾚﾝﾀﾞｰ|'+
                        'パーカー|ﾊﾟｰｶｰ|HOODIE|CAP|cap|Cap|キャップ|ｷｬｯﾌﾟ|'+
                        'Bigeast限定|キッズ|Kids|ＭＡ-1|MA-1)).*')

    event_df['日時'] = event_df['日時'].apply(
                                  lambda x:
                                      re.sub('(日|\(.\))', '',
                                      re.sub('\.', '-', x)) + ' 00:00:00'
                                          if type(x) == str
                                          and not x.endswith('00:00:00')
                                          else x)
    event_df['曜日'] = pd.to_datetime(event_df['日時']).dt.dayofweek
    unique_event_day = pd.to_datetime(event_df['日時'].unique()).sort_values()
    event_day_dict = {d: i for i, d in enumerate(unique_event_day, 1)}
    event_df['イベント回数'] = pd.to_datetime(event_df['日時']).map(event_day_dict)
    event_df['ツアー前半'] = event_df['イベント回数'].apply(lambda x: 1 if x <= event_df['イベント回数'].max()/2 else 0)
    tour_dict = {0: '後半', 1: '前半'}
    event_df['ツアー'] = event_df['ツアー前半'].map(tour_dict)
    artist_dict = {'Nissy': 1, 'AAA': 1, '東方神起': 1, '浜崎あゆみ': 0, '倖田來未': 0, '和楽器バンド': 0}
    event_df['人気アーティスト'] = event_df['アーティスト'].map(artist_dict)

    target_df = event_df[['品番', '商品名', '日時', '曜日', '会場', '会場規模', 'イベント回数',
                        'ツアー前半', 'Tシャツフラグ', 'タオルフラグ', 'ライトフラグ',
                        'バンダナフラグ', 'キーホルダーフラグ', 'バッグフラグ', '写真集フラグ',
                        'カレンダーフラグ', 'パーカーフラグ', 'CAPフラグ', '限定フラグ', '子供用フラグ', 'MA-1フラグ',
                        'その他フラグ', 'アーティスト', '人気アーティスト', '上代']]

    target_df[['Tシャツフラグ', 'タオルフラグ', 'ライトフラグ', 'バンダナフラグ',
                'キーホルダーフラグ', 'バッグフラグ', '写真集フラグ', 'カレンダーフラグ',
                'パーカーフラグ', 'CAPフラグ', '限定フラグ', '子供用フラグ', 'MA-1フラグ', 'その他フラグ']] = \
    target_df[['Tシャツフラグ', 'タオルフラグ', 'ライトフラグ', 'バンダナフラグ',
                'キーホルダーフラグ', 'バッグフラグ', '写真集フラグ', 'カレンダーフラグ',
                'パーカーフラグ', 'CAPフラグ', '限定フラグ', '子供用フラグ', 'MA-1フラグ', 'その他フラグ']].astype(int)

    flag_columns = [c for c in target_df.columns if 'フラグ' in c]
    flag_df = target_df[flag_columns]

    category_df = pd.DataFrame([flag_df[c].replace(1, c[:-3]) for c in flag_df.columns]).T
    s_category_df = category_df.stack().to_frame()
    grouped = s_category_df[s_category_df[0] != 0].groupby(level=0)
    s_category_df = grouped.last()
    category_series = s_category_df[s_category_df[0] != 0][0].values

    target_df['商品カテゴリ'] = s_category_df[s_category_df[0] != 0][0].values

    return target_df

def count_word(X, column):
    '''文字列をベクトルに変換
    '''
    xindex = X.index
    vectorizer = CountVectorizer()
    # count_vec = vectorizer.fit_transform(X[column])
    vectorizer.fit(X[column])
    count_vec = vectorizer.transform(X[column])
    count_df = pd.DataFrame(count_vec.toarray(), index=xindex,
                            columns=vectorizer.get_feature_names())

    return (vectorizer, count_df)

def pca_reduce_vec(X):
    '''次元削減 PCA
    '''
    xindex = X.index
    pca = decomposition.TruncatedSVD(n_components=2)
    # X_reduced = pca.fit_transform(X)
    pca.fit(X)
    X_reduced = pca.transform(X)
    X_reduced_df = pd.DataFrame(X_reduced, index=xindex)

    return (pca, X_reduced_df)


def rt_pca_reduce_vec(X, n=2):
    '''
    次元削減
    RandomTreesEmbedding & PCA
    '''
    xindex = X.index
    hasher = ensemble.RandomTreesEmbedding(n_estimators=400, random_state=0,
                                           max_depth=8)
    X_transformed = hasher.fit_transform(X)
    pca = decomposition.TruncatedSVD(n_components=n)
    X_reduced = pca.fit_transform(X_transformed)
    X_reduced_df = pd.DataFrame(X_reduced, index=xindex)

    return X_reduced_df

def km_cluster(X, n=100):
    '''クラスタリング（特徴量生成用）
    '''
    xindex = X.index
    km = KMeans(n_clusters=n,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    # X_cluster = km.fit_predict(X)
    km.fit(X)
    X_cluster = km.predict(X)
    X_cluster = pd.Series(X_cluster, index=xindex)

    return (km, X_cluster)

def standard_scaler(X):
    '''標準化
    '''
    xcols = X.columns
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = pd.DataFrame(X, columns=xcols)

    return X

def minmax_scaler(X):
    '''正規化
    '''
    xcols = X.columns
    ms = MinMaxScaler()
    X = ms.fit_transform(X)
    X = pd.DataFrame(X, columns=xcols)

    return X

def get_month_date(X, column):
    '''文字列型の時間から月、日を生成
    '''
    date_df = pd.to_datetime(X[column]).to_frame()
    xindex = X.index
    date_df.index = xindex
    date_df.columns = ['date']
    X = X.join(date_df)
    X['月'] = X['date'].dt.month
    X['日'] = X['date'].dt.day
    X.drop('date', axis=1, inplace=True)

    return X

def featuretools_agg(X, cols, methods=['count', 'max', 'mean']):
    es = ft.EntitySet(id = 'index')
    es.entity_from_dataframe(entity_id = 'data',
                             dataframe = X,
                             index = 'index')
    for col in cols:
        es.normalize_entity(base_entity_id='data',
                            new_entity_id=col,
                            index = col
                            )

    features, feature_names = ft.dfs(entityset = es,
                                     target_entity = 'data',
                                     agg_primitives = methods,
                                     max_depth = 2,
                                     verbose = 1,
                                     n_jobs = -1)

    return features

def featuretools_trans(X, methods=['multiply']):
    es = ft.EntitySet(id = 'index')
    es.entity_from_dataframe(entity_id = 'data',
                             dataframe = X,
                             index = 'index')

    features, feature_names = ft.dfs(entityset = es,
                                     target_entity = 'data',
                                     trans_primitives = methods,
                                     max_depth = 2,
                                     verbose = 1,
                                     n_jobs = 4)

    return features

# -------------------
# カテゴリカル変数 エンコード
# -------------------

def encode_category(X):
    '''カテゴリカル変数を数的変数に変換
    '''
    cols = X.select_dtypes(include='object').columns

    for c in cols:
        lbl = LabelEncoder()
        X[c] = lbl.fit_transform(list(X[c].values))

    return X

def ordinal_encoder(X, feature_columns, target_column):
    ce_ord = ce.OrdinalEncoder(cols=[feature_columns])

    return ce_ord.fit_transform(X, X[target_column])

def sum_encoder(X, feature_columns, target_column):
    ce_sum = ce.SumEncoder(cols = [feature_columns])

    return ce_sum.fit_transform(X, X[target_column])

def helmert_encoder(X, feature_columns, target_column):
    ce_helmert = ce.HelmertEncoder(cols = [feature_columns])

    return ce_helmert.fit_transform(X, X[target_column])

def backward_difference_encoder(X, feature_columns, target_column):
    ce_backward = ce.BackwardDifferenceEncoder(cols = [feature_columns])

    return ce_backward.fit_transform(X, X[target_column])

def polynomial_encoder(X, feature_columns, target_column):
    ce_poly = ce.PolynomialEncoder(cols = [feature_columns])

    return ce_poly.fit_transform(X, X[target_column])

# -------------------
# 特徴選択
# -------------------

def feature_select_boruta(X, y, n=400, max_depth=5, alpha=0.05):
    '''borutaで特徴選択した特徴量を返す
    '''
    rf = RandomForestRegressor(n_estimators=n, max_depth=max_depth, random_state=42, n_jobs=-1)
    feat_selector = BorutaPy(rf, n_estimators='auto', alpha=alpha, verbose=2, random_state=1)
    feat_selector.fit(X.values, y.values)
    X_filtered = feat_selector.transform(X.values)
    # feat_selector.support_
    # feat_selector.ranking_

    return X_filtered

def feature_select_label_boruta(X, y, n=400, max_depth=5, alpha=0.05):
    '''borutaで特徴選択した特徴量のラベルを返す
    '''
    rf = RandomForestRegressor(n_estimators=n, max_depth=max_depth, random_state=42, n_jobs=-1)
    feat_selector = BorutaPy(rf, n_estimators='auto', alpha=alpha, verbose=2, random_state=1)
    feat_selector.fit(X.values, y.values)

    return X.columns[feat_selector.support_]


# -------------------
# 評価
# -------------------

def get_importances(model, xcols):
    '''importanceをランキングで返す
    '''
    importances = [(c, f) for c, f in zip(xcols, model.feature_importances_)]
    sorted_importances = sorted(importances, key=lambda x: -x[1])

    return sorted_importances

def rmse_cv(model, n_folds, X, y):
    '''クロスバリデーションした結果をRMSEで返す（目的変数をlog変換した場合は使えない）
    '''
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse = np.sqrt(-cross_val_score(model, X.values, y,
                                    scoring="neg_mean_squared_error",
                                    cv = kf, n_jobs=-1))
    return rmse

def mae(y_pred, y_test, log=False):
    '''MAEを返す（目的変数をlog変換した場合に対応）
    '''
    if log:
        return mean_absolute_error(np.expm1(y_pred), np.expm1(y_test))
    else:
        return mean_absolute_error(y_pred, y_test)

def rmse(y_pred, y_test, log=False):
    '''RMSEを返す（目的変数をlog変換した場合に対応）
    '''
    if log:
        return np.sqrt(mean_squared_error(np.expm1(y_pred), np.expm1(y_test)))
    else:
        return np.sqrt(mean_squared_error(y_pred, y_test))

def cross_validate(model, X, y, train_index, test_index, log=False):
    '''クロスバリデーションした結果をRMSEで返す（目的変数をlog変換した場合に対応）
    '''
    X_train = X.iloc[train_index, :]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X.iloc[test_index, :])
    if log:
        return rmse(y_test, y_pred, log=True)
    else:
        return rmse(y_test, y_pred)

def parallel_cross_validate(model, X, y, n_folds=5, kfold=False):
    '''並列処理でクロスバリデーションを実行（KFold & ShuffleSplitに対応）
    '''
    if kfold:
        v = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        v = ShuffleSplit(n_splits=n_folds, test_size=0.3, random_state=42)

    indexes = [(train_index, test_index) for train_index, test_index in v.split(X)]
    score = joblib.Parallel(n_jobs=-1) \
                        (joblib.delayed(cross_validate) \
                        (model, X, y, train_index, test_index, log=True) \
                        for train_index, test_index in indexes)

    return score

# -------------------
# 可視化
# -------------------

def residplot(y_pred, y_test, log=False):
    '''残差プロット（目的変数をlog変換した場合に対応）
    '''
    if log:
        sns.residplot(np.expm1(y_pred), np.expm1(y_test),
                    lowess=True, color='g',
                    scatter_kws={'alpha': 0.5})
    else:
        sns.residplot(y_pred, y_test,
                    lowess=True, color='g',
                    scatter_kws={'alpha': 0.5})

# -------------------
# パラメータ最適化
# -------------------

class Lgbm_Objective(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        model = lgb.LGBMRegressor(
                    objective='regression', metric='l2',
                    boosting_type='gbdt',
                    n_estimators=trial.suggest_int('n_estimators', 800, 12800),
                    num_leaves=trial.suggest_int('num_leaves', 10, 1000),
                    learning_rate=trial.suggest_loguniform('learning_rate', 1e-3, 1.0))

        score_lgbm = parallel_cross_validate(model, self.X, self.y, n_folds=5, kfold=False)
        return np.mean(score_lgbm)

def optuna_tuning_lgbm(X, y):
    objective = Lgbm_Objective(X, y)

    study = optuna.create_study()
    study.optimize(objective, n_trials=200)
    print(study.best_trial)
    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study


class Gbdt_Objective(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        regressor_name = trial.suggest_categorical('regressor', 'GBDT')
        regressor_obj = GradientBoostingRegressor(
                            n_estimators = trial.suggest_int('n_estimators', 800, 12800),
                            learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                            max_depth = trial.suggest_int('max_depth', 4, 12),
                            max_features = 'sqrt',
                            loss = 'ls')

        score_gbdt = parallel_cross_validate(regressor_obj, self.X, self.y, n_folds=5, kfold=False)
        return np.mean(score_gbdt)

def optuna_tuning_gbdt(X, y):
    objective = Gbdt_Objective(X, y)

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
    return study

# -------------------
# ブレンディング スタッキング
# -------------------
class AveragingModels(BaseEstimator, RegressorMixin):
    '''
    使い方
    averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
    averaged_models.fit(train.values, y_train)
    stacked_train_pred = averaged_models.predict(train.values)
    '''
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
    使い方
    stacked_averaged_models = StackingAveragedModels(
                                    base_models=(ENet, GBoost, KRR),
                                    meta_model=lasso)
    stacked_averaged_models.fit(train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    '''
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                import ipdb; ipdb.set_trace()
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        X = np.append(X, out_of_fold_predictions, axis=1)

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(X, y)
        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        X = np.append(X, meta_features, axis=1)
        return self.meta_model_.predict(X)

class StackingRegressorModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        result_df = pd.DataFrame()
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            y_pred = model.predict(X)
            result_df['pred'+str(i)] = y_pred
            # instance = clone(model)
            # instance.fit(X, y)
            # y_pred = instance.predict(X)
            # result_df['pred'+str(i)] = y_pred

        self.meta_model.fit(result_df, y)
        return self

    def predict(self, X):
        result_df = pd.DataFrame()
        for i, model in enumerate(self.base_models):
            y_pred = model.predict(X)
            result_df['pred'+str(i)] = y_pred

        return self.meta_model.predict(result_df)
