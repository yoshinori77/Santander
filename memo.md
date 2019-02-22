# データ
## train
- trainのtarget=1はtarget=0よりもstdが大きめ
- 匿名データ。変数の意味や個人情報は不明
- 前処理済み？特徴量が全体的に正規分布すぎる
  - もしかしたらlog変換してるかも？
  - 標準化されている可能性もあるかも？

## Test
- testとtrainデータのレコード数が等しい。
  - もしかしたらtestとtrainデータに関連があるかも（Leak）
  - train targetを加工して使うべき？

## feature engineering
- featuretoolsで特徴量増やす
- EDAしてみる
  - 相関
- gokinjo(knn-based feature extract)
- rank gauss

## modeling
- boosting
  - LightGBM, CatBoost, Adaboost, GBDT
- 不均衡データなのでundersampling + bagging(Random Forest)
- stacking（predict_proba)
- クラスタリングが有効かも
- DNN

## 評価
- roc_auc_score
- accuracy_score
- confusion_matrix
- classification_report

## Try
- kernel, discussionを読み込む
- 
