# 機械学習パイプライン

私が普段使っているパイプラインです。

sampleデータは[こちら](<https://github.com/SasakiPeter/test_pubchempy>)で生成しました。

## 実装済み機能一覧

### 手法

* LightGBM, CatBoost
* RandomForest
* Ridge, Lasso, LinearRegression
* LogisticRegression
* SVR, SVC
* NN

### 評価指標

* RMSE, R2
* Logloss, roc-auc

### アンサンブル

* stacking

### 可視化

* Feature Importance
* CVをテーブル形式で出力

## 環境構築

docker で仮想環境を構築しています。次のコマンドで、環境が作成されます。

```shell
$ docker-compose up -d
```

## フォルダ構成

configフォルダ内の設定ファイルに基づいて、学習が行われます。

```
|- config/
	|- settings.py        - 設定ファイル
|- data/
	|- train.csv
	...
|- docker/
	|- Dockerfile
	|- requirements.txt
|- models/
	|- ...                - 学習済みモデル
|- pipeline/
	|- conf/
	|- core/
		|- management/
			|- commands/
				|- ...    - 実行コマンド
	|- predict/           - 予測モジュール
	|- preprocess/        - 前処理モジュール
	|- train/             - 学習モジュール
|- submits/
	|- ...                - 予測ファイル
```

## 実行

前処理、学習、予測までの全てのプロセスが実行されます。

```shell
$ docker-compose exec python manage.py sample runall
```

前処理のみ実行されます。

```shell
$ docker-compose exec python manage.py sample runpreprocess
```

学習のみ実行されます。

```shell
$ docker-compose exec python manage.py sample runtrain
```

予測のみ実行されます。

```shell
$ docker-compose exec python manage.py sample runtrain
```

