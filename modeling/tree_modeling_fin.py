# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import optuna
from optuna import Trial
from imblearn.under_sampling import ClusterCentroids

from imblearn.under_sampling import OneSidedSelection

# random_state
rs = 42 
# 교차검증
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs)
kf = KFold(n_splits=10, shuffle=True, random_state=rs)
# HP tuning 시, scoring 지표
scoring = 'accuracy'  
# 클래스 가중치
ratio = 0.5316642120765832  
# 모델 별 feature importance 계산 타입
fi_type = {"RandomForest" : ["entropy"], "LGBM" : ["split", "gain"], "XGBoost" : ["weight", "gain"]} 


## 모델 학습 전 데이터 준비 함수
def prepare_data(df): # scale
    # valid, test split
    X = df.drop(labels=["취업여부"], axis=1)
    y = df["취업여부"]

    X_resample, y_resample = ClusterCentroids(random_state=42).fit_resample(X, y)

    return train_test_split(X_resample, y_resample, stratify=y_resample, test_size=0.2, random_state=rs)

## confusion-matrix plot
def visualization_confusion_matrix(y_test, y_pred, title): 
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns = ["실패", "성공"], index = ["실패", "성공"])

    #Plotting the confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, cmap="Blues", vmin=0.1)
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

## feature importance plot
def visualization_feature_importance(model, cols, title, type):
    # 모델 별로 feature importance 구하기
    obj = title.split()[0]
    if obj == "LGBM" :
        ftr_importances_values = model.booster_.feature_importance(importance_type=type)
    elif obj == "XGBoost" :
        ftr_importances_values = model.get_booster().get_score(importance_type=type)
    else :
        ftr_importances_values = model.feature_importances_

    ftr_importances = pd.Series(ftr_importances_values, index=cols)
    ftr_top = ftr_importances.sort_values(ascending=False)
    
    plt.figure(figsize=(8, len(cols)*0.5))
    sns.barplot(x=ftr_top, y=ftr_top.index)
    plt.title(f"{title}_{type}")
    plt.show()



## tree 모델 학습 함수 - HP Tuning 진행 X
def tree_excution(df, title, do_cv = True, **params): # scale = "MinMax"
    print(f"[ {title} ]")

    # 데이터 준비
    X_valid, X_test, y_valid, y_test = prepare_data(df)

    obj = title.split()[0]

    # 케이스별 모델 학습
    if obj == "RandomForest" :
        # entropy
        model = RandomForestClassifier(criterion="entropy", **params)
    elif obj == "LGBM":
        model = LGBMClassifier(**params)
    elif obj == "XGBoost":
        model = XGBClassifier(**params)

    # 교차검증
    if do_cv:
        for i, (train_index, test_index) in enumerate(skf.split(X_valid, y_valid)):
            X_train = X_valid.iloc[train_index]
            y_train = y_valid.iloc[train_index]

            model.fit(X_train, y_train)

    else: model.fit(X_valid, y_valid)

    # 모델 학습 결과
    y_valid_pred = model.predict(X_valid)
    y_pred = model.predict(X_test)

    print("- Train ----")
    train_acc = accuracy_score(y_valid, y_valid_pred)
    tr_precision = precision_score(y_valid, y_valid_pred)
    tr_recall = recall_score(y_valid, y_valid_pred)
    tr_f1 = f1_score(y_valid, y_valid_pred)

    print(f"Train Accuracy: {train_acc:.3f}") # 정확도
    print(f"Precision: {tr_precision:.3f}") # 정밀도
    print(f"Recall: {tr_recall:.3f}") # 재현율
    print(f"F1-score: {tr_f1:.3f}") # F1 스코어

    # confusion-matrix plot
    visualization_confusion_matrix(y_valid, y_valid_pred, title + "_valid")

    print("\n- Test ----")
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy: {test_acc:.3f}") # 정확도
    print(f"Precision: {precision:.3f}") # 정밀도
    print(f"Recall: {recall:.3f}") # 재현율
    print(f"F1-score: {f1:.3f}") # F1 스코어
    
    visualization_confusion_matrix(y_test, y_pred, title + "_test")

    # feature importrance plot (XGBoost, LGBM인 경우 두 가지 모두 실행)
    visualization_feature_importance(model, df.drop(["취업여부"], axis=1).columns, title, fi_type[obj][0])
    if obj == "LGBM" or obj == "XGBoost" :
        visualization_feature_importance(model, df.drop(["취업여부"], axis=1).columns, title, fi_type[obj][1])

    return model


## tree 모델 학습 함수 - optuna
# 목적 함수 (RF)
def RandomForest_objective(trial: Trial, X, y, do_cv = True, **params):
    # 하이퍼파라미터 설정 (각자 튜닝하려는 것에 맞춰 수정해 주어야 한다!!!!!!!
    param = {
        # "n_estimators" : trial.suggest_int('n_estimators', params["n_estimators"][0], params["n_estimators"][1]),
        "max_depth" : trial.suggest_int('max_depth', params["max_depth"][0], params["max_depth"][1], log=True),
        # "min_samples_split" : trial.suggest_float('min_samples_split', params["min_samples_split"][0], params["min_samples_split"][1]),
        # "min_samples_leaf" : trial.suggest_float('min_samples_leaf', params["min_samples_leaf"][0], params["min_samples_leaf"][1])
    }

    # 모델 생성
    model = RandomForestClassifier(criterion="entropy", **param)

    # 교차검증 진행 여부에 따라 다르게 반환
    if do_cv:
        # K-Fold 교차검증 수행
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        return scores.mean() # 목적함수 값은 교차검증 정확도의 평균
    else:
        model.fit(X, y)
        return model.score(X, y)

# 목적 함수 (LGBM)
def LGBM_objective(trial: Trial, X, y, do_cv = True, **params):
    # 하이퍼파라미터 설정 (각자 튜닝하려는 것에 맞춰 수정해 주어야 한다!!!!!!!
    param = {
        "boosting_type" : params["boosting_type"],
        "max_depth" : trial.suggest_int('max_depth', params["max_depth"][0], params["max_depth"][1], log=True),
        "num_leaves" : trial.suggest_int('num_leaves', params["num_leaves"][0], params["num_leaves"][1]),
        "n_estimators" : trial.suggest_int('n_estimators', params["n_estimators"][0], params["n_estimators"][1]),
        "learning_rate" : trial.suggest_float('learning_rate', params["learning_rate"][0], params["learning_rate"][1]),
        "min_child_samples" : trial.suggest_int('min_child_samples', params["min_child_samples"][0], params["min_child_samples"][1]),
        "min_child_weight" : trial.suggest_float('min_child_weight', params["min_child_weight"][0], params["min_child_weight"][1]),
        "subsample" : trial.suggest_float('subsample', params["subsample"][0], params["subsample"][1]),
        }

    # 모델 생성
    model = LGBMClassifier(**param)

    # 교차검증 진행 여부에 따라 다르게 반환
    if do_cv:
        # K-Fold 교차검증 수행
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        return scores.mean() # 목적함수 값은 교차검증 정확도의 평균
    else:
        model.fit(X, y)
        return model.score(X, y)

# 목적 함수 (XGBoost)
def XGBoost_objective(trial: Trial, X, y, do_cv = True, **params):
    # 하이퍼파라미터 설정 (각자 튜닝하려는 것에 맞춰 수정해 주어야 한다!!!!!!!
    param = {
        "max_depth" : trial.suggest_int('max_depth', params["max_depth"][0], params["max_depth"][1], log=True),
        "min_child_weight" : trial.suggest_int('min_child_weight', params["min_child_weight"][0], params["min_child_weight"][1]),
        "gamma" : trial.suggest_float('gamma', params["gamma"][0], params["gamma"][1]),
    }

    # 모델 생성
    model = XGBClassifier(**param)

    # 교차검증 진행 여부에 따라 다르게 반환
    if do_cv:
        # K-Fold 교차검증 수행
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        return scores.mean() # 목적함수 값은 교차검증 정확도의 평균
    else:
        model.fit(X, y)
        return model.score(X, y)

# optuna 실행 함수
def tree_optuna_excution(df, title, do_cv = True, **params): # scale = "MinMax"
    print(f"[ {title}]")
    
    # 데이터 준비
    X_valid, X_test, y_valid, y_test = prepare_data(df)

    # Optuna 최적화
    study = optuna.create_study(direction='maximize')

    obj = title.split()[0]
    
    # 케이스별 모델 학습
    if obj == "RandomForest" :
        model = RandomForestClassifier
        objective = RandomForest_objective
    elif obj == "LGBM":
        model = LGBMClassifier
        objective = LGBM_objective
    elif obj == "XGBoost":
        model = XGBClassifier
        objective = XGBoost_objective


    study.optimize(lambda trial : objective(trial, X_valid, y_valid, do_cv, **params), n_trials = 200)

    # 최적의 하이퍼파라미터 출력
    print(f"Best trial: {study.best_trial}")
    print(f"Value: {study.best_trial.value:.4f}")
    best_params = study.best_params
    print("Best Params:")
    for k, v in best_params.items():
        print(f"{k} : {v}")

    # 최적의 하이퍼파라미터로 최종 모델 학습 (+ 교차검증까지)
    final_model = model(**best_params)

    # 교차검증
    if do_cv:
        for train_index, test_index in skf.split(X_valid, y_valid):
            X_train = X_valid.iloc[train_index]
            y_train = y_valid.iloc[train_index]

            final_model.fit(X_train, y_train)
    else: final_model.fit(X_valid, y_valid)


    # 모델 학습 결과
    y_valid_pred = final_model.predict(X_valid)
    y_pred = final_model.predict(X_test)

    print("- Train ----")
    train_acc = accuracy_score(y_valid, y_valid_pred)
    tr_precision = precision_score(y_valid, y_valid_pred)
    tr_recall = recall_score(y_valid, y_valid_pred)
    tr_f1 = f1_score(y_valid, y_valid_pred)

    print(f"Train Accuracy: {train_acc:.3f}") # 정확도
    print(f"Precision: {tr_precision:.3f}") # 정밀도
    print(f"Recall: {tr_recall:.3f}") # 재현율
    print(f"F1-score: {tr_f1:.3f}") # F1 스코어

    # confusion-matrix plot
    visualization_confusion_matrix(y_valid, y_valid_pred, title + "_valid")

    print("\n- Test ----")
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy: {test_acc:.3f}") # 정확도
    print(f"Precision: {precision:.3f}") # 정밀도
    print(f"Recall: {recall:.3f}") # 재현율
    print(f"F1-score: {f1:.3f}") # F1 스코어
    
    visualization_confusion_matrix(y_test, y_pred, title + "_test")

    # feature importrance plot (XGBoost, LGBM인 경우 두 가지 모두 실행)
    visualization_feature_importance(final_model, df.drop(["취업여부"], axis=1).columns, title, fi_type[obj][0])
    if obj == "LGBM" or obj == "XGBoost" :
        visualization_feature_importance(final_model, df.drop(["취업여부"], axis=1).columns, title, fi_type[obj][1])


    return final_model



## tree 모델 학습 함수 - GridSearchCV
def tree_gridsearchcv_excution(df, title, **param_grid): # scale = "MinMax"
    print(f"[ {title} ]")
    
    # 데이터 준비
    X_valid, X_test, y_valid, y_test = prepare_data(df)

    obj = title.split()[0]

    # 케이스별 모델 학습
    if obj == "RandomForest" :
        model = RandomForestClassifier(criterion="entropy")
    elif obj == "LGBM":
        model = LGBMClassifier()
    elif obj == "XGBoost":
        model = XGBClassifier()

    # GridSearchCV 생성
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        scoring=scoring,
        cv = skf,
        error_score='raise'   # 모델 학습에 실패한 경우 interrupt
    )
    # 그리드 서치를 사용하여 최적의 모델 훈련
    grid_search.fit(X_valid, y_valid)

    # 최적의 하이퍼파라미터 출력
    
    best_params = grid_search.best_params_
    print(f"Best Params: {best_params}")

    # 최적의 모델 얻기
    best_model = grid_search.best_estimator_

    # 모델 학습 결과
    y_valid_pred = best_model.predict(X_valid)
    y_pred = best_model.predict(X_test)

    print("- Train ----")
    train_acc = accuracy_score(y_valid, y_valid_pred)
    tr_precision = precision_score(y_valid, y_valid_pred)
    tr_recall = recall_score(y_valid, y_valid_pred)
    tr_f1 = f1_score(y_valid, y_valid_pred)

    print(f"Train Accuracy: {train_acc:.3f}") # 정확도
    print(f"Precision: {tr_precision:.3f}") # 정밀도
    print(f"Recall: {tr_recall:.3f}") # 재현율
    print(f"F1-score: {tr_f1:.3f}") # F1 스코어

    # confusion-matrix plot
    visualization_confusion_matrix(y_valid, y_valid_pred, title + "_valid")

    print("\n- Test ----")
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy: {test_acc:.3f}") # 정확도
    print(f"Precision: {precision:.3f}") # 정밀도
    print(f"Recall: {recall:.3f}") # 재현율
    print(f"F1-score: {f1:.3f}") # F1 스코어
    
    visualization_confusion_matrix(y_test, y_pred, title + "_test")

    # feature importrance plot (XGBoost, LGBM인 경우 두 가지 모두 실행)
    visualization_feature_importance(best_model, df.drop(["취업여부"], axis=1).columns, title, fi_type[obj][0])
    if obj == "LGBM" or obj == "XGBoost" :
        visualization_feature_importance(best_model, df.drop(["취업여부"], axis=1).columns, title, fi_type[obj][1])

    return best_model