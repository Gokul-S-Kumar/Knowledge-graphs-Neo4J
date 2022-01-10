import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, roc_curve, auc, plot_roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

if __name__ == '__main__':
    df = pd.read_csv('./Data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    le = LabelEncoder()
    le_count = 0
    for col in df.columns[1:]:
        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1
    print('{} columns were label encoded'.format(le_count))
    df = pd.get_dummies(df, drop_first = True)
    target = np.array(df.Attrition)
    df = df.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis = 1)
    features = np.array(df)
    x_train, x_test, y_train, y_test = train_test_split(features, target, random_state = 42, test_size = 0.25, stratify = target)
    models = []
    models.append(('logistic_regression', LogisticRegression(solver = 'liblinear', random_state = 42)))
    models.append(('random_forest', RandomForestClassifier(random_state = 42)))
    models.append(('svm', SVC(random_state = 42, probability = True)))
    models.append(('knn', KNeighborsClassifier()))
    models.append(('decision_tree_classifier', DecisionTreeClassifier(random_state = 42)))
    models.append(('gaussian_nb', GaussianNB()))
    models.append(('xgboost', XGBClassifier(use_label_encoder = False, random_state = 42, eval_metric = 'error', verbosity = 0)))
    models.append(('lightgbm', LGBMClassifier(random_state = 42)))
    models.append(('catboost', CatBoostClassifier(random_state = 42, verbose = False)))
    col = ['Algorithm', 'ROC AUC mean', 'ROC AUC std']
    df_results = pd.DataFrame(columns = col)
    i = 0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize = [10, 10])
    cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
    for name, model in models:
        tprs = []
        aucs = []
        for train, valid in cv.split(x_train, y_train):
            pred_proba = model.fit(x_train[train], y_train[train]).predict_proba(x_train[valid])
            fpr, tpr, t = roc_curve(y_train[valid], pred_proba[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        mean_tpr = np.mean(tprs, axis = 0)
        cv_auc_mean = round(np.array(aucs).mean() * 100, 2)
        cv_auc_std = round(np.array(aucs).std() * 100, 2)
        df_results.loc[i] = [name, cv_auc_mean, cv_auc_std]
        plt.plot(mean_fpr, mean_tpr, label = 'Mean AUC of {0} = {1}'.format(name, cv_auc_mean), lw = 2)
        i += 1
        print('Finished {}'.format(name))
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    plt.legend()
    plt.savefig('./outputs/roc_curve.jpg')
    plt.show()
    df_results = df_results.sort_values(by = ['ROC AUC mean'], ascending = False).reset_index(drop = True)    
    print('The AUC scores for different models are \n', df_results)
    df_results.to_csv('./outputs/AUC_results.csv')
    
    # The parameters being used below are obtained from GridSearch using KFold cross-validation.
    fin_model = CatBoostClassifier(depth = 4, iterations = 700, learning_rate = 0.01, random_state = 42, verbose = False)
    fin_model.fit(x_train, y_train)
    pred_proba = fin_model.predict_proba(x_test)
    plot_confusion_matrix(fin_model, x_test, y_test, cmap = 'Blues')
    plt.savefig('./outputs/fin_confusion_matrix.jpg')
    plot_roc_curve(fin_model, x_test, y_test)
    plt.savefig('./outputs/fin_roc_curve.jpg')
    plt.show()
    print('The ROC-AUC test score of the final model is: {}'.format(round(roc_auc_score(y_test, pred_proba[:, 1]) * 100, 2))
    fin_model.save_model('./outputs/Final_model_dump')