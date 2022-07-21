from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, roc_auc_score, roc_curve, log_loss, classification_report


def LogisticRegressionModel(x_train, y_train, x_val, y_val):
    LogisticRegressionModel = make_pipeline(LogisticRegression(max_iter=1000))
    LogisticRegressionModel.fit(x_train, y_train)

    metrics_LogisticRegressionModel = {
        'Mean_ACC_LogisticRegressionModel': cross_val_score(LogisticRegressionModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
        'Mean_AUC_LogisticRegressionModel': cross_val_score(LogisticRegressionModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
        'Mean_Precision_LogisticRegressionModel': cross_val_score(LogisticRegressionModel, x_val, y_val, cv=5, scoring='precision').mean(),
        'Mean_Recall_LogisticRegressionModel': cross_val_score(LogisticRegressionModel, x_val, y_val, cv=5, scoring='recall').mean()
    }

    return metrics_LogisticRegressionModel


def DecisionTreeModel(x_train, y_train, x_val, y_val):

    DecisionTreeClassifierModel = make_pipeline(
        DecisionTreeClassifier(random_state=42))

    DecisionTreeClassifierModel.fit(x_train, y_train)

    metrics_DecisionTreeClassifierModel = {'modele': 'dtc',
                                           'Mean_ACC_DecisionTreeClassifierModel': cross_val_score(DecisionTreeClassifierModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
                                           'Mean_AUC_DecisionTreeClassifierModel': cross_val_score(DecisionTreeClassifierModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
                                           'Mean_Precision_DecisionTreeClassifierModel': cross_val_score(DecisionTreeClassifierModel, x_val, y_val, cv=5, scoring='precision').mean(),
                                           'Mean_Recall_DecisionTreeClassifierModel': cross_val_score(DecisionTreeClassifierModel, x_val, y_val, cv=5, scoring='recall').mean()
                                           }
    return metrics_DecisionTreeClassifierModel


def KNeighborsClassifierModel(x_train, y_train, x_val, y_val):
    KNeighborsClassifierModel = make_pipeline(
        KNeighborsClassifier(n_neighbors=7))

    KNeighborsClassifierModel.fit(x_train, y_train)

    metrics_KNeighborsClassifierModel = {'modele': 'KNeighborsClassifierModel',
                                         'Mean_ACC_KNeighborsClassifierModel': cross_val_score(KNeighborsClassifierModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
                                         'Mean_AUC_KNeighborsClassifierModel': cross_val_score(KNeighborsClassifierModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
                                         'Mean_Precision_KNeighborsClassifierModel': cross_val_score(KNeighborsClassifierModel, x_val, y_val, cv=5, scoring='precision').mean(),
                                         'Mean_Recall_KNeighborsClassifierModel': cross_val_score(KNeighborsClassifierModel, x_val, y_val, cv=5, scoring='recall').mean()
                                         }
    return metrics_KNeighborsClassifierModel


def RandomForestClassifierModel(x_train, y_train, x_val, y_val):

    RandomForestClassifierModel = make_pipeline(
        RandomForestClassifier(max_depth=17, random_state=42))

    RandomForestClassifierModel.fit(x_train, y_train)

    metrics_RandomForestClassifierModel = {'modele': 'RandomForestClassifierModel',
                                           'Mean_ACC_RandomForestClassifierModel': cross_val_score(RandomForestClassifierModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
                                           'Mean_AUC_RandomForestClassifierModel': cross_val_score(RandomForestClassifierModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
                                           'Mean_Precision_RandomForestClassifierModel': cross_val_score(RandomForestClassifierModel, x_val, y_val, cv=5, scoring='precision').mean(),
                                           'Mean_Recall_RandomForestClassifierModel': cross_val_score(RandomForestClassifierModel, x_val, y_val, cv=5, scoring='recall').mean()
                                           }
    return metrics_RandomForestClassifierModel


def GaussianNBModel(x_train, y_train, x_val, y_val):
    GaussianNBModel = make_pipeline(GaussianNB())

    GaussianNBModel.fit(x_train, y_train)

    metrics_GaussianNBModel = {'modele': 'GaussianNBModel',
                               'Mean_ACC_GaussianNBModel': cross_val_score(GaussianNBModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
                               'Mean_AUC_GaussianNBModel': cross_val_score(GaussianNBModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
                               'Mean_Precision_GaussianNBModel': cross_val_score(GaussianNBModel, x_val, y_val, cv=5, scoring='precision').mean(),
                               'Mean_Recall_GaussianNBModel': cross_val_score(GaussianNBModel, x_val, y_val, cv=5, scoring='recall').mean()
                               }
    return metrics_GaussianNBModel


def AdaBoostClassifierModel(x_train, y_train, x_val, y_val):
    AdaBoostClassifierModel = make_pipeline(AdaBoostClassifier())

    AdaBoostClassifierModel.fit(x_train, y_train)

    metrics_AdaBoostClassifierModel = {'modele': 'AdaBoostClassifierModel',
                                       'Mean_ACC_AdaBoostClassifierModel': cross_val_score(AdaBoostClassifierModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
                                       'Mean_AUC_AdaBoostClassifierModel': cross_val_score(AdaBoostClassifierModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
                                       'Mean_Precision_AdaBoostClassifierModel': cross_val_score(AdaBoostClassifierModel, x_val, y_val, cv=5, scoring='precision').mean(),
                                       'Mean_Recall_AdaBoostClassifierModel': cross_val_score(AdaBoostClassifierModel, x_val, y_val, cv=5, scoring='recall').mean()
                                       }
    return metrics_AdaBoostClassifierModel


def CatBoostClassifierModel(x_train, y_train, x_val, y_val):
    CatBoostClassifierModel = make_pipeline(CatBoostClassifier(verbose=False))

    CatBoostClassifierModel.fit(x_train, y_train)

    metrics_CatBoostClassifierModel = {'modele': 'CatBoostClassifierModel',
                                       'Mean_ACC_CatBoostClassifierModel': cross_val_score(CatBoostClassifierModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
                                       'Mean_AUC_CatBoostClassifierModel': cross_val_score(CatBoostClassifierModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
                                       'Mean_Precision_CatBoostClassifierModel': cross_val_score(CatBoostClassifierModel, x_val, y_val, cv=5, scoring='precision').mean(),
                                       'Mean_Recall_CatBoostClassifierModel': cross_val_score(CatBoostClassifierModel, x_val, y_val, cv=5, scoring='recall').mean()
                                       }
    return metrics_CatBoostClassifierModel


def XGBClassifierModel(x_train, y_train, x_val, y_val):
    XGBClassifierModel = make_pipeline(XGBClassifier())

    XGBClassifierModel.fit(x_train, y_train)

    metrics_XGBClassifierModel = {'modele': 'XGBClassifierModel',
                                  'Mean_ACC_XGBClassifierModel': cross_val_score(XGBClassifierModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
                                  'Mean_AUC_XGBClassifierModel': cross_val_score(XGBClassifierModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
                                  'Mean_Precision_XGBClassifierModel': cross_val_score(XGBClassifierModel, x_val, y_val, cv=5, scoring='precision').mean(),
                                  'Mean_Recall_XGBClassifierModel': cross_val_score(XGBClassifierModel, x_val, y_val, cv=5, scoring='recall').mean()
                                  }
    return metrics_XGBClassifierModel


def LGBMClassifierModel(x_train, y_train):
    LGBMClassifierModel = make_pipeline(LGBMClassifier())
    LGBMClassifierModel.fit(x_train, y_train)

    
    return LGBMClassifierModel

def get_metrics_LGBMClassifierModel(LGBMClassifierModel,x_val,y_val):

    metrics_LGBMClassifierModel = {'modele': 'LGBMClassifierModel',
                                   'Mean_ACC_LGBMClassifierModel': cross_val_score(LGBMClassifierModel, x_val, y_val, cv=5, scoring='accuracy').mean(),
                                   'Mean_AUC_LGBMClassifierModel': cross_val_score(LGBMClassifierModel, x_val, y_val, cv=5, scoring='roc_auc').mean(),
                                   'Mean_Precision_LGBMClassifierModel': cross_val_score(LGBMClassifierModel, x_val, y_val, cv=5, scoring='precision').mean(),
                                   'Mean_Recall_LGBMClassifierModel': cross_val_score(LGBMClassifierModel, x_val, y_val, cv=5, scoring='recall').mean()
                                   }
    return metrics_LGBMClassifierModel