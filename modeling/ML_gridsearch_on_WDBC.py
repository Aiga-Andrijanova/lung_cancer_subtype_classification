import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, VarianceThreshold
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, GridSearchCV

from ucimlrepo import fetch_ucirepo
from boruta import BorutaPy
from tqdm.auto import tqdm

param_grids = {
    'logistic_regression': {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.05, 0.5, 1.0, 2.5, 5, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [50, 100, 150, 300, 500, 1000]
    },
    'decision_tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 6, 8, 10]
    },
    'random_forest': {
        'n_estimators': [100, 300],
        'criterion': ['entropy'],
        'max_depth': [5, 10],
        'min_samples_split': [10, 15],
        'max_features': [None]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform'],
        'algorithm': ['auto']
    },
    'svm-linear': {
        'C': [0.5, 1.0, 2.5, 5, 10]
    },
    'svm-poly': {
        'C': [0.5, 1.0, 2.5, 5, 10, 15],
        'degree': [2, 3, 4]
    },
    'svm-rbf': {
        'C': [0.5, 1.0, 2.5, 5, 10],
        'gamma': ['scale', 'auto', 0.05, 0.1, 1]
    },
    'xgboost': {
        'n_estimators': [100],
        'max_depth': [6],
        'learning_rate': [0.1],
        'subsample': [0.75],
        'colsample_bytree': [0.5],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 1, 10]
    },
    'adaboost': {
        'n_estimators': [250, 300, 350, 500],
        'learning_rate': [1.0],
        'algorithm': ['SAMME'],
        'estimator': [DecisionTreeClassifier(max_depth=3),
                      DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2)]
    },
    'naive_bayes': {
        'var_smoothing': [1e-11, 1e-10, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    },
    'gaussian_process': {
        'max_iter_predict': [15, 30, 50, 100, 150]
    },
    'lda':{
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto']
    }
}

def create_param_grid(fs_name, dr_name, model_name):
    grid = {}
    
    # Feature selection parameters
    if fs_name == 'rfe':
        grid.update({
            'feature_selection__n_features_to_select': [5, 10, 15],
            'feature_selection__step': [1, 2]
        })
    elif fs_name == 'boruta':
        grid.update({
            'feature_selection__n_estimators': ['auto', 100, 200, 300],
            'feature_selection__max_iter': [50, 100, 150]
        })
    elif fs_name == 'variance_threshold':
        grid.update({
            'feature_selection__threshold': [0.0, 0.05, 0.1, 0.15, 0.2]
        })
    elif fs_name == 'pearson' or fs_name=='mutual_info' or fs_name == 'chi2':
        grid.update({
            'k': [5, 10, 15],
        })

    # Dimensionality reduction parameters
    if dr_name == 'pca':
        grid.update({
            'dim_reduction__n_components': [2, 5, 10, 15]
        })

    # Model parameters
    grid.update({f'classifier__{k}': v for k, v in param_grids[model_name].items()})
    
    return grid

def create_pipeline(fs_method, dr_method, model):
    steps = [('preprocessor', preprocessor)]
    if fs_method:
        steps.append(('feature_selection', fs_method))
    if dr_method:
        steps.append(('dim_reduction', dr_method))
    steps.append(('classifier', model))  # Note: changed 'model' to 'classifier'
    return Pipeline(steps)

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

# Define features and target
y = (y['Diagnosis'].isin(['M'])).astype(int)  # 1 for M, 0 for B

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), list(X.columns.values)),
    ])


# Define feature selection methods
feature_selection_methods = {
    'none': None,
    'lasso': SelectFromModel(Lasso(alpha=0.1)),
    'elastic_net': SelectFromModel(ElasticNet(alpha=0.1, l1_ratio=0.5)),
    'pearson': SelectKBest(score_func=f_classif, k=15),
    'mutual_info': SelectKBest(score_func=mutual_info_classif, k=15, ),
    'rfe': RFE(estimator=SVC(C=0.1, kernel="linear")),
    'boruta': BorutaPy(SVC(C=0.1, kernel="linear"), verbose=0),
    'chi2': SelectKBest(score_func=chi2),
    'variance_threshold': VarianceThreshold(),
}

# Define dimensionality reduction methods
dim_reduction_methods = {
    'none': None,
    'pca': PCA()
}

# Define models
models = {
    'logistic_regression': LogisticRegression(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(n_jobs=-1),
    'knn': KNeighborsClassifier(n_jobs=-1),
    'svm-linear': SVC(probability=True, kernel='linear'),
    'svm-poly': SVC(probability=True, kernel='poly'),
    'svm-rbf': SVC(probability=True, kernel='rbf'),
    'naive_bayes': GaussianNB(),
    'adaboost': AdaBoostClassifier(algorithm='SAMME'),
    'gaussian_process': GaussianProcessClassifier(n_restarts_optimizer=10, n_jobs=-1),
    'lda': LinearDiscriminantAnalysis(),
    'xgboost': xgb.XGBClassifier(n_jobs=24, objective='binary:logistic', eval_metric='logloss')
}

# Function to evaluate model
def evaluate_model(y_true, y_pred, y_pred_proba):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': specificity,
        'f1': f1_score(y_true, y_pred),
        'roc-auc': roc_auc_score(y_true, y_pred_proba),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

results = []

save_name = 'WDBC_gridsearch_results.csv'

total_iterations = len(feature_selection_methods) * len(dim_reduction_methods) * len(models)

with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
    for fs_name, fs_method in feature_selection_methods.items():
        for dr_name, dr_method in dim_reduction_methods.items():
            for model_name, model in models.items():

                pbar.set_description(f"FS: {fs_name}, DR: {dr_name}, Model: {model_name}")

                pipeline = create_pipeline(fs_method, dr_method, model)
                param_grid = create_param_grid(fs_name, dr_name, model_name)
                
                grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=3)
                
                try:
                    grid_search.fit(X, y)
                    
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    
                    # Cross-validation with best model
                    cv_pred = cross_val_predict(best_model, X, y, cv=10)
                    cv_pred_proba = cross_val_predict(best_model, X, y, cv=10, method='predict_proba')[:, 1]
                    cv_metrics = evaluate_model(y, cv_pred, cv_pred_proba)
                    
                    results.append({
                        'feature_selection': fs_name,
                        'dim_reduction': dr_name,
                        'model': model_name,
                        'best_params': best_params,
                        'best_score': best_score,
                        **cv_metrics
                    })

                    pbar.set_postfix({'Best Score': best_score})
                    
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(save_name, index=False)

                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error with {model_name}, {fs_name}, {dr_name}: {str(e)}")
                    pbar.update(1)
                    continue