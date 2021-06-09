#Importing required Packages
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import umap.umap_ as umap
import lightgbm as lgbm
import sklearn.externals
import joblib
from azureml.core import Dataset, Run

#####DATA INPUT#####:

#The get context method will obtain all the information about current experiment
run = Run.get_context()
#Obtaining the workspace
ws = run.experiment.workspace
#Getting the dataset (which was registered as 'titanic_cleaned')
dataset = Dataset.get_by_name(workspace = ws, name = 'titanic_cleaned')
#Converting to pandas df
df = dataset.to_pandas_dataframe()

#####TRAIN_TEST_SPLIT#####
X = df.drop('Survived', axis = 1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

#####For lgbm, we need to define a dataset specific to LGBM#####
#Categorical Features for the dataset
cat_features = ['Alone', 'Sex', 'Pclass', 'Embarked']
#Train Dataset
train_dataset = lgbm.Dataset(data = X_train, label=y_train,
                             categorical_feature=cat_features,
                             free_raw_data=False)
#Test Dataset
test_dataset = lgbm.Dataset(data = X_test, label = y_test,
                            categorical_feature=cat_features,
                            free_raw_data=False)

#####Argument Parser######

#We can input arguments from authoring script that are defined here. They are usually model parameters
parser = argparse.ArgumentParser()
#Format:
#parser.add_argument('--arg-name-defined-in-auth-script', type = str/int/float, dest = variable name,
#default = default_value)

parser.add_argument('--boosting', type=str, dest='boosting', default='gbdt')
parser.add_argument('--num-boost-round', type=int, dest='num_boost_round', default=500)
parser.add_argument('--early-stopping-rounds', type=int, dest='early_stopping_rounds', default=200)
parser.add_argument('--drop-rate', type=float, dest='drop_rate', default=0.15)
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001)
parser.add_argument('--min-data-in-leaf', type=int, dest='min_data_in_leaf', default=20)
parser.add_argument('--feature-fraction', type=float, dest='feature_fraction', default=0.7)
parser.add_argument('--num-leaves', type=int, dest='num_leaves', default=40)

#Now we load all the above arguments in args object
args = parser.parse_args()

#We will now define LGBM parameters via a dictionary
lgbm_params = {
    'application': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': args.learning_rate,
    'boosting': args.boosting,
    'drop_rate': args.drop_rate,
    'min_data_in_leaf': args.min_data_in_leaf,
    'feature_fraction': args.feature_fraction,
    'num_leaves': args.num_leaves,
}


#####LOGGING#####

#The given function credits to Mastering Azure Machine Learning book
def azure_ml_callback(run):
    def callback(env):
        if env.evaluation_result_list:
            run.log('iteration', env.iteration + 1)
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                run.log("%s (%s)" % (eval_name, data_name), result)
    callback.order = 10
    return callback

evaluation_results = {}
clf = lgbm.train(train_set=train_dataset,
                 params=lgbm_params,
                 valid_sets=[train_dataset, test_dataset],
                 valid_names=['train', 'val'],
                 evals_result=evaluation_results,
                 num_boost_round=args.num_boost_round,
                 early_stopping_rounds=args.early_stopping_rounds,
                 verbose_eval=20,
                 callbacks = [azure_ml_callback(run)]
                )

#Metrics for testing
y_train_preds = np.round(clf.predict(X_train))
run.log("accuracy (test)", accuracy_score(y_train, y_train_preds))
run.log("ROC_AUC_SCORE", roc_auc_score(y_train, y_train_preds))
run.log("f1 (test)", f1_score(y_train, y_train_preds))

#Metrics for testing
y_test_preds = np.round(clf.predict(X_test))
run.log("accuracy (test)", accuracy_score(y_test, y_test_preds))
run.log("ROC_SUC_SCORE", roc_auc_score(y_test, y_test_preds))
run.log("f1 (test)", f1_score(y_test, y_test_preds))

#Saving the plot for Feature importance in the logs
fig, ax = plt.subplots(1, 1)
lgbm.plot_importance(clf, ax=ax)
run.log_image("feature importance", plot=fig)

#Saving and registering the model

joblib.dump(clf, './outputs/lgbm.pkl')
run.upload_file('./outputs/lgbm.pkl', './outputs/lgbm.pkl')
run.register_model(model_name='lgbm_titanic', model_path='./outputs/lgbm.pkl')
