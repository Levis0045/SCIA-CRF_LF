import argparse
from datetime import datetime
from pprint import pprint
import warnings

import joblib
import xgboost as xgb

import mlflow
import mlflow.xgboost

import pandas as pd
from pathlib import Path
import random

from sangkak_estimators import SangkakPosProjetReader, SangkakPosFeaturisation
from utils import fetch_logged_data

mlflow.set_tracking_uri("file:///media/elvis/Seagate Expansion Drive/Sangkak-challenge/mlruns")


def string_int_transform(r):
    mybytes = r.encode('utf-8')
    myint = int.from_bytes(mybytes, 'little')
    return myint

all_ref_data, all_labels = {}, {}

def apply_ref_transform(r):
    global all_ref_data
    if r not in all_ref_data: 
        ref = random.randint(-700000, -100)
        all_ref_data[r] = ref
        return ref
    else: return all_ref_data[r]

def apply_label_transform(r):
    global all_labels
    if r not in all_labels: 
        ref = random.randint(0, 15)
        all_labels[r] = ref
        return ref
    else: return all_labels[r]


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost Sangkak")
    parser.add_argument(
        "--lang",
        type=str,
        default="bbj",
        help="language of training data",
    )
    parser.add_argument(
        "--augment",
        default=False,
        action='store_true',
        help="Augment training data",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="(with only categorical features + shuffle) + n_estimator 10060 + lr=0.1",
        help="description of experiment",
    )
    parser.add_argument(
        "--shuffle",
        action='store_true',
        default=False,
        help="shuffle training data",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=10,
        help="number of early_stopping round",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="subsample ratio of columns when constructing each tree (default: 1.0)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of the training instances (default: 1.0)",
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0,
        help="L1 regularization (default: 0)",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.8,
        help="L2 regularization (default: 1.8)",
    )    
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=10060,
        help="n_estimators (default: 10060)",
    )    
    parser.add_argument(
        "--nthread",
        type=int,
        default=5,
        help="nthread (default: 1.8)",
    )    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="n_jobs (default: 10060)",
    )    
    return parser.parse_args()

def main():
    args = parse_args()

    # enable auto logging
    # this includes xgboost.sklearn estimators
    mlflow.xgboost.autolog()

    # Get path of test data 
    bbj_pos_path   = Path(f'./data_source/masakhane-pos/data/{args.lang}')
    train_data_path = bbj_pos_path / 'train.txt'
    dev_data_path = bbj_pos_path / 'dev.txt'
    test_data_path = bbj_pos_path / 'test.txt'

    # read data from source with sklearn estimator
    reader_estimator = SangkakPosProjetReader()
    list_train_data, _ = reader_estimator.fit(train_data_path).transform_analysis(augment=args.augment, limit=10)
    list_dev_data, _ = reader_estimator.fit(dev_data_path).transform_analysis(augment=args.augment, limit=10)
    list_test_data, _ = reader_estimator.fit(test_data_path).transform_analysis(augment=args.augment, limit=10)


    feature_estimator = SangkakPosFeaturisation()
    feature_estimator.fit([])

    Xtrain = feature_estimator.transform(list_train_data)
    Xdev  = feature_estimator.transform(list_dev_data)
    Xtest = feature_estimator.transform(list_test_data)

    ytrain = feature_estimator.transform(list_train_data, label=True)
    ydev   = feature_estimator.transform(list_dev_data, label=True)
    ytest  = feature_estimator.transform(list_test_data, label=True)

    features_types = {
        "categorical_features": [
            'word.letters',
            'word',
            'word.tones',
            'word.normalized',
            'word.lower()',
            'word.prefix',
            'word.root',
            '-1:word',
            #'-1:word.tag',
            '-1:word.normalized',
            '-1:word.lower()',
            '-1:word.prefix',
            '-1:word.root',
            '-1:word.letters',
            '+1:word.letters',
            '+1:word.prefix',
            '+1:word.root',
            '+1:word',
            '+1:word.lower()',
            #'+1:word.tag',
            '+1:word.normalized'
        ],
        "numerical_features": [
            'bias',
            'word.position',
            'word.have_tone',
            'word.ispunctuation',
            'word.isdigit()',
            'word.EOS',
            'word.BOS',
            'word.start_with_capital',
            'word.has_hyphen',
            '-1:word.position',
            '-1:word.start_with_capital',
            '-1:len(word-1)',
            '+1:word.position',
            '+1:word.start_with_capital',
            '+1:len(word+1)',
            '+1:word.isdigit()',
            '+1:word.ispunctuation',
            '+1:word.BOS',
            '+1:word.EOS',
            '-1:word.isdigit()',
            '-1:word.ispunctuation',
            '-1:word.BOS',
            '-1:word.EOS'
        ]
    }

    source_path = "./experimentations/preprocessing"
    project = f'{source_path}/sangkak_input_aug_{args.augment}_xgb_df_data_{args.lang}.joblib'
    if Path(project).exists():
        with open(project, 'rb') as f:
            data = joblib.load(f)
            xgb_df_train = data['xgb_df_train'] 
            xgb_df_dev  = data['xgb_df_dev'] 
            xgb_df_test = data['xgb_df_test'] 
    else:
        xgb_df_train = feature_estimator.transform_to_sagemaker_format(
            Xtrain, ytrain,
            normalize=features_types
        )
        xgb_df_dev = feature_estimator.transform_to_sagemaker_format(
            Xdev, ydev, 
            label='dev',
            normalize=features_types
        )
        xgb_df_test = feature_estimator.transform_to_sagemaker_format(
            Xtest, ytest, 
            label='test',
            normalize=features_types
        )
        joblib.dump({
            "xgb_df_train": xgb_df_train, 
            "xgb_df_dev": xgb_df_dev,
            "xgb_df_test": xgb_df_test
        }, project)


    all_data = pd.concat([xgb_df_train, xgb_df_dev, xgb_df_test], 
                        axis=0, ignore_index=True)

    i, all_labels_ = 0, {}
    for x in all_data['labels']:
        if x not in all_labels:
            all_labels[x] = i
            all_labels_[i] = x
            i += 1

    all_data_parse = all_data.copy()

    all_data_parse['labels'] = all_data['labels'].map(all_labels).astype("int")
    all_data_parse['word'] = all_data['word'].apply(apply_ref_transform).astype("int")
    all_data_parse['word.tones'] = all_data['word.tones'].apply(apply_ref_transform).astype("int")
    all_data_parse['word.normalized'] = all_data['word.normalized'].apply(apply_ref_transform).astype("int")
    all_data_parse['word.lower()'] = all_data['word.lower()'].apply(apply_ref_transform).astype("int")
    all_data_parse['word.prefix'] = all_data['word.prefix'].apply(apply_ref_transform).astype("int")
    all_data_parse['word.root'] = all_data['word.root'].apply(apply_ref_transform).astype("int")
    all_data_parse['-1:word'] = all_data['-1:word'].apply(apply_ref_transform).astype("int")
    #all_data_parse['-1:word.tag'] = all_data['-1:word.tag'].apply(apply_ref_transform).astype("int")
    all_data_parse['-1:word.normalized'] = all_data['-1:word.normalized'].apply(apply_ref_transform).astype("int")
    all_data_parse['-1:word.lower()'] = all_data['-1:word.lower()'].apply(apply_ref_transform).astype("int")
    all_data_parse['+1:word'] = all_data['+1:word'].apply(apply_ref_transform).astype("int")
    all_data_parse['+1:word.lower()'] = all_data['+1:word.lower()'].apply(apply_ref_transform).astype("int")
    #all_data_parse['+1:word.tag'] = all_data['+1:word.tag'].apply(apply_ref_transform).astype("int")
    all_data_parse['+1:word.normalized'] = all_data['+1:word.normalized'].apply(apply_ref_transform).astype("int")
    all_data_parse['+1:word.prefix'] = all_data['+1:word.prefix'].apply(apply_ref_transform).astype("int")
    all_data_parse['+1:word.root'] = all_data['+1:word.root'].apply(apply_ref_transform).astype("int")
    all_data_parse['-1:word.prefix'] = all_data['-1:word.prefix'].apply(apply_ref_transform).astype("int")
    all_data_parse['-1:word.root'] = all_data['-1:word.root'].apply(apply_ref_transform).astype("int")
    all_data_parse['+1:word.letters'] = all_data['+1:word.letters'].apply(apply_ref_transform).astype("int")
    all_data_parse['-1:word.letters'] = all_data['-1:word.letters'].apply(apply_ref_transform).astype("int")
    all_data_parse['word.letters'] = all_data['word.letters'].apply(apply_ref_transform).astype("int")

    # remove unused / non performants variables
    remove_unused_features = ['+1:word.isdigit()', '+1:word.ispunctuation', '-1:word.EOS',
            '+1:word.BOS', 'word.has_hyphen', '+1:word.EOS', '-1:word.BOS',
            '+1:word.EOS', '-1:word.isdigit()', '+1:word.BOS', 
            '-1:word.ispunctuation', '-1:word.BOS', '+1:word.normalized',
            '-1:word.EOS', '-1:word.tag', '+1:word.tag', 
            '-1:word.start_with_capital','+1:word.start_with_capital']
    for x in remove_unused_features:
        try: del all_data_parse[x]
        except: print("-- fail to removed: %s" %x)

    datasets = mlflow.data.from_pandas(all_data_parse, source=str(bbj_pos_path), 
                                        targets="labels", name="Masakhane POS Datasets")
    mlflow.log_input(datasets, context="xgboost.training")

    from sklearn.model_selection import train_test_split
    from sklearn.utils.multiclass import type_of_target

    xgb_X_train, xgb_X_test, xgb_y_train, xgb_y_test = train_test_split(
        all_data_parse.drop('labels', axis=1).copy(),
        all_data_parse['labels'].copy(),
        test_size=0.2, random_state=None, shuffle=args.shuffle
    )

    xgb_X_train, xgb_X_dev, xgb_y_train, xgb_y_dev = train_test_split(
        xgb_X_train, xgb_y_train, test_size=0.25, 
        random_state=None, shuffle=args.shuffle
    )

    num_class = len(list(set(all_data_parse['labels'])))
    print("Number of classes: %s" %num_class)

    print("Type of target of ytrain data set: %s" %type_of_target(xgb_y_train))
    print("Type of target of ytest data set: %s\n" %type_of_target(xgb_y_test))

    len_data = len(all_data_parse.index)

    def f_len(data):
        l = len(data)
        percent = l*100/len_data
        return {'l':l, 'p':int(percent)}

    print("- len of Xtrain data set: {l} ({p}%)".format(**f_len(xgb_X_train)))
    print("- len of Xtest data set: {l} ({p}%)".format(**f_len(xgb_X_test)))
    print("- len of Xdev data set: {l} ({p}%)".format(**f_len(xgb_X_dev)))
    print("len of ytrain data set: {l} ({p}%)".format(**f_len(xgb_y_train)))
    print("len of ytest data set: {l} ({p}%)".format(**f_len(xgb_y_test)))
    print("len of ydev data set: {l} ({p}%)".format(**f_len(xgb_y_dev)))

   
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from sklearn.utils.class_weight import compute_sample_weight
    #from mlflow.models import infer_signature

    params = {
        'objective':'multi:softmax', 
        'num_class': num_class,
        'early_stopping_rounds':args.early_stop, # use only for XGBClassifier
        'eval_metric':['merror','mlogloss'], 
        'seed':42, 'verbosity':1, 
        'base_score':None, #The initial prediction score of all instances, global bias
        'booster':'gbtree',
        'gamma':0, 
        'importance_type':'weight', 
        'learning_rate': args.learning_rate, 
        # max_bin:250, # Increasing this number improves the optimality of splits at the cost of higher computation time.
        'max_delta_step':1.0, # Imbalanced: predicting the right probability
        'max_depth':0, # non limit in number of tree :> use max RAM
        'n_estimators':args.n_estimators,
        'n_jobs': args.n_jobs, 
        'nthread': args.nthread, 
        'random_state':0, 
        # save_period:20, # The period to save the model
        'reg_alpha':0, # L1 regularization
        'reg_lambda':1.8, # L2 regularization
        'scale_pos_weight':1.5, # Imbalanced: Balance the positive and negative weights
        'silent': None, 
        # updater:"sync", # synchronizes trees in all distributed nodes.
        # grow_policy:"lossguide", # Controls a way new nodes are added to the tree.
        # subsample:0.8, # XGBoost would randomly sample half of the training data prior to growing trees
        'tree_method': "hist" # The tree construction algorithm used in XGBoost
    }
    
    experiment_name = f"POS-{args.lang}: XGboost"
    try: experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException: 
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id


    with mlflow.start_run(experiment_id=experiment_id, run_name=args.description, 
                          nested=True, description=args.description) as run:
        mlflow.set_tag("with_augmentation", args.augment)
        mlflow.set_tag("with_shuffle", args.shuffle)
        mlflow.set_tag("language", args.lang)

        warnings.filterwarnings("ignore")
        
        mlflow.log_params(params)
        mlflow.log_param("labels", list(all_labels_.values()))
        mlflow.log_param("language", args.lang)
        mlflow.log_param("data shuffle", args.shuffle)
        mlflow.log_param("data augmented", args.augment)

        #mlflow.log_input(xgb_X_train, context="training-features")
        #mlflow.log_input(xgb_y_train, context="training-labels")
        #mlflow.log_input(xgb_X_dev, context="dev-features")
        #mlflow.log_input(xgb_y_dev, context="dev-labels")        
        #mlflow.log_input(xgb_X_test, context="test-features")
        #mlflow.log_input(xgb_y_test, context="test-labels")   

        xgb_clf = xgb.XGBClassifier(
               **params
        )

        optimised_labels_weights = compute_sample_weight(
            class_weight='balanced',
            y=xgb_y_train
        )

        xgb_clf.fit(xgb_X_train, xgb_y_train,
                    verbose=1, # set to 1 to see xgb training round intermediate results
                    sample_weight=optimised_labels_weights, 
                    eval_set=[(xgb_X_train, xgb_y_train), 
                            (xgb_X_dev, xgb_y_dev)])

        #signature = infer_signature(xgb_X_train, xgb_clf.predict(xgb_X_train))
        mlflow.xgboost.log_model(xgb_clf, artifact_path="xgboost-model", 
                                 input_example=xgb_X_train,
                                 registered_model_name="xgboost-classifier-model")

        # Evaluate model on test dataset
        y_pred = xgb_clf.predict(xgb_X_test)
        f1_score(xgb_y_test, y_pred, average='micro')
        accuracy_score(xgb_y_test, y_pred)
        
        # training validations metrics
        results_loss = xgb_clf.evals_result()
        for i, v in enumerate(results_loss['validation_0']['mlogloss']):
            mlflow.log_metric("validation_train_mlogloss", value=v, step=i)
        for i, v in enumerate(results_loss['validation_1']['mlogloss']):
            mlflow.log_metric("validation_test_mlogloss", value=v, step=i)
        for i, v in enumerate(results_loss['validation_0']['merror']):
            mlflow.log_metric("validation_train_merror", value=v, step=i)
        for i, v in enumerate(results_loss['validation_1']['merror']):
            mlflow.log_metric("validation_test_merror", value=v, step=i)

        from sklearn.metrics import classification_report
        #from mlflow.models import MetricThreshold

        all_labels__  = all_labels_.copy()
        sorted_labels = sorted(all_labels_, key=lambda name: (name, all_labels_[name]))
        cl_report = classification_report(xgb_y_test, y_pred, labels=sorted_labels, 
                                       digits=3, zero_division=False, output_dict=True)

        mlflow.log_param("labels_items", all_labels__)

        mlflow.log_table(data=cl_report, artifact_file="crf_classification_report.json")

        for l, m in cl_report.items():
            try: l = all_labels__[int(l)]
            except: pass

            if type(m) == float:
                mlflow.log_metric(l, m)
            else:
                mlflow.log_metric(f"{l}_precision", m['precision'])
                mlflow.log_metric(f"{l}_recall", m['recall'])
                mlflow.log_metric(f"{l}_f1-score", m['f1-score'])
                mlflow.log_metric(f"{l}_support", m['support'])
        
        
        model_uri = mlflow.get_artifact_uri("xgboost-model")
        eval_data = xgb_X_test.copy()
        eval_data['label'] = y_pred

        """
        thresholds = {
            "accuracy_score": MetricThreshold(
                threshold=0.8,  # accuracy should be >=0.8
                min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
                min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
                greater_is_better=True,
            ),
        }
        """

        # Evaluate the logged model
        result = mlflow.evaluate(
            model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
            #validation_thresholds=thresholds,
        )

        print(f"metrics:\n{result.metrics}")
        print(f"artifacts:\n{result.artifacts}")

        run_id = mlflow.last_active_run().info.run_id
        print(f"\nLogged data and model in run {run_id}")

        # show logged data
        for key, data in fetch_logged_data(run_id).items():
            print(f"\n---------- logged {key} ----------")
            pprint(data)


if __name__ == "__main__":
    main()