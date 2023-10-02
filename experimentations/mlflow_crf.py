
from pprint import pprint

import argparse
import warnings
import mlflow
import argparse
from sklearn.utils.multiclass import type_of_target
from sklearn_crfsuite import metrics

from pathlib import Path
import pandas as pd
#from numpy import asarray as np_array
import json

from sangkak_estimators import SangkakPosProjetReader, SangkakPosFeaturisation
from utils import fetch_logged_data

mlflow.set_tracking_uri("file:///media/elvis/Seagate Expansion Drive/Sangkak-challenge/mlruns")

# Evaluate metrics
def eval_metrics(actual, pred, **kargs):
    f1 = metrics.flat_f1_score(actual, pred, **kargs)
    acc = metrics.flat_accuracy_score(actual, pred)
    return f1, acc

def parse_args():
    parser = argparse.ArgumentParser(description="sklearn CRF Sangkak")
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
        default="(with only categorical features + shuffle)",
        help="description of experiment",
    )
    parser.add_argument(
        "--re-organised-data",
        default=False,
        action='store_false',
        help="re-organised data of training",
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action='store_true',
        help="shuffle training data",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=100,
        help="number of iterations",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=0.0328771171605105,
        help="c2 regularization value for algorithm (default: 0.3)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default='lbfgs',
        help="crf algorithm (default: lbfgs)",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=0.0920512484757745 ,
        help="c1 regularization value for algorithm (default: 0.0920512484757745 )",
    )
    return parser.parse_args()

def main():
    # parse command-line arguments
    args = parse_args()

    # enable autologging
    mlflow.sklearn.autolog()

    # Get path of test data 
    pos_path   = Path(f'./data_source/masakhane-pos/data/{args.lang}')
    train_data_path = pos_path / 'train.txt'
    dev_data_path = pos_path / 'dev.txt'
    test_data_path = pos_path / 'test.txt'

    # read data from source with sklearn estimator
    reader_estimator = SangkakPosProjetReader()
    list_train_data, pd_train_data = reader_estimator.fit(train_data_path).transform_analysis(
                                                                augment=args.augment, limit=10)
    list_dev_data, pd_dev_data = reader_estimator.fit(dev_data_path).transform_analysis(
                                                                augment=args.augment, limit=10)
    list_test_data, pd_test_data = reader_estimator.fit(test_data_path).transform_analysis(
                                                                augment=args.augment, limit=10)

    feature_estimator = SangkakPosFeaturisation()
    feature_estimator.fit([])

    Xtrain = feature_estimator.transform(list_train_data)
    Xdev  = feature_estimator.transform(list_dev_data)
    Xtest = feature_estimator.transform(list_test_data)

    ytrain = feature_estimator.transform(list_train_data, label=True)
    ydev   = feature_estimator.transform(list_dev_data, label=True)
    ytest  = feature_estimator.transform(list_test_data, label=True)

    if args.re_organised_data:
        all_data_train = pd.concat([pd.DataFrame([json.dumps(x) for y in Xtrain for x in y], columns=["features"]), 
                                    pd.DataFrame([x for y in ytrain for x in y], columns=["labels"])], axis=1, ignore_index=True)
        all_data_dev = pd.concat([pd.DataFrame([json.dumps(x) for y in Xdev for x in y], columns=["features"]), 
                                    pd.DataFrame([x for y in ydev for x in y], columns=["labels"])], axis=1, ignore_index=True)
        all_data_test = pd.concat([pd.DataFrame([json.dumps(x) for y in Xtest for x in y], columns=["features"]), 
                                    pd.DataFrame([x for y in ytest for x in y], columns=["labels"])], axis=1, ignore_index=True)

        all_data_parse = pd.concat([all_data_train, all_data_dev, all_data_test], 
                                axis=0, ignore_index=True)

        all_data_parse.columns = ['features','labels']

        from sklearn.model_selection import train_test_split

        Xtrain, Xtest, ytrain, ytest = train_test_split(
            all_data_parse.drop('labels', axis=1).copy(),
            all_data_parse['labels'].copy(),
            test_size=0.2, random_state=None, shuffle=args.shuffle
        )

        Xtrain, Xdev, ytrain, ydev = train_test_split(
            Xtrain, ytrain, test_size=0.25, 
            random_state=None, shuffle=args.shuffle
        )

        Xtrain = [[json.loads(x[0])] for x in [x for x in Xtrain.values]]
        Xtest  = [[json.loads(x[0])] for x in [x for x in Xtest.values]]
        Xdev   = [[json.loads(x[0])] for x in [x for x in Xdev.values]]

        ydev   = [[x] for x in ydev]
        ytrain = [[x] for x in ytrain]
        ytest  = [[x] for x in ytest]

        num_class = len(list(set(all_data_parse['labels'])))
        print("Number of classes: %s" %num_class)

    try:
        print("Type of target of ytrain data set: %s" %type_of_target(ytrain))
        print("Type of target of ytest data set: %s\n" %type_of_target(ytest))
    except: pass

    import sklearn_crfsuite
    
    #project = f"sangkak-{language}"
    #build_date = str(datetime.now()).replace(' ','_')
    #model_name = Path(f"models/multi/crf_{project}_{build_date}.model")
    #model_file = str(model_name)
    #file_crf = Path(f"models/multi/crf_{project}_{build_date}.object")

    experiment_name = f"POS-{args.lang}: CRF"
    try: experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException: 
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id


    with mlflow.start_run(experiment_id=experiment_id, run_name=args.description, 
                          nested=True, description=args.description) as run:
        mlflow.set_tag("with_augmentation", args.augment)
        mlflow.set_tag("with_shuffle", args.shuffle)
        mlflow.set_tag("with_re_organised", args.re_organised_data)
        mlflow.set_tag("language", args.lang)
        warnings.filterwarnings("ignore")

        #from mlflow.models import infer_signature
        
        params = {
            "algorithm": args.algo,
            "c1": args.c1,
            "c2": args.c2, 
            "max_iterations": args.iter,
            "verbose": True,
            "num_memories":10000,
            "epsilon": 1e-3,
            "linesearch": "MoreThuente",
            "max_linesearch":100000,
            "delta":1e-4,
            #n_job=-1,
            #"c": 2,
            #"pa_type": 2,
            "all_possible_states":True,
            "all_possible_transitions":True
        }

        crf = sklearn_crfsuite.CRF(**params)
        crf.fit(Xtrain, ytrain, Xdev, ydev)    

        labels = list(crf.classes_)

        mlflow.log_params(params)
        mlflow.log_param("language", args.lang)
        mlflow.log_param("data shuffle", args.shuffle)
        mlflow.log_param("data re_organised", args.re_organised_data)
        mlflow.log_param("data augmented", args.augment)
        mlflow.log_param("labels", labels)

        data_train = mlflow.data.from_pandas(pd_train_data, source=str(pos_path), 
                                        targets="tags", name="Masakhane POS Train Datasets")
        mlflow.log_input(data_train, context="crf.training")
        data_test = mlflow.data.from_pandas(pd_test_data, source=str(pos_path), 
                                        targets="tags", name="Masakhane POS Test Datasets")
        mlflow.log_input(data_test, context="crf.testing")
        data_dev = mlflow.data.from_pandas(pd_dev_data, source=str(pos_path), 
                                        targets="tags", name="Masakhane POS Dev Datasets")
        mlflow.log_input(data_dev, context="crf.dev")
        del data_dev, data_test, data_train

        # obtaining metrics such as accuracy, etc. on the test set
        ypred = crf.predict(Xtest)
        print('- F1 score on the test set ')
        f1, acc = eval_metrics(ytest, ypred, average='weighted', 
                            labels=labels, zero_division=False)

        #predictions = crf.predict(Xtrain)
        # only work with pandas
        #signature = infer_signature(Xtest, ypred)
        #logging.getLogger("mlflow").setLevel(logging.DEBUG)
        df_train = feature_estimator.transform_to_sagemaker_format(
            Xtrain[:20], ytrain[:20]
        )
        mlflow.sklearn.log_model(crf, "crf_model", input_example=Xtrain,
                                 registered_model_name="sklearn-crf-classifier-model")

        print('Train set classification report:')
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        report = metrics.flat_classification_report(ytest, ypred, labels=sorted_labels, 
                                       digits=3, zero_division=False, 
                                       output_dict=True)

        # log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        for l, m in report.items():
            if type(m) == float:
                mlflow.log_metric(f"{l}", m)
            else:
                mlflow.log_metric(f"{l}_precision", m['precision'])
                mlflow.log_metric(f"{l}_recall", m['recall'])
                mlflow.log_metric(f"{l}_f1-score", m['f1-score'])
                mlflow.log_metric(f"{l}_support", m['support'])


        """
        model_uri = mlflow.get_artifact_uri("crf_model")
        eval_data = pd.DataFrame(Xtest.copy())
        eval_data['label'] = ypred

        # Evaluate the logged model
        result = mlflow.evaluate(
            model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
        )

        print(f"metrics:\n{result.metrics}")
        print(f"artifacts:\n{result.artifacts}")
        """

        run_id = mlflow.last_active_run().info.run_id
        print(f"\nLogged data and model in run {run_id}")

        # show logged data
        for key, data in fetch_logged_data(run_id).items():
            print(f"\n---------- logged {key} ----------")
            pprint(data)



if __name__ == "__main__":
    main()