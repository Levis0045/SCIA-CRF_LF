#!/bin/bash

# Experimentation on sklearn CRF environnment

for language in "bbj" "twi" "ibo" "xho" "kin" "nya"
do
    echo -e "\n \t [CRF] Working on language: $language \n"

    python3 experimentations/mlflow_crf.py --lang $language --description "crf: iter 100" --iter 100
    python3 experimentations/mlflow_crf.py --lang $language --description "crf: iter 200" --iter 200
    python3 experimentations/mlflow_crf.py --lang $language --description "crf: re-organised + iter 200" --re-organised-data --iter 200
    python3 experimentations/mlflow_crf.py --lang $language --description "crf: re-organised + iter 200 + shuffle" --re-organised-data --iter 200 --shuffle
    python3 experimentations/mlflow_crf.py --lang $language --description "crf: re-organised + iter 200 + shuffle + augment" --re-organised-data --iter 200 --shuffle --augment
    python3 experimentations/mlflow_crf.py --lang $language --description "crf: iter 200 + augment" --iter 200 --augment
done

# Experimentation on xgboost environnment

for language in "bbj" "twi" "ibo" "xho" "kin" "nya"
do
    echo -e "\n \t [Xgboost] Working on language: $language \n"

    python3 experimentations/mlflow_xgboost.py --lang $language --description "xgb: n_estimator 5060 + lr=0.01" --n-estimators 5060 --learning-rate 0.01
    python3 experimentations/mlflow_xgboost.py --lang $language --description "xgb: n_estimator 10060 + lr=0.01 + shuffle" --n-estimators 10060 --learning-rate 0.01 --shuffle
    python3 experimentations/mlflow_xgboost.py --lang $language --description "xgb: n_estimator 5060 + lr=0.1" --n-estimators 5060 --learning-rate 0.1
    python3 experimentations/mlflow_xgboost.py --lang $language --description "xgb: n_estimator 5060 + lr=0.1 + shuffle" --n-estimators 5060 --learning-rate 0.1 --shuffle
    python3 experimentations/mlflow_xgboost.py --lang $language --description "xgb: n_estimator 10060 + lr=0.1" --n-estimators 10060 --learning-rate 0.1
    python3 experimentations/mlflow_xgboost.py --lang $language --description "xgb: n_estimator 10060 + lr=0.1 + shuffle" --n-estimators 10060 --learning-rate 0.1 --shuffle
    python3 experimentations/mlflow_xgboost.py --lang $language --description "xgb: n_estimator 10060 + lr=0.1 + augment" --n-estimators 10060 --learning-rate 0.1 --augment
    python3 experimentations/mlflow_xgboost.py --lang $language --description "xgb: n_estimator 10060 + lr=0.1 + augment + shuffle" --n-estimators 10060 --learning-rate 0.1 --shuffle --augment

done


