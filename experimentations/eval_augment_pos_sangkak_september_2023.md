# CRF model (initial data)

- F1 score on the test set = 0.7986840525034278
- Accuracy on the test set = 0.8000719674090447

Train set classification report: 

              precision    recall  f1-score   support

        PART      0.674     0.589     0.629      4729
       CCONJ      0.770     0.658     0.710      1569
       SCONJ      0.851     0.825     0.838      2540
         ADJ      0.765     0.614     0.681      1765
         ADP      0.751     0.787     0.768      3599
         ADV      0.633     0.539     0.583      1989
        VERB      0.780     0.820     0.799     13556
         DET      0.755     0.739     0.747      6510
        INTJ      0.000     0.000     0.000        24
        NOUN      0.770     0.815     0.792     14914
        PRON      0.829     0.809     0.819      8263
       PROPN      0.889     0.886     0.887      6397
         NUM      0.964     0.888     0.924      1676
       PUNCT      1.000     1.000     1.000      5460
         AUX      0.774     0.821     0.797      4822
         SYM      0.000     0.000     0.000         0

   micro avg      0.800     0.800     0.800     77813
   macro avg      0.700     0.674     0.686     77813
weighted avg      0.799     0.800     0.799     77813

# CRF (with data reorganised train/test/dev)

- F1 score on the test set = 0.7986840525034278
- Accuracy on the test set = 0.8000719674090447

Train set classification report: 

              precision    recall  f1-score   support

        PART      0.674     0.589     0.629      4729
       CCONJ      0.770     0.658     0.710      1569
       SCONJ      0.851     0.825     0.838      2540
         ADJ      0.765     0.614     0.681      1765
         ADP      0.751     0.787     0.768      3599
         ADV      0.633     0.539     0.583      1989
        VERB      0.780     0.820     0.799     13556
         DET      0.755     0.739     0.747      6510
        INTJ      0.000     0.000     0.000        24
        NOUN      0.770     0.815     0.792     14914
        PRON      0.829     0.809     0.819      8263
       PROPN      0.889     0.886     0.887      6397
         NUM      0.964     0.888     0.924      1676
       PUNCT      1.000     1.000     1.000      5460
         AUX      0.774     0.821     0.797      4822
         SYM      0.000     0.000     0.000         0

   micro avg      0.800     0.800     0.800     77813
   macro avg      0.700     0.674     0.686     77813
weighted avg      0.799     0.800     0.799     77813


# CRF (data reorganised and shuffle)

- F1 score on the test set = 0.9850856592949803
- Accuracy on the test set = 0.9850924652692995

Train set classification report: 

              precision    recall  f1-score   support

        PART      0.965     0.957     0.961      4830
       CCONJ      0.969     0.983     0.976      1446
       SCONJ      0.982     0.987     0.985      3120
         ADJ      0.986     0.989     0.988      1962
         ADP      0.987     0.983     0.985      4192
         ADV      0.985     0.973     0.979      2407
        VERB      0.983     0.988     0.985     12896
         DET      0.977     0.975     0.976      6493
        INTJ      1.000     1.000     1.000        13
        NOUN      0.990     0.992     0.991     15618
        PRON      0.988     0.985     0.986      8392
       PROPN      0.997     0.997     0.997      5361
         NUM      0.996     0.990     0.993      1389
       PUNCT      1.000     1.000     1.000      5191
         AUX      0.971     0.972     0.971      4498
         SYM      1.000     1.000     1.000         5

    accuracy                          0.985     77813
   macro avg      0.986     0.986     0.986     77813
weighted avg      0.985     0.985     0.985     77813



# CRF (with only categorical features)



# ------------------------------------------------------

# XGboost (with only categorical features) + n_estimator 301

Accuracy: 0.76
Balanced Accuracy: 0.69

F-score macro: 0.70
F-score micro: 0.76

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.70      0.75      0.73     14916
           1       0.71      0.76      0.73     13554
           2       0.73      0.74      0.74      3602
           3       0.77      0.70      0.73      6508
           4       0.87      0.86      0.87      6396
           5       1.00      1.00      1.00      5461
           6       0.79      0.80      0.80      2540
           7       0.79      0.84      0.81      8263
           8       0.76      0.75      0.75      4822
           9       0.68      0.59      0.63      4727
          10       0.76      0.59      0.66      1765
          11       0.67      0.65      0.66      1569
          12       0.98      0.81      0.89      1679
          13       0.62      0.49      0.55      1989
          15       0.00      0.00      0.00        24

    accuracy                           0.76     77815
   macro avg       0.72      0.69      0.70     77815
weighted avg       0.76      0.76      0.76     77815


# XGboost (with only categorical features) + n_estimator 3060

--------------------- Key Metrics --------------------

Accuracy: 0.76
Balanced Accuracy: 0.70

F-score macro: 0.70
F-score micro: 0.76

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.70      0.73      0.71     14916
           1       0.77      0.73      0.75     13554
           2       0.68      0.70      0.69      3602
           3       0.76      0.72      0.74      6508
           4       0.84      0.86      0.85      6396
           5       1.00      1.00      1.00      5461
           6       0.81      0.85      0.83      2540
           7       0.81      0.82      0.82      8263
           8       0.75      0.83      0.79      4822
           9       0.64      0.57      0.60      4727
          10       0.57      0.61      0.59      1765
          11       0.65      0.72      0.68      1569
          12       0.94      0.82      0.87      1679
          13       0.57      0.55      0.56      1989
          15       0.00      0.00      0.00        24

    accuracy                           0.76     77815
   macro avg       0.70      0.70      0.70     77815
weighted avg       0.76      0.76      0.76     77815


# XGboost (with only categorical features) + n_estimator 5060 + lr=0.01

Accuracy: 0.76
Balanced Accuracy: 0.70

F-score macro: 0.70
F-score micro: 0.76

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.69      0.73      0.71     14916
           1       0.77      0.72      0.74     13554
           2       0.68      0.70      0.69      3602
           3       0.76      0.71      0.74      6508
           4       0.85      0.86      0.85      6396
           5       1.00      1.00      1.00      5461
           6       0.81      0.85      0.83      2540
           7       0.82      0.83      0.82      8263
           8       0.74      0.83      0.79      4822
           9       0.65      0.57      0.61      4727
          10       0.57      0.61      0.59      1765
          11       0.64      0.69      0.66      1569
          12       0.95      0.81      0.88      1679
          13       0.58      0.56      0.57      1989
          15       0.00      0.00      0.00        24

    accuracy                           0.76     77815
   macro avg       0.70      0.70      0.70     77815
weighted avg       0.76      0.76      0.76     77815

# XGboost (with only categorical features) + n_estimator 5060 + lr=0.1

Accuracy: 0.76
Balanced Accuracy: 0.70

F-score macro: 0.70
F-score micro: 0.76

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.70      0.73      0.71     14916
           1       0.77      0.73      0.75     13554
           2       0.68      0.70      0.69      3602
           3       0.76      0.72      0.74      6508
           4       0.84      0.86      0.85      6396
           5       1.00      1.00      1.00      5461
           6       0.81      0.85      0.83      2540
           7       0.81      0.82      0.82      8263
           8       0.75      0.83      0.79      4822
           9       0.64      0.57      0.60      4727
          10       0.57      0.61      0.59      1765
          11       0.65      0.72      0.68      1569
          12       0.94      0.82      0.87      1679
          13       0.57      0.55      0.56      1989
          15       0.00      0.00      0.00        24

    accuracy                           0.76     77815
   macro avg       0.70      0.70      0.70     77815
weighted avg       0.76      0.76      0.76     77815

# XGboost (with only categorical features) + n_estimator 10060 + lr=0.1

Accuracy: 0.76
Balanced Accuracy: 0.70

F-score macro: 0.70
F-score micro: 0.76

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.70      0.73      0.71     14916
           1       0.77      0.73      0.75     13554
           2       0.68      0.70      0.69      3602
           3       0.76      0.72      0.74      6508
           4       0.84      0.86      0.85      6396
           5       1.00      1.00      1.00      5461
           6       0.81      0.85      0.83      2540
           7       0.81      0.82      0.82      8263
           8       0.75      0.83      0.79      4822
           9       0.64      0.57      0.60      4727
          10       0.57      0.61      0.59      1765
          11       0.65      0.72      0.68      1569
          12       0.94      0.82      0.87      1679
          13       0.57      0.55      0.56      1989
          15       0.00      0.00      0.00        24

    accuracy                           0.76     77815
   macro avg       0.70      0.70      0.70     77815
weighted avg       0.76      0.76      0.76     77815



# XGboost (with only categorical features + shuffle) + n_estimator 10060 + lr=0.1

Accuracy: 0.99
Balanced Accuracy: 0.99

F-score macro: 0.98
F-score micro: 0.99

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     15625
           1       0.99      0.99      0.99     13079
           2       0.99      0.99      0.99      4072
           3       0.99      0.99      0.99      6498
           4       0.99      1.00      0.99      5487
           5       1.00      1.00      1.00      5159
           6       0.99      0.99      0.99      3062
           7       0.99      0.99      0.99      8178
           8       0.98      0.98      0.98      4492
           9       0.98      0.98      0.98      4993
          10       0.99      0.99      0.99      1874
          11       0.97      0.98      0.98      1417
          12       1.00      0.99      0.99      1339
          13       0.99      0.98      0.99      2523
          14       1.00      1.00      1.00        11
          15       0.86      1.00      0.92         6

    accuracy                           0.99     77815
   macro avg       0.98      0.99      0.98     77815
weighted avg       0.99      0.99      0.99     77815

