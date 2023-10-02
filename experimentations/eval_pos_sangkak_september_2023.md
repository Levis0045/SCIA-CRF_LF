# CRF model (initial data)

- F1 score on the test set = 0.8475494955232725
- Accuracy on the test set = 0.8480944712828771

Train set classification report: 

              precision    recall  f1-score   support

        PART      0.796     0.748     0.771       564
       CCONJ      0.747     0.781     0.764       178
       SCONJ      0.819     0.859     0.838       290
         ADJ      0.706     0.617     0.658       214
         ADP      0.910     0.807     0.856       462
         ADV      0.718     0.615     0.662       257
        VERB      0.839     0.867     0.853      1558
         DET      0.795     0.810     0.802       749
        INTJ      0.000     0.000     0.000         2
        NOUN      0.806     0.861     0.833      1866
        PRON      0.895     0.895     0.895       954
       PROPN      0.908     0.870     0.889       729
         NUM      0.994     0.792     0.881       197
       PUNCT      1.000     1.000     1.000       758
         AUX      0.847     0.857     0.852       537
         SYM      0.000     0.000     0.000         0

   micro avg      0.848     0.848     0.848      9315
   macro avg      0.736     0.711     0.722      9315
weighted avg      0.849     0.848     0.848      9315


# CRF (with data reorganised train/test/dev)

- F1 score on the test set = 0.8455994333899967
- Accuracy on the test set = 0.8459145582745748

Train set classification report: 

              precision    recall  f1-score   support

        PART      0.743     0.714     0.729       280
       CCONJ      0.724     0.759     0.741        83
       SCONJ      0.856     0.850     0.853       147
         ADJ      0.829     0.607     0.701       112
         ADP      0.881     0.797     0.837       222
         ADV      0.675     0.692     0.683       120
        VERB      0.840     0.873     0.856       805
         DET      0.798     0.790     0.794       386
        INTJ      0.000     0.000     0.000         2
        NOUN      0.794     0.860     0.826       944
        PRON      0.890     0.864     0.877       478
       PROPN      0.908     0.868     0.887       431
         NUM      1.000     0.862     0.926       116
       PUNCT      1.000     1.000     1.000       408
         AUX      0.861     0.861     0.861       288
         SYM      0.000     0.000     0.000         0

   micro avg      0.846     0.846     0.846      4822
   macro avg      0.737     0.712     0.723      4822
weighted avg      0.847     0.846     0.846      4822


# CRF (data reorganised and shuffle)

- F1 score on the test set = 0.8747545689984426
- Accuracy on the test set = 0.8755703027789299

Train set classification report: 

              precision    recall  f1-score   support

        PART      0.815     0.778     0.796       306
       CCONJ      0.815     0.759     0.786        87
       SCONJ      0.875     0.899     0.887       179
         ADJ      0.779     0.645     0.706        93
         ADP      0.895     0.887     0.891       240
         ADV      0.821     0.671     0.738       143
        VERB      0.861     0.890     0.875       798
         DET      0.802     0.853     0.827       375
        INTJ      0.000     0.000     0.000         1
        NOUN      0.852     0.887     0.869       965
        PRON      0.906     0.900     0.903       528
       PROPN      0.941     0.941     0.941       337
         NUM      0.988     0.837     0.906        98
       PUNCT      0.997     1.000     0.999       377
         AUX      0.890     0.854     0.872       294
         SYM      0.000     0.000     0.000         1

    accuracy                          0.876      4822
   macro avg      0.765     0.738     0.750      4822
weighted avg      0.875     0.876     0.875      4822


# CRF (with only categorical features)

- F1 score on the test set = 0.8460970009477421
- Accuracy on the test set = 0.8469514724180838

Train set classification report: 

              precision    recall  f1-score   support

        PART      0.747     0.706     0.726       310
       CCONJ      0.774     0.722     0.747        90
       SCONJ      0.894     0.886     0.890       201
         ADJ      0.806     0.693     0.745       114
         ADP      0.832     0.860     0.845       264
         ADV      0.855     0.684     0.760       155
        VERB      0.815     0.852     0.833       770
         DET      0.809     0.803     0.806       385
        INTJ      0.000     0.000     0.000         0
        NOUN      0.840     0.859     0.849       979
        PRON      0.868     0.909     0.888       497
       PROPN      0.893     0.887     0.890       337
         NUM      0.935     0.758     0.837        95
       PUNCT      1.000     0.997     0.999       355
         AUX      0.822     0.844     0.833       269
         SYM      0.000     0.000     0.000         1

   micro avg      0.847     0.847     0.847      4822
   macro avg      0.743     0.716     0.728      4822
weighted avg      0.847     0.847     0.846      4822


# XGboost (with only categorical features) + n_estimator 301

Accuracy: 0.49
Balanced Accuracy: 0.41

F-score macro: 0.40
F-score micro: 0.49

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.44      0.51      0.47       944
           1       0.50      0.44      0.47       805
           2       0.52      0.57      0.54       222
           3       0.48      0.56      0.52       386
           4       0.41      0.34      0.38       431
           5       0.99      0.78      0.87       408
           6       0.65      0.37      0.48       147
           7       0.64      0.53      0.58       478
           8       0.52      0.72      0.60       288
           9       0.45      0.32      0.37       280
          10       0.16      0.14      0.15       112
          11       0.19      0.49      0.27        83
          12       0.25      0.20      0.22       116
          13       0.14      0.16      0.15       120
          15       0.00      0.00      0.00         2

    accuracy                           0.49      4822
   macro avg       0.42      0.41      0.40      4822
weighted avg       0.51      0.49      0.49      4822

# XGboost (with only categorical features) + n_estimator 3060

--------------------- Key Metrics --------------------

Accuracy: 0.66
Balanced Accuracy: 0.60

F-score macro: 0.55
F-score micro: 0.66

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.61      0.59      0.60       944
           1       0.65      0.64      0.65       805
           2       0.64      0.72      0.67       222
           3       0.70      0.64      0.67       386
           4       0.64      0.55      0.59       431
           5       0.96      0.97      0.96       408
           6       0.68      0.84      0.75       147
           7       0.77      0.78      0.77       478
           8       0.67      0.81      0.74       288
           9       0.53      0.53      0.53       280
          10       0.38      0.51      0.43       112
          11       0.54      0.59      0.56        83
          12       0.59      0.29      0.39       116
          13       0.40      0.47      0.43       120
          14       0.00      0.00      0.00         0
          15       0.00      0.00      0.00         2

    accuracy                           0.66      4822
   macro avg       0.55      0.56      0.55      4822
weighted avg       0.66      0.66      0.66      4822

# XGboost (with only categorical features) + n_estimator 5060 + lr=0.01

Accuracy: 0.50
Balanced Accuracy: 0.45

F-score macro: 0.43
F-score micro: 0.50

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.49      0.44      0.46       944
           1       0.51      0.46      0.48       805
           2       0.55      0.60      0.57       222
           3       0.49      0.59      0.54       386
           4       0.41      0.45      0.43       431
           5       0.99      0.78      0.87       408
           6       0.68      0.53      0.60       147
           7       0.60      0.56      0.58       478
           8       0.57      0.69      0.62       288
           9       0.45      0.34      0.39       280
          10       0.20      0.28      0.23       112
          11       0.21      0.58      0.31        83
          12       0.24      0.22      0.23       116
          13       0.19      0.21      0.20       120
          15       0.00      0.00      0.00         2

    accuracy                           0.50      4822
   macro avg       0.44      0.45      0.43      4822
weighted avg       0.52      0.50      0.51      4822

# XGboost (with only categorical features) + n_estimator 5060 + lr=0.1

Accuracy: 0.71
Balanced Accuracy: 0.62

F-score macro: 0.63
F-score micro: 0.71

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.62      0.72      0.67       944
           1       0.67      0.72      0.69       805
           2       0.72      0.73      0.72       222
           3       0.74      0.68      0.71       386
           4       0.76      0.55      0.64       431
           5       0.99      0.99      0.99       408
           6       0.74      0.86      0.79       147
           7       0.78      0.81      0.80       478
           8       0.74      0.80      0.77       288
           9       0.64      0.59      0.61       280
          10       0.54      0.50      0.52       112
          11       0.63      0.54      0.58        83
          12       0.78      0.28      0.41       116
          13       0.49      0.47      0.48       120
          15       0.00      0.00      0.00         2

    accuracy                           0.71      4822
   macro avg       0.66      0.62      0.63      4822
weighted avg       0.71      0.71      0.71      4822

# XGboost (with only categorical features) + n_estimator 10060 + lr=0.1

Accuracy: 0.72
Balanced Accuracy: 0.62

F-score macro: 0.63
F-score micro: 0.72

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.63      0.72      0.67       944
           1       0.67      0.72      0.70       805
           2       0.72      0.73      0.72       222
           3       0.75      0.69      0.72       386
           4       0.77      0.56      0.65       431
           5       0.99      0.99      0.99       408
           6       0.74      0.86      0.79       147
           7       0.78      0.82      0.80       478
           8       0.75      0.82      0.78       288
           9       0.64      0.60      0.62       280
          10       0.55      0.50      0.52       112
          11       0.66      0.54      0.60        83
          12       0.80      0.28      0.42       116
          13       0.53      0.48      0.50       120
          15       0.00      0.00      0.00         2

    accuracy                           0.72      4822
   macro avg       0.67      0.62      0.63      4822
weighted avg       0.72      0.72      0.71      4822

# XGboost (with only categorical features + shuffle) + n_estimator 10060 + lr=0.1

Accuracy: 0.78
Balanced Accuracy: 0.64

F-score macro: 0.66
F-score micro: 0.78

--------------- Classification Report ---------------

{'NOUN': 0, 'VERB': 1, 'ADP': 2, 'DET': 3, 'PROPN': 4, 'PUNCT': 5, 'SCONJ': 6, 'PRON': 7, 'AUX': 8, 'PART': 9, 'ADJ': 10, 'CCONJ': 11, 'NUM': 12, 'ADV': 13, 'SYM': 14, 'INTJ': 15}
              precision    recall  f1-score   support

           0       0.73      0.80      0.77       986
           1       0.74      0.77      0.75       777
           2       0.80      0.76      0.78       242
           3       0.76      0.73      0.75       370
           4       0.76      0.79      0.77       333
           5       0.95      1.00      0.97       376
           6       0.80      0.92      0.86       181
           7       0.84      0.82      0.83       495
           8       0.80      0.84      0.82       281
           9       0.69      0.61      0.65       304
          10       0.79      0.49      0.61       128
          11       0.72      0.70      0.71        93
          12       0.74      0.51      0.61       101
          13       0.71      0.56      0.62       152
          14       0.00      0.00      0.00         1
          15       0.00      0.00      0.00         2

    accuracy                           0.78      4822
   macro avg       0.68      0.64      0.66      4822
weighted avg       0.77      0.78      0.77      4822
