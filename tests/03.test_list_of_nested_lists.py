from nerval import crm, plot_confusion_matrix


y_true = [[['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC'],
           ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-DATE', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG',
           'O', 'B-PER', 'O', 'O', 'O']],
          [['B-TITLE', 'I-TITLE', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'B-DATE', 'O', 'O', 'B-PER', 'O'],
          ['B-LOC', 'O', 'O', 'O', 'O', 'B-ORG',
           'I-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]]

y_pred = [[['O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC'],
           ['B-GEOPOL', 'I-CURRENCY', 'B-GEOPOL', 'B-GEOPOL', 'I-ORG', 'O', 'O', 'B-GEOPOL', 'I-ORG', 'B-GEOPOL', 'O',
           'B-LOC', 'O', 'B-DATE', 'O', 'O', 'I-CURRENCY', 'I-PERCENT', 'O', 'B-DATE', 'O', 'B-DATE', 'O', 'I-PERCENT']],
          [['B-TITLE', 'O', 'B-DATE', 'O', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'I-CURRENCY', 'O', 'I-PERCENT', 'O',
           'O', 'O', 'O', 'I-ORG', 'O', 'O', 'O'],
          ['I-ORG', 'B-DATE', 'B-GEOPOL', 'I-CURRENCY', 'B-GEOPOL', 'I-CURRENCY', 'O', 'B-LOC', 'B-GEOPOL', 'B-GEOPOL',
           'B-LOC', 'I-CURRENCY', 'O', 'O', 'O', 'O', 'B-GEOPOL', 'B-GEOPOL', 'B-GEOPOL', 'I-CURRENCY', 'O', 'O', 'B-DATE',
           'I-CURRENCY', 'I-CURRENCY', 'O', 'B-GEOPOL', 'B-GEOPOL', 'B-DATE', 'O', 'I-PERCENT']]]


# Create Classification Report and Confusion Matrix
cr, cm, cm_labels = crm(y_true, y_pred, scheme='BIO')
print(cr)
print(cm_labels)
print(cm)


# Plot Not Normalised Confusion Matrix
# normalise: None, 'true', 'pred', 'all'. Default is None.
plot_confusion_matrix(cm,
                      cm_labels,
                      normalize=None,
                      decimal_places=2,
                      figsize=(15,15),
                      SMALL_SIZE=8,
                      MEDIUM_SIZE=12,
                      BIGGER_SIZE=14,
                      cmap='OrRd',
                      xticks_rotation='vertical',
                      title='Confusion Matrix')


# Plot Normalised Confusion Matrix where normalise = 'true'
# In an unnormalised confusion matrix, the sum of each row represents the total actual value for each class label (ORG, PER, LOC,...).
# The confusion matrix normalised on 'true' will show what % of that actual total was predicted by the classifier to belong to each one of the available classes.
# The values on the diagonal match the RECALL scores in the classification report.
plot_confusion_matrix(cm,
                      cm_labels,
                      normalize='true',
                      decimal_places=2,
                      figsize=(15,15),
                      SMALL_SIZE=8,
                      MEDIUM_SIZE=12,
                      BIGGER_SIZE=14,
                      cmap='OrRd',
                      xticks_rotation='vertical',
                      title='Confusion Matrix: normalize=true')


# Plot Normalised Confusion Matrix where normalise = 'pred'
# In an unnormalised confusion matrix, the sum of each column represents the total predicted value for each class label (ORG, PER, LOC,...).
# The confusion matrix normalised on 'pred' will show what % of the total class predictions was made against each of the available classes.
# The values on the diagonal match the PRECISION scores in the classification report.
plot_confusion_matrix(cm,
                      cm_labels,
                      normalize='pred',
                      decimal_places=2,
                      figsize=(15,15),
                      SMALL_SIZE=8,
                      MEDIUM_SIZE=12,
                      BIGGER_SIZE=14,
                      cmap='OrRd',
                      xticks_rotation='vertical',
                      title='Confusion Matrix: normalize=pred')
