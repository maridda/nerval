from nerval import crm, plot_confusion_matrix


y_true = [['O','O', 'O', 'B-PERSON', 'E-PERSON', 'O', 'B-ROLE', 'E-ROLE', 'O', 'B-PERSON', 'E-PERSON', 'O',
          'B-ROLE', 'M-ROLE', 'E-ROLE', 'O', 'O', 'O', 'O', 'O', 'B-PERSON', 'E-PERSON', 'O', 'B-ROLE', 'E-ROLE',
          'O', 'O', 'O', 'W-LOC']]

y_pred =  [['O','O', 'O', 'B-PERSON', 'E-PERSON', 'O', 'B-ROLE', 'E-ROLE', 'O', 'B-PERSON', 'E-PERSON', 'O',
          'B-ROLE', 'M-ROLE', 'E-ROLE', 'O', 'O', 'O', 'O', 'O', 'B-PERSON', 'E-PERSON', 'O', 'B-ROLE', 'E-ROLE',
          'O', 'O', 'O', 'W-LOC']]


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
