# nerval
Entity-level confusion matrix and classification report to evaluate Named Entity Recognition (NER) models.


## Labelling schemes supported:
- IO
- BIO1 (IOB1)
- BIO2 (IOB2)
- IOE1
- IOE2
- IOBES
- BILOU
- BMEWO


## Options for the 'scheme' argument:
- **BIO** for the following schemes: IO / BIO1 (IOB1) / BIO2 (IOB2) / IOBES / BILOU / BMEWO
- **IOE** for the following schemes: IOE1 / IOE2
- **BIO** is the default scheme.


## Output:
- Classification report
- Confusion matrix
- Labels for the confusion matrix
- Confusion matrix plot


## How to use it:

```python
>>> from nerval import crm

>>> y_true = [['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC']]
>>> y_pred = [['O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC']]

>>> cr, cm, cm_labels = crm(y_true, y_pred, scheme='BIO')
True Entities: 2
Pred Entities: 2

True Entities with 3 or more tags: 0
Pred Entities with 3 or more tags: 0

True positives:  0
False positives (true = 'O'):  1
False positives (true <> pred):  1
ToT False positives:  2
False negatives:  1

>>> print(cr)
precision  recall  f1_score  true_entities  pred_entities
PER                0.00    0.00      0.00           1.00           0.00
LOC                0.00    0.00      0.00           1.00           1.00
PER__              0.00    0.00      0.00           0.00           1.00
micro_avg          0.00    0.00      0.00           2.00           2.00
macro_avg          0.00    0.00      0.00           2.00           2.00
weighted_avg       0.00    0.00      0.00           2.00           2.00

>>> print(cm)
[[0 1 0 0]
 [1 0 0 0]
 [0 0 0 1]
 [0 0 0 0]]

>>> print(cm_labels)
['LOC', 'O', 'PER', 'PER__']
```

```python
>>> from nerval import plot_confusion_matrix

>>> y_true = [['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC']]
>>> y_pred = [['O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC']]

>>> plot_confusion_matrix(cm, cm_labels, normalize=None, decimal_places=2, figsize=(15,15), SMALL_SIZE=8, MEDIUM_SIZE=12, BIGGER_SIZE=14, cmap='OrRd', xticks_rotation='vertical', title='Confusion Matrix')
```

### Note 1:
**y_true** and **y_pred** could be:
- flat lists
- lists of flat lists
- lists of nested lists.

Flat lists and lists of nested lists will be converted to lists of lists.


### Note 2:
The __ at the end of some entities means that true and pred have the same entity name for the first token but the prediction is somewhat different from the true label.
Examples:
```python
>>> y_true = ['B-ORG', 'I-ORG', 'I-ORG']
>>> y_pred = ['B-ORG']

>>> y_true = ['B-ORG', 'I-ORG', 'I-ORG']
>>> y_pred = ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG']

>>> y_true = ['B-ORG', 'I-ORG', 'I-ORG']
>>> y_pred = ['B-ORG', 'I-PER']

>>> y_true = ['B-ORG', 'I-ORG', 'I-ORG']
>>> y_pred = ['I-ORG', 'I-PER']
```

### Note 3:
The normalize argument could be:
- None
- 'true'
- 'pred'
- 'all'

Default is None.


### Note 4:
In case of division by zero, the result will default to zero.


## Installation
```bash
pip install nerval
```


## License
[MIT](https://github.com/mdadda/nerval/blob/main/LICENCE.txt)


## Citation
```text
@misc{nerval,
  title={{nerval}: Entity-level confusion matrix and classification report to evaluate Named Entity Recognition (NER) models.},
  url={https://github.com/mdadda/nerval},
  note={Software available from https://github.com/mdadda/nerval},
  author={Mariangela D'Addato},
  year={2022},
}
```
