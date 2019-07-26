
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
try:
    import scikitplot as skplt
except:
    get_ipython().system('pip install scikitplot')
import scikitplot as skplt
from sklearn.metrics import precision_recall_curve


# In[2]:

def roc_curve(y_test, predicted_probas):
    skplt.metrics.plot_roc_curve(y_test, predicted_probas)
    print(plt.show())


# In[3]:

def cumulative_gain(y_test, predicted_probas):
    skplt.metrics.plot_cumulative_gain(y_test, predicted_probas)
    print(plt.show())


# In[4]:

def precision_recall_vs_threshold(y_test, predicted_probas):     
    precisions, recalls, thresholds = precision_recall_curve(y_test, predicted_probas)
    """
        Modified from:
        Hands-On Machine learning with Scikit-Learn
        and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

    print(plt.show())


# In[5]:

def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(y_test, predicted_probas, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    p, r, thresholds = precision_recall_curve(y_test, predicted_probas)
    y_pred_adj = adjusted_classes(predicted_probas, t)
    #print(confusion_matrix(y_test, y_pred_adj))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0, 1.01]);
    plt.xlim([0, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
    plt.show()


# In[6]:

def confusion_matrix(y_test, y_pred):
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    print(plt.show())


# In[7]:

def precision_recall(y_test, predicted_probas):
    skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
    print(plt.show())

