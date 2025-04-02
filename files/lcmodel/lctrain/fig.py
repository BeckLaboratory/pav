"""
Model training and QC figures.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics


FIG_COLOR = {
    'train': 'dodgerblue',
    'validation': 'darkorchid',
    'test': 'orangered'
}

def get_training_fig(
    train_history_list,
    fig_size=(7, 7), fig_dpi=300,
    type_train='train', type_val='validation'
    ):

    fig, ax1 = plt.subplots(1, 1, figsize=fig_size, dpi=fig_dpi)
    ax2 = ax1.twinx()

    type_train_printable = type_train[0].upper() + type_train[1:].lower()
    type_val_printable = type_val[0].upper() + type_val[1:].lower()

    for train_history in train_history_list:
        ax1.plot(train_history['accuracy'], color=FIG_COLOR[type_train], label='Train')
        ax1.plot(train_history['val_accuracy'], color=FIG_COLOR[type_val], label='Validation')

        ax2.plot(train_history['loss'], color=FIG_COLOR[type_train], label='Train', ls=':')
        ax2.plot(train_history['val_loss'], color=FIG_COLOR[type_val], label='Validation', ls=':')

    #ax1.set_ylim([ax1.get_ylim()[0], 1.0])
    ax1.set_ylabel('Accuracy (solid)')

    ax2.set_ylabel('Loss (dotted)')

    ax1.set_xlabel('Epoch')
    ax1.set_title('Training Accuracy and Loss')

    legend_items = [
        mpl.lines.Line2D([0], [0], color=FIG_COLOR[type_train], ls='-', label=f'Accuracy ({type_train_printable})'),
        mpl.lines.Line2D([0], [0], color=FIG_COLOR[type_val], ls='-', label=f'Accuracy ({type_val_printable})'),
        mpl.lines.Line2D([0], [0], color=FIG_COLOR[type_train], ls=':', label=f'Loss ({type_train_printable})'),
        mpl.lines.Line2D([0], [0], color=FIG_COLOR[type_val], ls=':', label=f'Loss ({type_val_printable})')
    ]

    ax1.legend(handles=legend_items)

    fig.tight_layout()

    return fig

def get_roc_fig(
    y_test_list, y_pred_list,
    fig_size=(7, 7), fig_dpi=300
    ):

    test_is_list = isinstance(y_test_list, (list, tuple))
    pred_is_list = isinstance(y_pred_list, (list, tuple))

    if test_is_list != pred_is_list:
        raise ValueError('y_test_list and y_pred_list must both be a list or not')

    if not test_is_list:
        y_test_list = [y_test_list]
        y_pred_list = [y_pred_list]

    roc_auc_list = list()

    fig, ax1 = plt.subplots(1, 1, figsize=fig_size, dpi=fig_dpi)

    for y_test, y_pred in zip(y_test_list, y_pred_list):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred)
        roc_auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
        roc_auc_list.append(roc_auc)

        ax1.plot(fpr, tpr, color='black')

    legend_items = [
        mpl.lines.Line2D([0], [0], color='black', ls='-', label='ROC')
    ]

    ax1.legend(handles=legend_items, fontsize=12)

    ax1.set_xlabel('FPR (False Positive Rate)', size=12)
    ax1.set_ylabel('TPR (True Positive Rate)', size=12)

    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])

    fig.tight_layout()

    return fig, roc_auc_list

