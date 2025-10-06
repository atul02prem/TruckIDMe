import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def _get_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    elif hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        s = np.array(s)
        s = s - np.max(s, axis=1, keepdims=True)
        e = np.exp(s)
        return e / np.sum(e, axis=1, keepdims=True)
    else:
        raise ValueError(f"Model {type(clf).__name__} has neither predict_proba nor decision_function.")

def _macro_curve_and_auc(y_bin, y_score):
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    valid_classes = []
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        valid_classes.append(i)

    if len(valid_classes) == 0:
        raise ValueError("No valid classes with positive examples in y_test; cannot compute ROC.")

    macro_auc = np.mean([roc_auc[i] for i in valid_classes])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid_classes)

    fpr_macro, tpr_macro = all_fpr, mean_tpr
    return fpr, tpr, roc_auc, fpr_macro, tpr_macro, macro_auc

def plot_multimodel_multiclass_roc(models_dict, X_test, y_test, label_encoder,
                                   show_macro=True, show_micro=True,
                                   title="Multiclass ROC — Model Comparison",
                                   savepath=None):
    n_classes = len(label_encoder.classes_)
    y_bin = label_binarize(y_test, classes=np.arange(n_classes))
    results = {}

    plt.figure(figsize=(9, 7))

    linestyles = ["-", "--", "-.", ":"]
    style_cycle = 0

    for name, clf in models_dict.items():
        y_score = _get_scores(clf, X_test)

        if show_micro:
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro,
                     linestyle=linestyles[style_cycle % len(linestyles)],
                     linewidth=2,
                     label=f"{name} — micro ROC (AUC={auc_micro:.3f})")
        else:
            auc_micro = None

        if show_macro:
            fpr, tpr, roc_auc, fpr_macro, tpr_macro, auc_macro = _macro_curve_and_auc(y_bin, y_score)
            plt.plot(fpr_macro, tpr_macro,
                     linestyle=linestyles[(style_cycle+1) % len(linestyles)],
                     linewidth=2,
                     label=f"{name} — macro ROC (AUC={auc_macro:.3f})")
        else:
            fpr, tpr, roc_auc, _, _, auc_macro = _macro_curve_and_auc(y_bin, y_score)

        results[name] = {
            "micro_auc": auc_micro,
            "macro_auc": auc_macro,
            "per_class_auc": {label_encoder.classes_[i]: roc_auc[i] for i in roc_auc.keys()}
        }

        style_cycle += 2  

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)

    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8, frameon=True, ncol=1)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()

    return results
