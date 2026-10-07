import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_rocs(feature_set, roc_by_model, path):
    plt.figure(figsize=(7, 6))
    for model, d in roc_by_model.items():
        plt.plot(d["fpr"], d["tpr"], lw=2, label=f"{model} (AUC = {d['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"Pooled out-of-fold ROC - {feature_set} (PCA 95%)")
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest"); plt.title(title); plt.colorbar()
    labels = ["No expansion", "Expansion"]
    plt.xticks([0, 1], labels); plt.yticks([0, 1], labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
