# Adapted from FOCUS / Explicit-Visual-Prompt; see NOTICE.
"""Per-image COD metrics shared by online and saved-mask evaluation."""

from . import metrics


def calc_metrics(y_pred, y_true):
    """Return S-measure, mean E-measure, weighted F-measure, MAE, adaptive F.

    Inputs are masks in [0, 1]. The metric implementations normalize each
    prediction, threshold ground truth at 128 after scaling by 255, and compute
    E-measure over all 256 thresholds. No metric is rounded before averaging.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and ground truth must have the same shape")
    calculators = (
        metrics.Smeasure(), metrics.Emeasure(), metrics.WeightedFmeasure(),
        metrics.MAE(), metrics.Fmeasure(),
    )
    for calculator in calculators:
        calculator.step(pred=y_pred * 255, gt=y_true * 255)
    structure, enhanced, weighted_f, mae, adaptive_f = calculators
    return (
        structure.get_results()["sm"],
        enhanced.get_results()["em"]["curve"].mean(),
        weighted_f.get_results()["wfm"],
        mae.get_results()["mae"],
        adaptive_f.get_results()["fm"]["adp"],
    )
