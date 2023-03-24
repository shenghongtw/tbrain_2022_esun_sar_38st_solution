import numpy as np
def tbrain_metric(y_true, y_pred, num=1):
    pred = y_pred[np.where(y_true == 1)]
    pred.sort()
    return (y_true.sum()-num) / ((y_pred >= pred[num]).sum())