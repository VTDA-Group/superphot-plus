import numpy as np
from alerce.core import Alerce

alerce = Alerce()

def get_pred_class(ztf_name, reflect_style=False):
    """Get alerce probabilities corresponding to the four (no SN IIn)
    classes in our ZTF classifier.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the object.
    reflect_style : bool, optional
        If True, change format of output labels. Default is False.

    Returns
    -------
    str
        Predicted class label.
    """
    global alerce
    o = alerce.query_probabilities(oid=ztf_name, format="pandas")
    o_transient = o[o["classifier_name"] == "lc_classifier_transient"]
    label = o_transient[o_transient["ranking"] == 1]["class_name"].iat[0]
    return SnClass.get_reflect_style(label) if reflect_style else label