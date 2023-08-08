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

def gaussian(x, A, mu, sigma):
    return A * np.exp( - ( x - mu )**2 / sigma**2 / 2.)

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def get_numpyro_cube(params, max_flux):
    
    aux_bands = []
    for k in params:
        if k[:4] == "beta" and k != "beta":
            aux_bands.append(k[5:])

    logA, beta, log_gamma = params["logA"], params["beta"], params["log_gamma"]
    t0, log_tau_rise, log_tau_fall, log_extra_sigma = (
        params["t0"],
        params["log_tau_rise"],
        params["log_tau_fall"],
        params["log_extra_sigma"],
    )

    A = max_flux * 10**logA
    gamma = 10**log_gamma
    tau_rise = 10**log_tau_rise
    tau_fall = 10**log_tau_fall
    extra_sigma = 10**log_extra_sigma  # pylint: disable=unused-variable
    
    cube = [A, beta, gamma, t0, tau_rise, tau_fall, extra_sigma]

    for b in aux_bands:
        cube.extend(
            [
                params[f"A_{b}"],
                params[f"beta_{b}"], 
                params[f"gamma_{b}"],
                params[f"t0_{b}"],
                params[f"tau_rise_{b}"],
                params[f"tau_fall_{b}"],
                params[f"extra_sigma_{b}"],
            ]
        )
    return np.array(cube), np.array(aux_bands)
