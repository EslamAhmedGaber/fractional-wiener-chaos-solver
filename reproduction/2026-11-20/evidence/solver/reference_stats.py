"""Cross-fitted control variates for independent-path reference moments."""
import numpy as np


def reference_moments(values, control, control_variance, batches=64):
    """values (time, path, state), centred Gaussian control (time, path).

    Each evaluation batch estimates covariance about its own sample mean.
    The control is centred at its known population moment; the slope and
    the training mean used to choose it use only the opposite half sample.
    """
    nt, paths, dims = values.shape
    if paths % batches or batches % 2 or paths // batches < 2:
        raise ValueError("Need an even number of equal batches of at least two paths")
    bm = np.empty((nt, batches, dims))
    bc = np.empty((nt, batches, dims, dims))
    for t in range(nt):
        y = values[t].reshape(batches, -1, dims)
        x = control[t].reshape(batches, -1)
        ybar = y.mean(1)
        yc = y - ybar[:, None, :]
        sample_cov = np.einsum("bki,bkj->bij", yc, yc) / (y.shape[1] - 1)
        xbar = x.mean(1)
        sq = x*x - control_variance[t]
        sqbar = sq.mean(1)
        for train, test in ((slice(None, batches//2), slice(batches//2, None)),
                            (slice(batches//2, None), slice(None, batches//2))):
            xx = x[train].reshape(-1)
            yy = y[train].reshape(-1, dims)
            centred = yy - yy.mean(0)
            beta = (xx-xx.mean()) @ centred / (len(xx)-1) / control_variance[t]
            bm[t, test] = ybar[test] - xbar[test, None] * beta
            sqtrain = sq[train].reshape(-1)
            sqtrain = sqtrain - sqtrain.mean()
            denom = sqtrain @ sqtrain
            beta_cov = (centred.T * sqtrain) @ centred / denom
            bc[t, test] = sample_cov[test] - sqbar[test, None, None] * beta_cov
    return (bm.mean(1), bc.mean(1), bm.std(1, ddof=1)/np.sqrt(batches),
            bc.std(1, ddof=1)/np.sqrt(batches))

