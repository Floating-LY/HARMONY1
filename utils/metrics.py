import numpy as np
import properscoring as ps

from scipy.stats import norm
def CRPS(mean, std, obs):
    crps_matrix = np.zeros_like(mean)
    
    for i in range(mean.shape[0]):  
        for j in range(mean.shape[1]):  
            for k in range(mean.shape[2]):  
                mean_ijk = mean[i, j, k]
                std_ijk = std[i, j, k]
                obs_ijk = obs[i, j, k]
                #sample from distribution
                samples = np.random.normal(mean_ijk, std_ijk, size=100)
                
                crps_ijk = ps.crps_ensemble(obs_ijk, samples)
                crps_matrix[i, j, k] = crps_ijk
    avg_crps = np.mean(crps_matrix)
    return avg_crps

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true,mean,std,samples):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    if samples is not None:
        crps = np.mean(ps.crps_ensemble(true, samples))
    
    if mean is not None and samples is None:
        crps = CRPS(mean ,std,true)
    else: 
        crps = np.mean(ps.crps_ensemble(true, pred))
    
    return mae, mse, rmse, mape, mspe, crps