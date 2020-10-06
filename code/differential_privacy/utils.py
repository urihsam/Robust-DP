from dependency import *
"""
Robust sigma is used for certified robustness, which also plays a role in DP. To transform robust sigma to dp epsilon and delta, 
some requirements for the loss function need to be satisfied as follows,
    loss L is G-Lipschitz, K-strongly convex and,
    L in T-th optimization iteration L_T is defined as L_T- L_optimal = I
"""
def robust_sigma_to_dp_eps(dp_delta, noise_sigma, train_size, max_T, G, K, I, moment_order=1):
    coeff = 2.0/np.sqrt(2.0*I)
    tmp1 = 4 * coeff * G**2 * max_T * np.log(1.0/dp_delta) * (moment_order+1)
    tmp2 = np.square(noise_sigma) * train_size * (train_size-1) * np.sqrt(K) * moment_order
    dp_eps = np.sqrt(tmp1 / tmp2)

    return dp_eps


def robust_sigma_to_dp_delta(dp_eps, noise_sigma, train_size, max_T, G, K, I, moment_order=1):
    coeff = 2.0/np.sqrt(2.0*I)
    tmp1 = 4 * coeff * G**2 * max_T * (moment_order+1)
    tmp2 = np.square(noise_sigma) * train_size * (train_size-1) * np.sqrt(K) * np.square(dp_eps) * moment_order
    dp_delta = tmp2 / tmp1

    return dp_delta


def sgd_eps_delta_to_sigma(eps, delta, max_T, q, coeff=2):
    return coeff * q * np.sqrt(max_T * np.log(1.0/delta)) / eps