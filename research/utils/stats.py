import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_entire_stat_tresh(shape, vs_np, us_np, title="Sequence image sample", thresh = 0.95):
    v_mean = vs_np.mean(axis=0, where=vs_np>np.quantile(vs_np, thresh))# y direction    
    u_mean = us_np.mean(axis=0, where=us_np>np.quantile(us_np, thresh))# x direction    
    # --- Compute flow magnitude
    norm = np.sqrt(u_mean ** 2 + v_mean ** 2)
    # --- Display
    plt.figure(figsize=(8, 8))
    # --- Quiver plot arguments

    nvec = 25  # Number of vectors to be displayed along each image dimension
    nl, nc = shape
    step = max(nl//nvec, nc//nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u_mean[::step, ::step]
    v_ = v_mean[::step, ::step]

    plt.imshow(norm)
    plt.quiver(x, y, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)
    plt.title("Optical flow magnitude and vector field")
    plt.axis('off')
    plt.tight_layout()

    plt.show()
    
def plot_entire_stat_mask(shape, vs_np, us_np, mask, title="Sequence image sample"):
    v_mean = vs_np.mean(axis=0, where=mask)# y direction    
    u_mean = us_np.mean(axis=0, where=mask)# x direction    
    # --- Compute flow magnitude
    norm = np.sqrt(u_mean ** 2 + v_mean ** 2)
    # --- Display
    plt.figure(figsize=(8, 8))
    # --- Quiver plot arguments

    nvec = 25  # Number of vectors to be displayed along each image dimension
    nl, nc = shape
    step = max(nl//nvec, nc//nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u_mean[::step, ::step]
    v_ = v_mean[::step, ::step]

    plt.imshow(norm)
    plt.quiver(x, y, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)
    plt.title("Optical flow magnitude and vector field")
    plt.axis('off')
    plt.tight_layout()

    plt.show()
    
def analyze_hs(hs, S, plot=True, title="H(S)"):
    errs = []
    for cp in range(4, len(S)-4):
        res1 = stats.linregress(np.log10(S[cp:]), np.log10(hs[cp:]))
        res2 = stats.linregress(np.log10(S[:cp]), np.log10(hs[:cp]))
        print(f"S={S[cp]}; Err={(res1.stderr + res2.stderr):.3f}")
        errs.append(res1.stderr + res2.stderr)
    cross = np.argmin(np.array(errs)) + 4
    res_l = stats.linregress(np.log10(S[:cross]), np.log10(hs[:cross]))
    res_h = stats.linregress(np.log10(S[cross:]), np.log10(hs[cross:]))
    if plot:
        n_sigm = 3
        print(f"Opt S = {S[cross]}; H_l(S) = {res_l.slope}; H_h(S) = {res_h.slope}")
        plt.figure()
        plt.title(title)
        print(res_l)
        regr_l =  10**(res_l.intercept+res_l.slope*np.array(np.log10(S)))
        regr_l_upper_bound =  10**(res_l.intercept+res_l.slope*np.array(np.log10(S))+n_sigm*res_l.stderr)
        regr_l_lower_bound =  10**(res_l.intercept+res_l.slope*np.array(np.log10(S))-n_sigm*res_l.stderr)
        plt.plot(S[:cross+1], hs[:cross+1], 'o', color='blue', label=f"$H_l(S)$")
        plt.plot(S, regr_l, lw=1, label=f'$H_l(S) = {res_l.slope:.2f}, S<{S[cross]}$', color='blue', ls='--')
        plt.fill_between(S, regr_l_lower_bound, regr_l_upper_bound, facecolor='blue', alpha=0.25,
                label=f'${n_sigm} \sigma(H_l(S))$')
        
        
        regr_h =  10**(res_h.intercept+res_h.slope*np.array(np.log10(S)))
        regr_h_upper_bound =  10**(res_h.intercept+res_h.slope*np.array(np.log10(S))+n_sigm*res_h.stderr)
        regr_h_lower_bound =  10**(res_h.intercept+res_h.slope*np.array(np.log10(S))-n_sigm*res_h.stderr)
        plt.plot(S[cross:], hs[cross:], 'o', color='red', label=f"$H_h(S)$")
        plt.plot(S, regr_h, lw=1, label=f'$H_h(S) = {res_h.slope:.2f}, S>={S[cross]}$', color='red', ls='--')
        plt.fill_between(S, regr_h_lower_bound, regr_h_upper_bound, facecolor='red', alpha=0.25,
                label=f'${n_sigm} \sigma(H_l(S))$')
        
        plt.legend()
        plt.axvline(S[cross])
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='both')
        plt.plot()
    return cross, res_l.slope, res_h.slope