import pandas as pd
import numpy as np 
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, entropy, wasserstein_distance, energy_distance

def write_file(filePath, text):
    file = open(filePath, 'w')
    sample = ''
    for t in text:
        sample = sample + str(t) + '\n'
    file.write(sample)
    file.close()


def smooth(df_p, df_e, slipe, k=5):
    '''
    Applying sliding size-fixed windows to expression profiles to obtain smoothed data

    Input: 
    df_p: dataframe of pseudo time 
    df_e: dataframe of expression profile with columns as genes, indexs as cells
    slipe: sliding length
    k: window size

    Output: a list containing the num of each linear trajectory and a dataframe of 
            expression profile after preprocessing
    '''
    count = []    
    for i in range(len(df_p.columns)):
        df_p_c = df_p.iloc[:, i:(i + 1)].dropna()
        df_p_c.columns = ['pseudotime']
        df_p_c.sort_values(by=['pseudotime'], ascending=True, inplace=True)
        df_p_e = pd.merge(df_p_c, df_e, left_index=True, right_index=True)
        df_ee = df_p_e.iloc[:, 1:]
        
        start = 0
        end = 0
        j = 0

        df_e_new = pd.DataFrame()
        while end==0:
            j = j + 1
            if (start+k) < len(df_ee.index):
                df = df_ee.iloc[start:(start+k), :]
            else:
                df = df_ee.iloc[(len(df_ee.index)-k):len(df_ee.index), :]
                end = 1

            # take zeros into consideration
            res = df.apply(lambda x:x.value_counts().get(0,0), axis=0).astype(float)
            for item in range(len(df.columns)):
                res[item]=0.0 if res[item]>=k/2 else df.iloc[:, item].mean()
            df_mean = res.to_frame().T
            df_mean.index=[j]

            if j==0:
                df_e_new = df_mean
            else:
                df_e_new = pd.concat([df_e_new, df_mean])

            start = start + slipe
        
        if i == 0:
            df_e_all = df_e_new
        else:
            df_e_all = pd.concat([df_e_all, df_e_new])

        count.append(len(df_e_new))
    return count, df_e_all

#############new funcs:

def compute_f_divergence(p1, p2, divergence_type="kullback-leibler"):
    kde_p1 = gaussian_kde(p1)
    kde_p2 = gaussian_kde(p2)
    
    grid = np.linspace(
        max(min(p1), min(p2)),
        min(max(p1), max(p2)),
        1000
    )
    
    density_p1 = kde_p1(grid)
    density_p2 = kde_p2(grid)
    
    epsilon = 1e-10
    density_p1 = (density_p1 + epsilon) / np.sum(density_p1 + epsilon)
    density_p2 = (density_p2 + epsilon) / np.sum(density_p2 + epsilon)
    
    if divergence_type == "kullback-leibler":
        return entropy(density_p1, density_p2)
    elif divergence_type == "jensen-shannon":
        m = 0.5 * (density_p1 + density_p2)
        return 0.5 * (entropy(density_p1, m) + entropy(density_p2, m))
    else:
        raise ValueError("Unknown divergence type")

def smooth_divergence(df_pse, df_exp, slide=1, k=5, distance=7):  # example: distance=7 = symmetric KL
    df_exp_smooth = pd.DataFrame(columns=df_exp.columns)
    count = []

    for i in range(len(df_pse.columns)):
        # Extract one pseudotime column (one state/branch)
        df_p_c = df_pse.iloc[:, i:(i + 1)].dropna()
        df_p_c.columns = ['pseudotime']
        df_p_c.sort_values(by='pseudotime', ascending=True, inplace=True)

        # Merge with expression data — safe and aligned
        df_p_e = pd.merge(df_p_c, df_exp, left_index=True, right_index=True)
        df_exp_sorted = df_p_e.iloc[:, 1:].reset_index(drop=True)

        N = len(df_exp_sorted)
        num_windows = (N - k) // slide + 1

        for j in range(num_windows):
            start = j * slide
            end = start + k
            if end + 1 > N:
                break

            data_t = df_exp_sorted.iloc[start:end, :]
            data_t1 = df_exp_sorted.iloc[start + 1:end + 1, :]

            row_values = {}
            for gene in df_exp.columns:
                p1 = data_t[gene].values
                p2 = data_t1[gene].values
                mean_diff = abs(p1.mean() - p2.mean())  # ← Normi-style mean diff
            
                if distance == 1:  # KS
                    div = ks_2samp(p1, p2).statistic
                    
                elif distance == 5:  # forward KL
                    div = compute_f_divergence(p1, p2, "kullback-leibler")
                    
                elif distance == 6:  # backward KL
                    div = compute_f_divergence(p2, p1, "kullback-leibler")
                    
                elif distance == 7:  # Symmetric KL
                    kl_pq = compute_f_divergence(p1, p2, "kullback-leibler")
                    kl_qp = compute_f_divergence(p2, p1, "kullback-leibler")
                    div = 0.5 * (kl_pq + kl_qp)
                    
                elif distance == 8:  # JS divergence
                    div = compute_f_divergence(p1, p2, "jensen-shannon")
                    
                

                elif distance == 9:  # Wasserstein distance
                    div = wasserstein_distance(p1, p2)
                    

                elif distance == 10:  # Energy distance
                    try:
                        div = energy_distance(p1, p2)
                    except:
                        div = 0.0  # fallback in case of error
                    row_values[gene] = div
                
                elif distance == 11:  # Cramér distance (approximate)
                    x_sorted = np.sort(p1)
                    y_sorted = np.sort(p2)
                    min_len = min(len(x_sorted), len(y_sorted))
                    if min_len == 0:
                        div = 0.0
                    else:
                        div = np.linalg.norm(x_sorted[:min_len] - y_sorted[:min_len])
                    
                
                else:
                    raise ValueError("Unsupported distance type!")

            
                #use multi feature score(mean+divergence):
                #alpha = 0.5  # control blend between mean and divergence
                #score = alpha * mean_diff + (1 - alpha) * div
                #row_values[gene] = score
                row_values[gene] = div  # use divergence only

            # Append smoothed row
            df_exp_smooth = pd.concat([df_exp_smooth, pd.DataFrame(row_values, index=[0])], ignore_index=True)
            #count.append(k)
            count.append(num_windows)

    return count, df_exp_smooth








