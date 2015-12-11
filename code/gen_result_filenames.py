import numpy as np
# enumerate results filenames
result_files = ('modern_net_results_mean.np',
                'modern_net_results_median.np'
                'modern_net_drop_bin_scaled_results.np',
                'modern_net_replace_bin_scaled_results.np',
                'modern_net_predicted_bin_scaled_results.np',
                'modern_net_mode_bin_scaled_results.np',
                'modern_net_facanal_bin_scaled_results.np',
                'net_results_mean.np',
                'net_results_median.np'
                'net_drop_bin_scaled_results.np',
                'net_replace_bin_scaled_results.np',
                'net_predicted_bin_scaled_results.np',
                'net_mode_bin_scaled_results.np',
                'net_facanal_bin_scaled_results.np')

# save with pickle
np.savetxt('result_filenames.np', result_files, fmt="%s")