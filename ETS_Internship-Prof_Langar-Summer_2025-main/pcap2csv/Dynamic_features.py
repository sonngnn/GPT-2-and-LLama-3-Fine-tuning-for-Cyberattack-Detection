# Dynamic_features.py

import numpy as np
from scipy import stats

class Dynamic_features:
    def dynamic_calculation(self, ethsize):
        sum_packets = sum(ethsize)
        min_packets = min(ethsize)
        max_packets = max(ethsize)
        mean_packets = sum_packets / len(ethsize) if ethsize else 0
        std_packets = np.std(ethsize) if ethsize else 0

        return sum_packets, min_packets, max_packets, mean_packets, std_packets

    def dynamic_count(self, protcols_count):  # calculates the Number feature
        return sum(protcols_count.values())

    def dynamic_two_streams(self, incoming, outgoing):
        if not incoming or not outgoing:
            return 0, 0, 0, 0, 0, 0

        inco_ave = sum(incoming) / len(incoming)
        outgoing_ave = sum(outgoing) / len(outgoing)
        magnitude = (inco_ave + outgoing_ave) ** 0.5

        inco_var = np.var(incoming)
        outgo_var = np.var(outgoing)
        radius = (inco_var + outgo_var) ** 0.5

        if len(incoming) >= 2 and len(outgoing) >= 2:
            try:
                correlation, _ = stats.pearsonr(incoming, outgoing)
            except Exception:
                correlation = 0
        else:
            correlation = 0

        if len(incoming) > 0:
            covariance = sum(
                (a - inco_ave) * (b - outgoing_ave) for (a, b) in zip(incoming, outgoing)
            ) / len(incoming)
        else:
            covariance = 0

        var_ratio = inco_var / outgo_var if outgo_var != 0 else 0
        weight = len(incoming) * len(outgoing)

        return magnitude, radius, correlation, covariance, var_ratio, weight
