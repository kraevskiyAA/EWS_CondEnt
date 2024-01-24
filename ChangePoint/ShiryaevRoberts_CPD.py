import numpy as np


class Stat(object):
    def __init__(self, threshold, direction="unknown", init_stat=0.0):
        self._direction = str(direction)
        self._threshold = float(threshold)
        self._stat = float(init_stat)
        self._alarm = self._stat / self._threshold
    
    @property
    def direction(self):
        return self._direction

    @property
    def stat(self):
        return self._stat
        
    @property
    def alarm(self):
        return self._alarm
        
    @property
    def threshold(self):
        return self._threshold
    
    def update(self, **kwargs):
        # Statistics may use any of the following kwargs:
        #   ts - timestamp for the value
        #   value - original value
        #   mean - current estimated mean
        #   std - current estimated std
        #   adjusted_value - usually (value - mean) / std
        # Statistics call this after updating '_stat'
        self._alarm = self._stat / self._threshold


# Тут реализована модификация статистики взвешенного экспоненциального среднего.
class MeanExpNoDataException(Exception):
    pass

class MeanExp(object):
    def __init__(self, new_value_weight, load_function='median'):
        self._load_function = load_function
        self._new_value_weight = new_value_weight
        self.load([])

        
    @property
    def value(self):
        if self._weights_sum <= 1:
            raise MeanExpNoDataException('self._weights_sum <= 1')
        return self._values_sum / self._weights_sum

    def update(self, new_value, **kwargs):
        self._values_sum = (1 - self._new_value_weight) * self._values_sum + new_value
        self._weights_sum = (1 - self._new_value_weight) * self._weights_sum + 1.0

    def load(self, old_values):
        if old_values:
            old_values = [value for ts, value in old_values]
            mean = float(self._load_function(old_values))
            self._weights_sum = min(float(len(old_values)), 1.0 / self._new_value_weight)
            self._values_sum = mean * self._weights_sum
        else:
            self._values_sum = 0.0
            self._weights_sum = 0.0


class AdjustedShiryaevRoberts_mean(Stat):
    def __init__(self, mean_diff, threshold, max_stat=float("+inf"), init_stat=0.0):
        super(AdjustedShiryaevRoberts_mean, self).__init__(threshold,
                                                      direction="up",
                                                      init_stat=init_stat)
        self._mean_diff = mean_diff
        self._max_stat = max_stat

    def update(self, adjusted_value, **kwargs):
        likelihood = np.exp(self._mean_diff * (adjusted_value - self._mean_diff / 2.))
        self._stat = min(self._max_stat, (1. + self._stat) * likelihood)
        Stat.update(self)

    def update_corrected(self, adjusted_value, mean_recalc, **kwargs):
        likelihood = np.exp(mean_recalc * (adjusted_value - mean_recalc/ 2.))
        self._stat = min(self._max_stat, (1. + self._stat) * likelihood)
        Stat.update(self)

def compute_SR_mean(time_series, 
                    alpha = 0.01,
                    beta = 0.05,
                    mean_diff = 0.15,
                    max_stat = 1e9):
    
    if isinstance(mean_diff, list):
        pass
    else:
        mean_diff = [mean_diff]


    c = 1
    for theta in mean_diff:
        mean_exp = MeanExp(new_value_weight=alpha)
        var_exp = MeanExp(new_value_weight=beta)
        sr = AdjustedShiryaevRoberts_mean(theta, 1., max_stat=max_stat)

        stat_trajectory_mean, mean_values, var_values, diff_values = [], [], [], []

        prev_mean  = time_series[0]
        for ts, x_k in enumerate(time_series):

            try:
                mean_estimate = mean_exp.value
            except MeanExpNoDataException:
                mean_estimate = prev_mean
            
            try:
                var_estimate = var_exp.value
            except MeanExpNoDataException:
                var_estimate = 1.
            
            predicted_diff_value = (x_k - mean_estimate)
            predicted_diff_mean = np.sqrt(var_estimate)
            sr.update(predicted_diff_value/predicted_diff_mean)
            
            diff_values.append(predicted_diff_value/predicted_diff_mean)
            
            mean_exp.update(x_k)
            diff_value = (x_k - mean_estimate) ** 2
            var_exp.update(diff_value)
            
            stat_trajectory_mean.append(sr._stat)
            mean_values.append(mean_estimate)
            var_values.append(np.sqrt(var_estimate)) 

            prev_mean = x_k
        if c != 1:
            sr_stats = np.concatenate((sr_stats, np.array(stat_trajectory_mean).reshape(-1,1)), axis = 1)
        else:
            sr_stats = np.array(stat_trajectory_mean).reshape(-1,1)
            
        c += 1

    return np.mean(sr_stats, axis = 1)