#This script contains the aggregation function class that is used to combine the results of the phase models over multiple windows
#As default max aggregation is used although the mean aggregation function is used as a shortcut to calcuae mean

import numpy as np

class AggregationFunction:

    def __init__(self,name,aggregate):
        self.name=name
        self.aggregate=aggregate

    def __call__(self,x):
        return self.aggregate(x)

max_aggregation=AggregationFunction('Max Aggregation',max)
mean_aggregation=AggregationFunction('Mean Aggregation',np.mean)
min_aggregation=AggregationFunction('Min Aggregation',min)
sum_aggregation=AggregationFunction('Sum Aggregation',np.sum)