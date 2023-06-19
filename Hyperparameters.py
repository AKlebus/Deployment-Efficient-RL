""" default = {
    "map_dimension": 4   , 
    "delta":         0.2 ,# failure probability
    "epsilon":       0.9 ,# target accuracy
    "c_beta":        1e-5,# scales beta, but should be > 0
    "H":             30  ,# time horizon
    "K":             50  ,# number of deployments
    "N":             1    # batch size
} """

default_continuous = {
    "map_dimension": 4   , 
    "delta":         0.2 ,# failure probability
    "epsilon":       0.9 ,# target accuracy
    "c_beta":        1e-5,# scales beta, but should be > 0
    "H":             200  ,# time horizon
    "K":             10  ,# number of deployments
    "N":             100    # batch size
}

default_stochastic = {
    "epsilon":       0.9 ,# target accuracy
    "i_max":        10  ,# number of iterations
    "beta":        1  ,# bonus coefficient
    "H":             10  ,# time horizon
    "N":             100,    # batch size
    "v_min_squared": 1 # magic coeficient. Use 1 for deterministic, 0.1 / 0.01 for stochastic according to Jiawei
}