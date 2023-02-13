import numpy as np
from data.toy_regression.noisy_sinus_data import myData

def generate_sinus(show_plots=True, task_set=0, data_random_seed=42):
    """Generate a set of tasks for 1D regression.

    Args:
        show_plots: Visualize the generated datasets.
        data_random_seed: Random seed that should be applied to the
            synthetic data generation.
        task_set: int for the regression task 

    Returns:
        data_handlers: A data handler
    """
    print("Generating sinus data...")
    fracs =   [ 0.25  ,  0.25  ,  0.25  ]
    stds =    [ 0.0   ,  0.3   ,  0.1   ]
    spreads = [ 3.5   ,  1.5   ,  1.5   ] 
    offsets = [ -6.5  ,  -2.5  ,  2.5   ] 
    
    fracs =   [ .5    ,  .5     ]
    stds =    [ 0.1   ,  0.1    ]
    spreads = [ 5     ,  3      ] 
    offsets = [ -6.5  ,  2.5    ] 
    
    fracs =   [ .5    ,  .5     ]
    stds =    [ 0.1   ,  0.1    ]
    spreads = [ 2.0    , 5      ] 
    offsets = [ -4.0    ,  2.5    ] 
    
    train_inter=[-6.5, 8]
    test_inter=[-6.5, 8]
    
    fracs =  [ 0.25, 0.25,  0.25, 0.25]
    stds =   [ 0.01, 0.01,  0.25, 0.01]
    spreads =[ 1.8, 2.1,   1.2,  2.0]
    offsets =[ -6.5, -4.1,  -1.2,  4.0]
    
    fracs =   [ 1    ]
    stds =    [ 0.2  ]
    spreads = [ 18   ] 
    offsets = [ -9   ] 
    
    train_inter=[-9, 9]
    test_inter=[-9, 9]

    data = myData(n_train = 200,
                  fracs =  fracs,
                  f = lambda x: np.cos(x),
                  stds = stds,
                  spreads = spreads,
                  offsets = offsets,
                  train_inter = train_inter,
                  test_inter = test_inter
        )
    
    return data

