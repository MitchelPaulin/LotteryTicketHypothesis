from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *
import pandas as pd
from LT_TabularBase.TabularLearner import *

train = pd.read_csv("../../data/chocolateratings/chocolaterating.csv")

# what we want to predict
dep_var = 'Rating'

# categorical variables, i.e. variables with discrete values
cat_names = [ 'Company', 'Specific Bean Origin', 'ReviewDate', 'CompanyLocation', 'BeanType', 'BroadBeanOrigin']

# cont variables, i.e. variables with continous values
cont_names = [ 'CocoaPercent']

# Transformations, pre processors
# Categorify: What this does is it will analyze a column, and if it sees that a column is made up of discrete categories, it will assign an integer to each of those categories and replace what was originally in the column with an integer.
# Fill Missing: This will create, for anything that has a missing value an extra column is created which is set to True if the data is provided and False if it is not. We do this because very often the fact that this particular piece of data is missing is interesting and will help us make better predictions.
# Normalize: Normalize the range of data
procs = [FillMissing, Categorify, Normalize]

# Set the seed for reproducibility, this is tested and does work
set_seed(42, reproducible=True)

# Create the data loader
splits = RandomSplitter(valid_pct=0.2)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/chocolateratings/', procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

# Create the learner
learn = LT_tabular_learner(to_nn, metrics=rmse)

# Train for 5 epochs
learn.fit_one_cycle(15, max_lr=slice(1e-03))

"""
Now, before we learn again, we need to 
1. Prune the original model weights
2. For the weights that were not pruned, reset them to what they were originally in learn
"""

# Experiment two
set_seed(42, reproducible=True) # set the seed again to ensure exact same testing conditions

splits = RandomSplitter(valid_pct=0.2)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/chocolateratings/', procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

learn2 = LT_tabular_learner(to_nn, metrics=rmse)
learn.model.LT_prune_layers(p=0.95)
learn2.model.LT_copy_pruned_weights(learn.model)
learn2.model.LT_prune_layers(p=0.95)
learn2.fit_one_cycle(15, max_lr=slice(1e-03))