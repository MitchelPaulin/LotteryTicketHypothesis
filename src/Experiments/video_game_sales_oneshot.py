from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *
import pandas as pd
from LT_TabularBase.TabularLearner import *

train = pd.read_csv("../../data/videogames/vgsales.csv")

dep_var = 'NA_Sales'

cat_names = ['Platform', 'Year', 'Genre', 'Publisher']

cont_names = ['EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank']

procs = [FillMissing, Categorify]

# Set the seed for reproducibility, this is tested and does work
set_seed(69, reproducible=True)

# Create the data loader
splits = RandomSplitter(valid_pct=0.15)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/videogames/', procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

# Create the learner
learn = LT_tabular_learner(to_nn, metrics=rmse)

# Train for 5 epochs
learn.fit_one_cycle(5, max_lr=slice(1e-03))

"""
Now, before we learn again, we need to 
1. Prune the original model weights
2. For the weights that were not pruned, reset them to what they were originally in learn
"""

# Experiment two
set_seed(69, reproducible=True) # set the seed again to ensure exact same testing conditions

splits = RandomSplitter(valid_pct=0.15)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/titanic/', procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

learn2 = LT_tabular_learner(to_nn, metrics=rmse)
learn.model.LT_prune_layers(p=0.95)
learn2.model.LT_copy_pruned_weights(learn.model)
learn2.model.LT_prune_layers(p=0.95)
learn2.fit_one_cycle(5, max_lr=slice(1e-03))