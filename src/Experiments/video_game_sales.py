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
set_seed(42, reproducible=True)

# Create the data loader
splits = RandomSplitter(valid_pct=0.15)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/videogames/', procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

# Create the learner
learn = LT_tabular_learner(to_nn, metrics=rmse)

# Train for 5 epochs
learn.fit_one_cycle(5, max_lr=slice(1e-03))