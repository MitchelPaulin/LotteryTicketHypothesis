from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *
import pandas as pd
from LT_TabularBase.TabularLearner import *

# TODO: The error rate is bugged on this one still.

train = pd.read_csv("../../data/health/train.csv")

# What we are predicting
dep_var = 'Response'

# Categorical variables
cat_names = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']

# Continuous Variables
cont_names = ['Age', 'Annual_Premium', 'Vintage']

# Pre Processors
procs = [Categorify, FillMissing, Normalize]

# Set seed for reproducibility
set_seed(42, reproducible=True)

# Create data loader
splits = RandomSplitter(valid_pct=0.2)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/health', procs=procs, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

# Baseline
# Create the learner
learn = LT_tabular_learner(to_nn, metrics=error_rate, emb_drop=0.8)

# Train
learn.fit_one_cycle(10, max_lr=slice(1e-03))