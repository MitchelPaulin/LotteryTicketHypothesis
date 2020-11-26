from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *
import pandas as pd
from LT_TabularBase.TabularLearner import *

train = pd.read_csv("../../data/winequality/winequality-red.csv")

# What we are predicting
dep_var = 'quality'

# Continuous variables
cont_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# Pre Processors
# Do not need to Categorify or FillMissing
# Normalize: Normalize the range of data
procs = [Categorify, FillMissing, Normalize]

# Set seed for reproducibility
set_seed(7, reproducible=True)

# Create data loader
splits = RandomSplitter(valid_pct=0.2)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/winequality', procs=procs, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

# Baseline
# Create the learner
learn = LT_tabular_learner(to_nn, metrics=rmse)

# Train
learn.fit_one_cycle(35, max_lr=slice(1e-03))