from copy import deepcopy
from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *
import pandas as pd
from LT_TabularBase.TabularLearner import *

# to get data set run ! kaggle competitions download -c titanic and move to /data/titanic

train = pd.read_csv("../../data/titanic/train.csv")

test = pd.read_csv("../../data/titanic/test.csv")

# what we want to predict
dep_var = 'Survived'

# categorical variables, i.e. variables with discrete values
cat_names = [ 'Sex', 'Ticket', 'Cabin', 'Embarked']

# cont variables, i.e. variables with continous values
cont_names = [ 'Age', 'SibSp', 'Parch', 'Fare']

# Transformations, pre processors
# Categorify: What this does is it will analyze a column, and if it sees that a column is made up of discrete categories, it will assign an integer to each of those categories and replace what was originally in the column with an integer.
# Fill Missing: This will create, for anything that has a missing value an extra column is created which is set to True if the data is provided and False if it is not. We do this because very often the fact that this particular piece of data is missing is interesting and will help us make better predictions.
# Normalize: Normalize the range of data
procs = [FillMissing, Categorify, Normalize]

# Create an initial learner to hold our old weights
set_seed(42, reproducible=True) # set the seed again to ensure exact same testing conditions
splits = RandomSplitter(valid_pct=0.2)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/titanic/', procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)
learn_old = LT_tabular_learner(to_nn, metrics=error_rate)


# Experiment
set_seed(42, reproducible=True) # set the seed again to ensure exact same testing conditions

splits = RandomSplitter(valid_pct=0.2)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/titanic/', procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

learn = LT_tabular_learner(to_nn, metrics=error_rate)
ROUNDS = 10
s = 0.25
p = 0.95 # target 
while learn.model.LT_calculate_pruned_percentage() < p:
    learn.fit_one_cycle(5, max_lr=slice(1e-03))
    learn.model.LT_prune_layers(s)
    learn_new = LT_tabular_learner(to_nn, metrics=error_rate)
    learn_new.model.LT_copy_pruned_weights(learn.model)
    learn_new.model.LT_copy_unpruned_weights(learn_old.model)
    learn_new.model.LT_prune_layers(learn.model.LT_calculate_pruned_percentage())
    print(learn_new.model.LT_calculate_pruned_percentage())
    learn = learn_new