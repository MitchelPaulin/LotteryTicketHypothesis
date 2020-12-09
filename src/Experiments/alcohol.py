from fastai import *
from fastai.tabular import *
from fastai.tabular.all import *
import pandas as pd
from LT_TabularBase.TabularLearner import *

train = pd.read_csv("../../data/alcohol/student.csv")

# what we want to predict
dep_var = 'Dalc'

# categorical variables, i.e. variables with discrete values
cat_names = [ 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

# cont variables, i.e. variables with continous values
cont_names = [ 'traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1', 'G2', 'G3']

# Transformations, pre processors
# Categorify: What this does is it will analyze a column, and if it sees that a column is made up of discrete categories, it will assign an integer to each of those categories and replace what was originally in the column with an integer.
# Fill Missing: This will create, for anything that has a missing value an extra column is created which is set to True if the data is provided and False if it is not. We do this because very often the fact that this particular piece of data is missing is interesting and will help us make better predictions.
# Normalize: Normalize the range of data
procs = [FillMissing, Categorify, Normalize]

# Set the seed for reproducibility, this is tested and does work
set_seed(42, reproducible=True)

# Create the data loader
splits = RandomSplitter(valid_pct=0.2)(range_of(train))
to_nn = TabularDataLoaders.from_df(train, '../data/alcohol/', procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, splits=splits, bs=32)

# Create the learner
learn = LT_tabular_learner(to_nn, metrics=rmse)

# epochs
learn.fit_one_cycle(10, max_lr=slice(1e-03))