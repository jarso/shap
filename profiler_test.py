import shap
from shap import TreeExplainer as TreeCext
from shap.explainers.pytree import TreeExplainer

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import time
import cProfile
pr = cProfile.Profile()

X, y = make_regression(n_samples=100, n_features=6, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=6, random_state=0)
regr.fit(X, y)
model = regr

# import sys
# print(sys.version)
#
# print(shap.__version__)

# X,y = shap.datasets.boston()git
#
# model = RandomForestRegressor(n_estimators=100, max_depth=8)
# model.fit(X, y)


start = time.time()
t = TreeCext(model).shap_values(X, banz=True)
print(time.time() - start)


# start = time.time()
#
# pr.enable()
# t = TreeExplainer(model).shap_values(X)
# pr.disable()
# pr.print_stats(sort='cumtime')
#
# print(time.time() - start)
