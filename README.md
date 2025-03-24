skl_seas_ens - A wrapper for skl-classifiers to handle 
complex seasonal effects which cannot be handled by adding
trend/seasonal features 
============================================================

**skl_seas_ens** is a classifier which manages a ensemble of base classifiers.
We evaluating, we choose the base classifier which is responsible for the
'season' which our data belongs to, which is determined based on a 'temporal' feature
of our data. When training this classifier, we exclude data which is temporally to 
far removed from the season it is responsible for.

Thus, skl_seas_ens is best suited for cases, where we suspect our feautures to exhibit 
complex interactions depending on the season, which cannot
easily be modeled by adding trend/season features. Moreover, since
we exclude temporally far removed data from training, it is best suited
for use cases, where training data is plentiful. 

It is a classifier compliant to the requirements for an
skl classifier, see https://scikit-learn.org/stable/developers/develop.html.
Thus is is compatible with methods such as sklearn.model_selection.GridSearchCV
 and sklearn.model_selection.cross_validate.

