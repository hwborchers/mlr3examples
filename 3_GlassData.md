---
title: "*mlr3* 'glass data' Example"
author: "Hans W. Borchers"
date: "July 2020"
---



*Remark*: Loading *mlr3* alone will lead to an error, because the learners are not available. A simple solution is to load *mlr3verse* which will load all necessary basic packages of the 'mlr3' family.


```r
library(mlr3verse, quietly = TRUE)
```

```
## Warning: package 'mlr3verse' was built under R version 4.0.2
```

```
## Warning: package 'paradox' was built under R version 4.0.2
```


## The 'glass' dataset


```r
Glass <- RWeka::read.arff("glass.arff")
str(Glass)
```

```
## 'data.frame':	214 obs. of  10 variables:
##  $ RI  : num  1.52 1.52 1.52 1.51 1.53 ...
##  $ Na  : num  12.8 12.2 13.2 14.4 12.3 ...
##  $ Mg  : num  3.5 3.52 3.48 1.74 0 2.85 3.65 2.84 0 3.9 ...
##  $ Al  : num  1.12 1.35 1.41 1.54 1 1.44 0.65 1.28 2.68 1.3 ...
##  $ Si  : num  73 72.9 72.6 74.5 70.2 ...
##  $ K   : num  0.64 0.57 0.59 0 0.12 0.57 0.06 0.55 0.08 0.55 ...
##  $ Ca  : num  8.77 8.53 8.43 7.59 16.19 ...
##  $ Ba  : num  0 0 0 0 0 0.11 0 0 0.61 0 ...
##  $ Fe  : num  0 0 0 0 0.24 0.22 0 0 0.05 0.28 ...
##  $ Type: Factor w/ 7 levels "build wind float",..: 1 3 1 6 2 2 3 1 7 2 ...
```


```r
barplot(table(Glass$Type))
grid()
```

![](3_GlassData_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

## Logistic regression

First, we have to set up a 'task'. An *mlr3* classification task is given by a name, a data frame, and a nominal, i.e. factor, attribute of the data.


```r
task <- TaskClassif$new(id = "glass",
                        backend = Glass, target = "Type")
task
```

```
## <TaskClassif:glass> (214 x 10)
## * Target: Type
## * Properties: multiclass
## * Features (9):
##   - dbl (9): Al, Ba, Ca, Fe, K, Mg, Na, RI, Si
```

Next, we decide upon a learner for the task, in this case a classification learner, for instance logistic regression, naive Bayes, a decision tree or random forest, etc.

We will try logistic regression, provided in the *stats* package and available in *mlr3learners* as `classif.log_reg`.


```r
learner <- lrn("classif.log_reg")
learner
```

```
## <LearnerClassifLogReg:classif.log_reg>
## * Model: -
## * Parameters: list()
## * Packages: stats
## * Predict Type: response
## * Feature types: logical, integer, numeric, character, factor, ordered
## * Properties: twoclass, weights
```


### Learning

To generate a fitted model the learner needs to be trained on some data. We split the data set into training and testing data, selecting 10% of the data as test data, the rest is used for training.


```r
n <- nrow(Glass)
itest  <- sample(1:n, 20)
itrain <- setdiff(1:n, itest)
```

Now we can train the model with the training data and display the learned model.


```r
learner$train(task, row_ids = itrain)
learner$model
```

```
## 
## Call:  stats::glm(formula = task$formula(), family = "binomial", data = task$data(), 
##     model = FALSE)
## 
## Coefficients:
## (Intercept)           Al           Ba           Ca           Fe            K  
##     556.644       -1.450       -5.970       -5.048        2.180       -4.742  
##          Mg           Na           RI           Si  
##      -6.915       -4.208       -3.916       -5.820  
## 
## Degrees of Freedom: 193 Total (i.e. Null);  184 Residual
## Null Deviance:	    241.6 
## Residual Deviance: 156.2 	AIC: 176.2
```

The output will depend on the learner, actually it simply displays what the chosen learner returns.


### Prediction

Predicting the class of new data is done through a call to `learner$predict_newdata()`. To determine the accuracy, it is better to call `learner$predict()`, as this also returns the class of the test data in the original data.


```r
predicts <- learner$predict(task, row_ids = itest)
predicts
```

```
## <PredictionClassif> for 20 observations:
##     row_id                truth             response
##         42 build wind non-float     build wind float
##         79     vehic wind float     build wind float
##          7     vehic wind float     build wind float
## ---                                                 
##          1     build wind float     build wind float
##         29 build wind non-float build wind non-float
##        192     build wind float build wind non-float
```

It is possible to generate prediction probabilities. We need to tell the learner to include those probobilities.

```r
learner <- lrn("classif.log_reg", predict_type = "prob")  # "response"
learner$train(task, row_ids = itrain)
predicts <- learner$predict(task, row_ids = itest)
head(predicts)
```
```
Error in dimnames(x) <- dn : 
  length of 'dimnames' [2] not equal to array extent
```
TODO: Find out what is causing this error.

### Accuracy

From these predictions we can derive the accuracy with `msr()` resp. `msrs(c(...))`.


```r
a <- predicts$score(msrs(c("classif.acc", "classif.ce")))
a
```

```
## classif.acc  classif.ce 
##        0.35        0.65
```

All possible accuracy measures will be displayed with

```r
# as.data.table(mlr_measures)
mlr_measures
```

```
## <DictionaryMeasure> with 53 stored values
## Keys: classif.acc, classif.auc, classif.bacc, classif.bbrier,
##   classif.ce, classif.costs, classif.dor, classif.fbeta, classif.fdr,
##   classif.fn, classif.fnr, classif.fomr, classif.fp, classif.fpr,
##   classif.logloss, classif.mbrier, classif.mcc, classif.npv,
##   classif.ppv, classif.precision, classif.recall, classif.sensitivity,
##   classif.specificity, classif.tn, classif.tnr, classif.tp,
##   classif.tpr, debug, oob_error, regr.bias, regr.ktau, regr.mae,
##   regr.mape, regr.maxae, regr.medae, regr.medse, regr.mse, regr.msle,
##   regr.pbias, regr.rae, regr.rmse, regr.rmsle, regr.rrse, regr.rse,
##   regr.rsq, regr.sae, regr.smape, regr.srho, regr.sse,
##   selected_features, time_both, time_predict, time_train
```


The confusion matrix is returned from


```r
confusion_matrix <- predicts$confusion
print(confusion_matrix)
```

```
##                       truth
## response               build wind float build wind non-float vehic wind float
##   build wind float                    6                    1                2
##   build wind non-float                3                    1                0
##   vehic wind float                    0                    0                0
##   vehic wind non-float                0                    0                0
##   containers                          0                    0                0
##   tableware                           0                    0                0
##   headlamps                           0                    0                0
##                       truth
## response               vehic wind non-float containers tableware headlamps
##   build wind float                        0          0         0         0
##   build wind non-float                    0          2         1         4
##   vehic wind float                        0          0         0         0
##   vehic wind non-float                    0          0         0         0
##   containers                              0          0         0         0
##   tableware                               0          0         0         0
##   headlamps                               0          0         0         0
```


## Random Forest

We can repeat the computation with a Random Forest learner such as provided in the *ranger* package by simply replacing the learner option "classif.log_reg" with "ranger".

To keep it readable, we repeat all necessary commands. The dataset is the 'glass' dataset as above; the 'task stays the same.


```r
# Fix the appropriate Random Forrest learner
learner <- lrn("classif.ranger")

# Start the learner and display the model
learner$train(task, row_ids = itrain)
```

```
## Warning: Dropped unused factor level(s) in dependent variable: vehic wind non-
## float.
```

```r
learner$model
```

```
## Ranger result
## 
## Call:
##  ranger::ranger(dependent.variable.name = task$target_names, data = task$data(),      probability = self$predict_type == "prob", case.weights = task$weights$weight) 
## 
## Type:                             Classification 
## Number of trees:                  500 
## Sample size:                      194 
## Number of independent variables:  9 
## Mtry:                             3 
## Target node size:                 1 
## Variable importance mode:         none 
## Splitrule:                        gini 
## OOB prediction error:             20.10 %
```

```r
# Predict classification on the test set
predicts <- learner$predict(task, row_ids = itest)

# Determine the accuracy on the test set
predicts$score(msrs(c("classif.acc", "classif.ce")))
```

```
## classif.acc  classif.ce 
##         0.8         0.2
```

The accuracy is 80% for Random Forrest while it was less than 50 % for a logistic regression.

Of course, calculating the accuracy from the test set is not correct, we need a more reliable approach.
