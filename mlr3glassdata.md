*Remark*: Loading *mlr3* alone will lead to an error, because the learners are not available. A simple solution is to load *mlr3verse* which will load all necessary basic packages of the 'mlr3' family.


```r
library(mlr3verse, quietly = TRUE)
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


## Learning

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
##    379.5677      -0.1933      -4.7213      -3.7510       0.9842      -3.7306  
##          Mg           Na           RI           Si  
##     -5.6911      -2.4291       9.9175      -4.2384  
## 
## Degrees of Freedom: 193 Total (i.e. Null);  184 Residual
## Null Deviance:	    246 
## Residual Deviance: 156.6 	AIC: 176.6
```

The output will depend on the learner, actually it simply displays what the chosen learner returns.


## Prediction

Predicting the class of new data is done through a call to `learner$predict_newdata()`. To determine the accuracy, it is better to call `learner$predict()`, as this also returns the class of the test data in the original data.


```r
predicts <- learner$predict(task, row_ids = itest)
predicts
```

```
## <PredictionClassif> for 20 observations:
##     row_id                truth             response
##         19           containers build wind non-float
##         70     build wind float     build wind float
##          2     vehic wind float build wind non-float
## ---                                                 
##        175     build wind float build wind non-float
##         12 build wind non-float     build wind float
##         39 build wind non-float build wind non-float
```


## Accuracy

From these predictions we can derive the accuracy with `msr()` resp. `msrs(c(...))`.


```r
a <- predicts$score(msrs(c("classif.acc", "classif.ce")))
a
```

```
## classif.acc  classif.ce 
##        0.45        0.55
```

The confusion matrix is returned from


```r
confusion_matrix <- predicts$confusion
print(confusion_matrix)
```

```
##                       truth
## response               build wind float build wind non-float vehic wind float
##   build wind float                    4                    3                1
##   build wind non-float                2                    5                2
##   vehic wind float                    0                    0                0
##   vehic wind non-float                0                    0                0
##   containers                          0                    0                0
##   tableware                           0                    0                0
##   headlamps                           0                    0                0
##                       truth
## response               vehic wind non-float containers tableware headlamps
##   build wind float                        0          0         0         0
##   build wind non-float                    0          1         1         1
##   vehic wind float                        0          0         0         0
##   vehic wind non-float                    0          0         0         0
##   containers                              0          0         0         0
##   tableware                               0          0         0         0
##   headlamps                               0          0         0         0
```

