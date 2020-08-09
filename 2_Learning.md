---
title: "*mlr3* learners"
author: "Hans W. Borchers"
date: "July 2020"
output:
  html_document:
    css: "mlr3.css"
    keep_md: true
    toc: true
    toc_float:
      collapsed: false  
---



## Available learners

The *mlr3learners* package provides those essential learners for *mlr3* that are maintained by the 'mlr-org' team.


```r
library("mlr3verse", quietly = TRUE)
# library("mlr3learners")
```
The command `mlr_learners` mentioned in the *mlr3* book is a dictionary of all learners in the `mlr3learners` package.


```r
# as.data.table(mlr_learners)
mlr_learners
```

```
## <DictionaryLearner> with 24 stored values
## Keys: classif.cv_glmnet, classif.debug, classif.featureless,
##   classif.glmnet, classif.kknn, classif.lda, classif.log_reg,
##   classif.multinom, classif.naive_bayes, classif.qda, classif.ranger,
##   classif.rpart, classif.svm, classif.xgboost, regr.cv_glmnet,
##   regr.featureless, regr.glmnet, regr.kknn, regr.km, regr.lm,
##   regr.ranger, regr.rpart, regr.svm, regr.xgboost
```


An overview of the available learners can also be found on the https://mlr3learners.mlr-org.com/ page.

On this page, there is also a link to a list of "Learner Extension Packages". The additional packages have to be installed by the user.

### Classification learners

    classif.cv_glmnet    Penalized Logistic Regression    glmnet
    classif.glmnet       Penalized Logistic Regression    glmnet
    classif.kknn         k-Nearest Neighbors              kknn
    classif.lda          LDA                              MASS
    classif.log_reg      Logistic Regression              stats
    classif.multinom     Multinomial log-linear model     nnet
    classif.naive_bayes  Naive Bayes                      e1071
    classif.qda          QDA                              MASS
    classif.ranger       Random Forest                    ranger
    classif.svm          SVM                              e1071
    classif.xgboost      Gradient Boosting                xgboost
 
### Regression learners 
 
    regr.cv_glmnet       Penalized Linear Regression      glmnet
    regr.glmnet          Penalized Linear Regression      glmnet
    regr.kknn            k-Nearest Neighbors              kknn
    regr.km              Kriging                          DiceKriging
    regr.lm              Linear Regression                stats
    regr.ranger          Random Forest                    ranger
    regr.svm             SVM                              e1071
    regr.xgboost         Gradient Boosting                xgboost


## Parameter sets

All learners have different sets of parameters. To see the parameters for a specific learner, generate this learner and display its parameter set, for example the "nearest neighbor" learner in the *kknn* package (which needs to be installed).

```r
# learner_kknn = mlr_learners$get("regr.kknn")
learner_kknn = lrn("regr.kknn")
print(learner_kknn)
```

```
## <LearnerRegrKKNN:regr.kknn>
## * Model: -
## * Parameters: list()
## * Packages: kknn
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: -
```

Its parameter set is

```r
learner_kknn$param_set
```

```
## <ParamSet>
##          id    class lower upper
## 1:        k ParamInt     1   Inf
## 2: distance ParamDbl     0   Inf
## 3:   kernel ParamFct    NA    NA
## 4:    scale ParamLgl    NA    NA
## 5:  ykernel ParamUty    NA    NA
##                                                            levels default value
## 1:                                                                      7      
## 2:                                                                      2      
## 3: rectangular,triangular,epanechnikov,biweight,triweight,cos,... optimal      
## 4:                                                     TRUE,FALSE    TRUE      
## 5:
```
and can be changed with

```r
learner_kknn$param_set$values = list(k = 3)
learner_kknn$param_set$values$k
```

```
## [1] 3
```
A more direct approach would be
```r
learner_knn = lrn("regr.kknn", k = 3)
```


## Additional Learners

*mlr3* can be extended by learners that are not predefined, see the [mlr3learners](https://mlr3learners.mlr-org.com) page. We are interested to apply the C5.0 decision tree algorithm to our data.

```r
# remotes::install_github("mlr3learners/mlr3learners.c50",
#                          force = TRUE)
library(mlr3learners.c50)
```

Then a C5.0 learner can be defined with

```r
learner_c50 = lrn("classif.C5.0")
learner_c50
```

```
## <LearnerClassifC5.0:classif.C5.0>
## * Model: -
## * Parameters: list()
## * Packages: C50
## * Predict Type: response
## * Feature types: numeric, factor, ordered
## * Properties: missings, multiclass, twoclass, weights
```
and its parameter set contains quite a lot of options.

TODO: As an example, load the C4.5 algorithm and apply to the data.
