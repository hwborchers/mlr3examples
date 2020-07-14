
## Available learners

The *mlr3learners* package provides those essential learners for *mlr3* that are maintained by the 'mlr-org' team.

```r
library("mlr3learners")
```
The command `mlr_learners` mentioned in the *mlr3* book does not (yet?) function. 

An overview of the available learners can be found on the https://mlr3learners.mlr-org.com/ page.

On this page, there is also a link to a list of "Learner Extension Packages". The additional packages have to be installed by the user.

## Classification learners

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
 
## Regression learners 
 
    regr.cv_glmnet      Penalized Linear Regression       glmnet
    regr.glmnet         Penalized Linear Regression       glmnet
    regr.kknn           k-Nearest Neighbors               kknn
    regr.km             Kriging                           DiceKriging
    regr.lm             Linear Regression                 stats
    regr.ranger         Random Forest                     ranger
    regr.svm            SVM                               e1071
    regr.xgboost        Gradient Boosting                 xgboost
