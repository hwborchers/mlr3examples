## Data

We will look at how the 'mlr3' package family supports testing of learners. We will use the same data as before.

```r
Glass <- RWeka::read.arff("glass.arff")
dim(Glass)
```

```
## [1] 214  10
```

The task, as defined before, will also be needed.

```r
task <- TaskClassif$new(id = "glass",
                        backend = Glass, target = "Type")
```


## Resampling

*Resampling* is a method to create training and test splits of the data. 'mlr3' supports cross-validation, leave-one-out, bootstrap, and hold-out methods.

To see all resampling methods available in the base 'mlr3' family, display the `mlr_resamplings` dictionary:


```r
as.data.table(mlr_resamplings)
```

```
##            key        params iters
## 1:   bootstrap repeats,ratio    30
## 2:      custom                   0
## 3:          cv         folds    10
## 4:     holdout         ratio     1
## 5:    insample                   1
## 6: repeated_cv repeats,folds   100
## 7: subsampling repeats,ratio    30
```

We will try out the k-fold cross-validation, here given as "cv", doing it k times (with $k^2 models to generate). In the literature the recommended value is $k=10$. We we do a simple 10-fold cross-validation to keep running times small.


### Cross-validation

First we select the resampling strategy.


```r
# resample = mlr_resamplings$get("cv")  # rsmp
resample = rsmp("cv")  # rsmp("cv", folds = 10)
resample
```

```
## <ResamplingCV> with 10 iterations
## * Instantiated: FALSE
## * Parameters: folds=10
```

As learner, let's take this time the classical CART algorithm for classification.


```r
learner <- lrn("classif.rpart")
learner
```

```
## <LearnerClassifRpart:classif.rpart>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```


```r
result = resample(task = task, learner, resample)
```

```
## INFO  [21:42:12.449] Applying learner 'classif.rpart' on task 'glass' (iter 6/10) 
## INFO  [21:42:12.512] Applying learner 'classif.rpart' on task 'glass' (iter 4/10) 
## INFO  [21:42:12.538] Applying learner 'classif.rpart' on task 'glass' (iter 1/10) 
## INFO  [21:42:12.560] Applying learner 'classif.rpart' on task 'glass' (iter 10/10) 
## INFO  [21:42:12.578] Applying learner 'classif.rpart' on task 'glass' (iter 2/10) 
## INFO  [21:42:12.595] Applying learner 'classif.rpart' on task 'glass' (iter 3/10) 
## INFO  [21:42:12.616] Applying learner 'classif.rpart' on task 'glass' (iter 9/10) 
## INFO  [21:42:12.642] Applying learner 'classif.rpart' on task 'glass' (iter 7/10) 
## INFO  [21:42:12.660] Applying learner 'classif.rpart' on task 'glass' (iter 8/10) 
## INFO  [21:42:12.677] Applying learner 'classif.rpart' on task 'glass' (iter 5/10)
```


```r
result$score(msr("classif.acc"))
```

```
##              task task_id               learner    learner_id     resampling
##  1: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##  2: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##  3: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##  4: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##  5: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##  6: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##  7: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##  8: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##  9: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
## 10: <TaskClassif>   glass <LearnerClassifRpart> classif.rpart <ResamplingCV>
##     resampling_id iteration prediction classif.acc
##  1:            cv         1     <list>   0.7272727
##  2:            cv         2     <list>   0.7272727
##  3:            cv         3     <list>   0.7272727
##  4:            cv         4     <list>   0.5909091
##  5:            cv         5     <list>   0.6666667
##  6:            cv         6     <list>   0.6666667
##  7:            cv         7     <list>   0.7619048
##  8:            cv         8     <list>   0.6666667
##  9:            cv         9     <list>   0.9047619
## 10:            cv        10     <list>   0.6666667
```


```r
# result$aggregate()
round(result$aggregate(msrs(c("classif.acc", "classif.ce"))), 2)
```

```
## classif.acc  classif.ce 
##        0.71        0.29
```
