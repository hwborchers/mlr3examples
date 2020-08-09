---
title: "Resampling and Benchmarking"
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
mytask <- TaskClassif$new(id = "glass",
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


### Resampling strategy

First we select the resampling strategy as "cross validation". The default is *ten-fold* and can be changed with the `folds` argument. The drawback is that other resampling methods will have different option arguments. 


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

As learner, let's again take the classical CART algorithm for classification.


```r
learner_cart <- lrn("classif.rpart")
# learner_cart
```


### Cross validation

Now we apply the resampling procedure to get a better estimate of the accuracy. No need to define training and testing data, resampling will do this for us.

```r
rresult = resample(task = mytask, learner_cart, resample)
```

```
## INFO  [19:44:06.609] Applying learner 'classif.rpart' on task 'glass' (iter 1/10) 
## INFO  [19:44:06.697] Applying learner 'classif.rpart' on task 'glass' (iter 2/10) 
## INFO  [19:44:06.730] Applying learner 'classif.rpart' on task 'glass' (iter 3/10) 
## INFO  [19:44:06.750] Applying learner 'classif.rpart' on task 'glass' (iter 4/10) 
## INFO  [19:44:06.785] Applying learner 'classif.rpart' on task 'glass' (iter 5/10) 
## INFO  [19:44:06.805] Applying learner 'classif.rpart' on task 'glass' (iter 6/10) 
## INFO  [19:44:06.840] Applying learner 'classif.rpart' on task 'glass' (iter 7/10) 
## INFO  [19:44:06.906] Applying learner 'classif.rpart' on task 'glass' (iter 8/10) 
## INFO  [19:44:06.982] Applying learner 'classif.rpart' on task 'glass' (iter 9/10) 
## INFO  [19:44:07.058] Applying learner 'classif.rpart' on task 'glass' (iter 10/10)
```


```r
rresult$score(msr("classif.acc"))
```

```
##                  task task_id                   learner    learner_id
##  1: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##  2: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##  3: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##  4: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##  5: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##  6: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##  7: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##  8: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##  9: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
## 10: <TaskClassif[44]>   glass <LearnerClassifRpart[32]> classif.rpart
##             resampling resampling_id iteration prediction classif.acc
##  1: <ResamplingCV[19]>            cv         1  <list[1]>   0.5909091
##  2: <ResamplingCV[19]>            cv         2  <list[1]>   0.6363636
##  3: <ResamplingCV[19]>            cv         3  <list[1]>   0.7727273
##  4: <ResamplingCV[19]>            cv         4  <list[1]>   0.6818182
##  5: <ResamplingCV[19]>            cv         5  <list[1]>   0.6666667
##  6: <ResamplingCV[19]>            cv         6  <list[1]>   0.5714286
##  7: <ResamplingCV[19]>            cv         7  <list[1]>   0.6190476
##  8: <ResamplingCV[19]>            cv         8  <list[1]>   0.9047619
##  9: <ResamplingCV[19]>            cv         9  <list[1]>   0.8571429
## 10: <ResamplingCV[19]>            cv        10  <list[1]>   0.6666667
```


```r
# result$aggregate()
round(rresult$aggregate(msrs(c("classif.acc", "classif.ce"))), 2)
```

```
## classif.acc  classif.ce 
##         0.7         0.3
```


## Benchmarking

Create a benchmark design:

```r
design = benchmark_grid(
  tasks = mytask,
  learners = list(lrn("classif.rpart"), lrn("classif.ranger"),
                  lrn("classif.C5.0"), lrn("classif.kknn")),
  resamplings = rsmp("cv", folds = 5)
)
print(design)
```

```
##                 task                    learner         resampling
## 1: <TaskClassif[44]>  <LearnerClassifRpart[32]> <ResamplingCV[19]>
## 2: <TaskClassif[44]> <LearnerClassifRanger[32]> <ResamplingCV[19]>
## 3: <TaskClassif[44]>   <LearnerClassifC5.0[30]> <ResamplingCV[19]>
## 4: <TaskClassif[44]>   <LearnerClassifKKNN[30]> <ResamplingCV[19]>
```

Start the benchmarking process:

```r
bmark = benchmark(design)
```

```
## INFO  [19:44:07.642] Benchmark with 20 resampling iterations 
## INFO  [19:44:07.643] Applying learner 'classif.rpart' on task 'glass' (iter 1/5) 
## INFO  [19:44:07.681] Applying learner 'classif.rpart' on task 'glass' (iter 2/5) 
## INFO  [19:44:07.729] Applying learner 'classif.rpart' on task 'glass' (iter 3/5) 
## INFO  [19:44:07.780] Applying learner 'classif.rpart' on task 'glass' (iter 4/5) 
## INFO  [19:44:07.803] Applying learner 'classif.rpart' on task 'glass' (iter 5/5) 
## INFO  [19:44:07.838] Applying learner 'classif.ranger' on task 'glass' (iter 1/5) 
## INFO  [19:44:09.456] Applying learner 'classif.ranger' on task 'glass' (iter 2/5) 
## INFO  [19:44:09.533] Applying learner 'classif.ranger' on task 'glass' (iter 3/5) 
## INFO  [19:44:09.623] Applying learner 'classif.ranger' on task 'glass' (iter 4/5) 
## INFO  [19:44:09.696] Applying learner 'classif.ranger' on task 'glass' (iter 5/5) 
## INFO  [19:44:09.767] Applying learner 'classif.C5.0' on task 'glass' (iter 1/5) 
## INFO  [19:44:09.933] Applying learner 'classif.C5.0' on task 'glass' (iter 2/5) 
## INFO  [19:44:09.966] Applying learner 'classif.C5.0' on task 'glass' (iter 3/5) 
## INFO  [19:44:10.012] Applying learner 'classif.C5.0' on task 'glass' (iter 4/5) 
## INFO  [19:44:10.043] Applying learner 'classif.C5.0' on task 'glass' (iter 5/5) 
## INFO  [19:44:10.086] Applying learner 'classif.kknn' on task 'glass' (iter 1/5) 
## INFO  [19:44:10.179] Applying learner 'classif.kknn' on task 'glass' (iter 2/5) 
## INFO  [19:44:10.208] Applying learner 'classif.kknn' on task 'glass' (iter 3/5) 
## INFO  [19:44:10.249] Applying learner 'classif.kknn' on task 'glass' (iter 4/5) 
## INFO  [19:44:10.282] Applying learner 'classif.kknn' on task 'glass' (iter 5/5) 
## INFO  [19:44:10.374] Finished benchmark
```

Define a list of measures and then aggregate the measures:

```r
measures = list(msr("classif.acc"), msr("time_train"))
bmark$aggregate(measures)
```

```
##    nr      resample_result task_id     learner_id resampling_id iters
## 1:  1 <ResampleResult[18]>   glass  classif.rpart            cv     5
## 2:  2 <ResampleResult[18]>   glass classif.ranger            cv     5
## 3:  3 <ResampleResult[18]>   glass   classif.C5.0            cv     5
## 4:  4 <ResampleResult[18]>   glass   classif.kknn            cv     5
##    classif.acc time_train
## 1:   0.7197121     0.0094
## 2:   0.7994463     0.0456
## 3:   0.6960133     0.0170
## 4:   0.6965670     0.0024
```

## Extending the example

```r
learners = c("classif.featureless", "classif.rpart", "classif.ranger", "classif.kknn")
learners = lapply(learners, lrn,
  predict_type = "prob", predict_sets = c("train", "test"))
```
