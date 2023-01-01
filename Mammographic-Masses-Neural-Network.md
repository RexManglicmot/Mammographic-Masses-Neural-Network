Mammographic Masses Neural Network
================
Rex Manglicmot
2023-01-01

-   <a href="#continuing-working-document"
    id="toc-continuing-working-document">Continuing Working Document</a>
-   <a href="#introduction" id="toc-introduction">Introduction</a>
-   <a href="#loading-the-libraries" id="toc-loading-the-libraries">Loading
    the Libraries</a>
-   <a href="#loading-the-data" id="toc-loading-the-data">Loading the
    Data</a>
-   <a href="#cleaning-the-data" id="toc-cleaning-the-data">Cleaning the
    Data</a>
-   <a href="#exploratory-data-analysis"
    id="toc-exploratory-data-analysis">Exploratory Data Analysis</a>
-   <a href="#neural-networks" id="toc-neural-networks">Neural Networks</a>
-   <a href="#limitations" id="toc-limitations">Limitations</a>
-   <a href="#conclusion" id="toc-conclusion">Conclusion</a>
-   <a href="#inspiration-for-this-project"
    id="toc-inspiration-for-this-project">Inspiration for this project</a>

## Continuing Working Document

## Introduction

## Loading the Libraries

``` r
library(tidyverse)
```

## Loading the Data

``` r
#get data from UCI website
url <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data'

#load data into object
data_orig <- read.csv(url)
```

## Cleaning the Data

``` r
# look at data
str(data_orig)
```

    ## 'data.frame':    960 obs. of  6 variables:
    ##  $ X5  : chr  "4" "5" "4" "5" ...
    ##  $ X67 : chr  "43" "58" "28" "74" ...
    ##  $ X3  : chr  "1" "4" "1" "1" ...
    ##  $ X5.1: chr  "1" "5" "1" "5" ...
    ##  $ X3.1: chr  "?" "3" "3" "?" ...
    ##  $ X1  : int  1 1 0 1 0 0 0 1 1 1 ...

It would seem that the data is messy.

On the UCI website it listed that the there are a total of 961
observatins but we have only 960. Therefore, the column name would
appear to be an observation. However, upo further inspection, the “X5.1”
and “X3.1” column does not make any sense because values should be based
on 1-5 and 1-4, respectively. I will therefore delete that observation
from the dataset and relabel the columns.

But first, I will make a copy of the original dataset for manipulation.

``` r
#make a copy
data <- data_orig

#change column names
colnames(data) <- c('BIRAD', 'age', 'shape', 'margin', 'density', 'result')

#get rid of BIRAD since on the UCI website it is non-predictive
data <- subset(data, select = -(BIRAD))
```

## Exploratory Data Analysis

## Neural Networks

## Limitations

## Conclusion

## Inspiration for this project
