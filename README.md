
# BBC-News-Classifier

For more projects I've done please refer to: https://github.com/mijchou

## Introduction
KNN classifier on BBC News Categories

This work aims to build a News classifier, to identify News from 5 categories: business, entertainment, politics, sport and tech. We will perform a knn predictive analysis with the **class** package along text preprocessings using the **tm** package. The classifier is built upon 2225 [BBC News Datasets](http://mlg.ucd.ie/datasets/bbc.html) from 2005-2006. Datasets can be found under the folder __bbc-fulltext__.

**R Packages used:**
* tm: Text-Mining Package 
* plyr: Tools for Splitting, Applying and Combining Data
* class: Functions for Classification **(knn)**

**Sources:<br/>**

[Package ‘tm’](https://cran.r-project.org/web/packages/tm/tm.pdf) <br/>
[The caret Package](https://topepo.github.io/caret/) <br/>
[Package ‘class’](https://cran.r-project.org/web/packages/class/class.pdf) <br/>

## Setup

1. Load libraries
``` r
libs <- c("tm", "plyr", "class", "e1071")
lapply(libs, require, character.only = TRUE)
```

2. Set options - do not read strings as factors
``` r
options(stringAsFactors = FALSE)
```

3. Define 5 categories & specify the path of data
``` r
categories <- c("business", "entertainment",
                "politics", "sport", "tech")
pathname <- "../bbc-fulltext"
```

## Text cleaning

Create **cleanCorpus()** to clean the text data:

1. Remove punctuation
2. Remove white space
3. Convert all words to lowercase
4. Remove words - **stopwords()** specifies the group of predefined stopwords to be removed
<br/> i.e. **stopwords("English")** = the, is, at, which, and on

``` r
cleanCorpus <- function(corpus) {
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, tolower)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"))
  return(corpus.tmp)
}
```

##  Build Term-Document Matrix (TDM)

A Term-Document Matrix describes the frequency of terms in the collection of documents. <br/>

1. Read in data from prespecified path <br/>
2. Apply **cleanCorpus()** created from the previous step <br/>
3. Apply **TermDocumentMatrix()** to create the resulting term-document matrices <br/>
4. *Sparsity* specifies the percentage of emptiness a word occurs in documents. <br/>
i.e. a term occurring 0 times in 70/100 documents = the term has a sparsity of 0.7 <br/>
We wish to remove words with sparsity > 0.7. <br/>
i.e. The resulting matrix retains words occuring in 30% of documents or more.

Create an object ***tdm*** - note that several paths (articles) are being read in.

``` r
generateTDM <- function(cate, path) {
  s.dir <- sprintf("%s/%s", path, cate)
  s.cor <- Corpus(DirSource(directory = s.dir))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl)
  
  s.tdm <- removeSparseTerms(s.tdm, 0.7) # setting sparsity threshold
  result <- list(name = cate, tdm = s.tdm)
}

tdm <- lapply(categories, generateTDM, path = pathname)
```

## Attach names

Create **binCategoryToTDM()** and bind predefined categories to the existing TDM:

1. Coerce TDM from a tdm object to matrix & transpose <br/>
2. Coerce the transposed matrices into data frames <br/>
3. Bind the resulting data frames with the name of each element of tdm <br/>
(Encouraged: Print out tdm produced from above to see how it looks like!) <br/>
4. Rename using the predefined names of categories

``` r
bindCategoryToTDM <- function(tdm) {
  s.mat <- t(data.matrix(tdm[["tdm"]]))
  s.df <- as.data.frame(s.mat, stringsAsFactors = F)
  s.df <- cbind(s.df, rep(tdm[["name"]], nrow(s.df)))
  colnames(s.df)[ncol(s.df)] <- "targetcategory"
  return(s.df)
}

cateTDM <- lapply(tdm, bindCategoryToTDM)
```

## Stack

Create a final data frame for analysis.

1. Bind the tdm data frames into one stack row by row.
2. Assign 0 to missing values.

``` r
tdm.stack <- do.call(rbind.fill, cateTDM)
tdm.stack[is.na(tdm.stack)] <- 0
```

## Hold-out

Create random samples (indicies) for training and test sets. (0.7:0.3)

``` r
train.idx <- sample(nrow(tdm.stack), ceiling(nrow(tdm.stack) * 0.7))
test.idx <- (1:nrow(tdm.stack)) [- train.idx]
```

## Modelling

We will add two more steps of preprocessing before we start feeding the data (news article) into our models.

``` r
tdm.cate <- tdm.stack[, "targetcategory"]
tdm.stack.nl <- tdm.stack[, !colnames(tdm.stack) %in% "targetcategory"]
```

1. tdm.cate stores the column of the target category (including both training and test sets.)
2. tdm.stack.nl stores the rest of the columns. <br/>

A quick break down of *!colnames(tdm.stack) %in% "targetcategory"*:
colnames(tdm.stack) returns all the column names of the tdm.stack. While %in% is the operator of searching, %in% "targetcategory" searches for and return the column. We want the columns that are NOT "targetcategory" that we inverse the result with an ! operator.

These are prepared to forward into the next part:

### KNN

With a K-Nearest-Neighbour (knn) model, each data point (the news article we intend to pair up with a category) search for the nearest k neighbouring data points. The category receiving the most votes from the k data points will be the final winning class (assigned to the news article.) 3 main arguments will be required to feed into the **knn()** function.

1. *training* = matrix (or data frame) of the training set cases, without the category being specified.
2. *test* = matrix (or data frame) of the test set cases.
3. *cl* = true classifications, or the true category, of the training set cases.

And a couple of optional arguments. <br/>

4. *k* = the number of neighbours considered. (default k = 1)
5. *l* = minimum number of votes gained for definite decision. (default l = 0)
6. *prob* = if set TRUE, the proportion of votes (for the winning class) will be returned. (default prob = FALSE)

Now we feed the dataset required into the model. Compare the predictions and the actual values.

``` r
knn.pred <- knn(train = tdm.stack.nl[train.idx, ],
                test = tdm.stack.nl[test.idx, ],
                cl = tdm.cate[train.idx]),
                k = 1)
knn.mat <- table("Predictions" = knn.pred, "Actual" = tdm.cate[test.idx])
knn.mat
```

    ##                Actual
    ## Predictions     business entertainment politics sport tech
    ##   business           153            12       12    10    6
    ##   entertainment        7           102        3     7    5
    ##   politics             2             0      101     0    9
    ##   sport                2             5        6   131    3
    ##   tech                 0             0        0     0   91


It looks pretty good so far - the main diagonal gives us the number of correct prediction for each category.

``` r
knn.acc <- sum(diag(knn.mat))/sum(knn.mat)
knn.acc
```

    ## [1] 0.8665667

Which indicate the proportion of the correct predictions = 0.87 <br/>
Seems like we've done a pretty good prediction!
