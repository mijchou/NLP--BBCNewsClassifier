# init
libs <- c("tm", "plyr", "class", "e1071")
lapply(libs, require, character.only = T)

# Set options
options(stringAsFactors = F)

# Set parameters
categories <- c("business", "entertainment",
                "politics", "sport", "tech")
pathname <- "../bbc-fulltext"

# clean text
cleanCorpus <- function(corpus) {
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, tolower)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"))
  return(corpus.tmp)
}

# build TDM
generateTDM <- function(cate, path) {
  s.dir <- sprintf("%s/%s", path, cate)
  s.cor <- Corpus(DirSource(directory = s.dir))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl)
  
  s.tdm <- removeSparseTerms(s.tdm, 0.7)
  result <- list(name = cate, tdm = s.tdm)
}

tdm <- lapply(categories, generateTDM, path = pathname)

# attach name
bindCategoryToTDM <- function(tdm) {
  s.mat <- t(data.matrix(tdm[["tdm"]]))
  s.df <- as.data.frame(s.mat, stringsAsFactors = F)
  
  s.df <- cbind(s.df, rep(tdm[["name"]], nrow(s.df)))
  colnames(s.df)[ncol(s.df)] <- "targetcategory"
  return(s.df)
}

cateTDM <- lapply(tdm, bindCategoryToTDM)

# stack
tdm.stack <- do.call(rbind.fill, cateTDM)
tdm.stack[is.na(tdm.stack)] <- 0

# hold-out
train.idx <- sample(nrow(tdm.stack), ceiling(nrow(tdm.stack) * 0.7))
test.idx <- (1:nrow(tdm.stack)) [- train.idx]
head(test.idx)
head(train.idx)

# modelling
tdm.cate <- tdm.stack[, "targetcategory"]
tdm.stack.nl <- tdm.stack[, !colnames(tdm.stack) %in% "targetcategory"]

# KNN
knn.pred <- knn(tdm.stack.nl[train.idx, ], tdm.stack.nl[test.idx, ], tdm.cate[train.idx])
knn.mat <- table("Predictions" = knn.pred, "Actual" = tdm.cate[test.idx])
knn.acc <- sum(diag(knn.mat))/sum(knn.mat)

knn.mat
knn.acc

# SVM
svm.fit <- svm(targetcategory~., data = tdm.stack[train.idx, ])
svm.pred <- predict(svm.fit, newdata = tdm.stack.nl[test.idx, ])
svm.mat <- table("Predictions" = svm.pred, "Actual" = tdm.cate[test.idx])
svm.acc <- sum(diag(svm.mat))/sum(svm.mat)

svm.mat
svm.acc