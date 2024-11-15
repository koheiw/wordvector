library(udpipe)
library(word2vec)

data(brussels_reviews, package = "udpipe")
x <- subset(brussels_reviews, language == "nl")
x <- tolower(x$feedback)

model <- word2vec::word2vec(x = x, dim = 15, iter = 20)
emb <- as.matrix(model)

pred <- predict(model, emb["bus",], type = "nearest", top_n = 10)
pred

# similarity in the library
cross <- rowSums(sqrt(crossprod(t(emb), emb["bus",]) / ncol(emb)))
head(sort(cross, decreasing = TRUE))

# cosine similarity 
cosine <- Matrix::rowSums(proxyC::simil(emb, emb["bus",,drop = FALSE]))
head(sort(cosine, decreasing = TRUE))

# they are very similar but different
cor(cross, cosine, use = "pair")
cor(cross, cosine, use = "pair", method = "spearman")    
