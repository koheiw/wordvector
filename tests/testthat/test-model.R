require(quanteda)
require(wordvector)
options(wordvector_threads = 5)

toks0 <- tokens("a b c d e")
toks <- tokens("a b c d e f")

set.seed(1234)
wov0 <- textmodel_word2vec(toks, dim = 3, type = "cbow", min_count = 0, #sample = 1,
                           normalize = FALSE)

set.seed(1234)
wov1 <- textmodel_word2vec(toks, dim = 3, type = "cbow", min_count = 0, #sample = 1, 
                          normalize = FALSE)

set.seed(1234)
wov2 <- textmodel_word2vec(toks, dim = 3, type = "cbow", min_count = 0, #sample = 1, 
                          model = wov0, 
                          normalize = FALSE)

f <- intersect(rownames(wov0$values), rownames(wov1$values))

par(mfrow = c(2, 1), mar = c(3, 3, 3, 2))
hist(wov0$values[f,] - wov1$values[f,], xlim = c(-1, 1), main = "wov1")
hist(wov0$values[f,] - wov2$values[f,], xlim = c(-1, 1), main = "wov2")
par(mfrow = c(1, 1))

# --------------------------------

corp <- data_corpus_inaugural %>%
    corpus_reshape()
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>%
     tokens_remove(stopwords(), padding = FALSE)
toks0 <- head(toks, 3000)
toks1 <- toks2 <- tail(toks, 3000)

#set.seed(1234)
wov0 <- textmodel_word2vec(toks0, dim = 50, type = "cbow", min_count = 5, sample = 1,
                           normalize = FALSE)

#set.seed(1234)
wov1 <- textmodel_word2vec(toks1, dim = 50, type = "cbow", min_count = 5, sample = 1, 
                           normalize = FALSE, verbose = TRUE)

#set.seed(1234)
wov2 <- textmodel_word2vec(toks2, dim = 50, type = "cbow", min_count = 5, sample = 1, 
                           model = wov0, normalize = FALSE, verbose = TRUE,)

f <- intersect(rownames(wov0$values), rownames(wov1$values))
f

dfmt <- tail(dfm_group(dfm(toks)), 10)
dov0 <- textmodel_doc2vec(dfmt, wov0)
dov1 <- textmodel_doc2vec(dfmt, wov1)
dov2 <- textmodel_doc2vec(dfmt, wov2)

Matrix::diag(proxyC::simil(dov0$values, dov1$values))
Matrix::diag(proxyC::simil(dov0$values, dov2$values))



