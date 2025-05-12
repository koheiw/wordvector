require(quanteda)
require(wordvector)

toks0 <- tokens("a b c d e")
toks <- tokens("a b c d e f")

wov0 <- textmodel_word2vec(toks, dim = 3, type = "cbow", min_count = 0, #sample = 1,
                           normalize = FALSE)
wov <- textmodel_word2vec(toks, dim = 3, type = "cbow", min_count = 0, #sample = 1, 
                          model = wov0, 
                          normalize = FALSE)
#wov0$values
#wov$values
f <- intersect(rownames(wov0$values), rownames(wov$values))
f
sim <- proxyC::simil(wov0$values[f,], wov$values[f,], diag = TRUE)
hist(sim@x, xlim = c(-1, 1))

# --------------------------------


corp <- data_corpus_inaugural %>%
    corpus_reshape()
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>%
     tokens_remove(stopwords(), padding = FALSE)
toks0 <- head(toks, 5000)
toks <- tail(toks, 1000)

wov0 <- textmodel_word2vec(toks, dim = 50, type = "cbow", 
                           normalize = FALSE)
wov <- textmodel_word2vec(toks, dim = 50, type = "cbow", 
                          model = wov0, iter = 1,
                          normalize = FALSE)

head(probability(wov0, "america"))
head(probability(wov, "america"))

f <- intersect(rownames(wov0$values), rownames(wov$values))
f
sim <- proxyC::simil(wov0$values[f,], wov$values[f,], diag = TRUE)
hist(sim@x, xlim = c(-1, 1))

