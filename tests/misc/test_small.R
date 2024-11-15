library(quanteda)
library(wordvector)
library(LSX)
library(word2vec)

#data_corpus_guardian <- readRDS("/home/kohei/Dropbox/Public/data_corpus_guardian2016-10k.rds")
#corp <- data_corpus_guardian %>% 
corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_tolower()
toks_grp <- tokens_group(toks)
ndoc(toks)

set.seed(1234)
wov <- wordvector::word2vec(toks, dim = 50, iter = 5, min_count = 5, type = "skip-gram",
                            verbose = TRUE, threads = 1)
synonyms(wov, c("fellow-citizens", "country"))

dov <- doc2vec(toks_grp, wov)
dov[1,]

# -------------------------

set.seed(1234)
lis0 <- as.list(toks)
wov0 <- word2vec::word2vec(lis0, 
                           dim = 50, iter = 5, min_count = 5, type = "skip-gram",
                           verbose = TRUE, threads = 1)

synonyms(wov0, c("fellow-citizens", "country"))

lis_grp0 <- as.list(toks_grp)
dov0 <- word2vec::doc2vec(wov0, newdata = sapply(lis_grp0, paste, collapse = " "))
dov0[1,]

# ----------------------------------------------

dist0 <- proxyC::dist(dov0)
plot(hclust(as.dist(dist0)))

dist <- proxyC::dist(dov)
plot(hclust(as.dist(dist)))
hist(as.matrix(dov))

cor(dist[,1], dist0[,1])
