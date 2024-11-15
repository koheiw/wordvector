library(quanteda)
library(wordvector)
library(LSX)
library(word2vec)

data_corpus_guardian <- readRDS("C:/Users/watan/Dropbox/Public/data_corpus_guardian2016-10k.rds")
corp <- data_corpus_guardian %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                  padding = TRUE) %>% 
    tokens_tolower()
lis <- as.list(toks)

#options(wordvector_threads = 0)
wdv <- wordvector::word2vec(toks, dim = 50, iter = 5, min_count = 5, verbose = TRUE, sample = 0.001) #type = "skip-gram")
lsa <- wordvector::lsa(toks, dim = 300, min_count = 5, verbose = TRUE)
lsa2 <- wordvector::lsa(toks, dim = 100, min_count = 5, verbose = TRUE, weight = "sqrt")
lsa3 <- wordvector::lsa(toks, dim = 100, min_count = 5, verbose = TRUE, reduce = TRUE)
w2v <- word2vec::word2vec(lis, dim = 50, iter = 5, min_count = 5, #type = "skip-gram",
                          threads = 6)

analogy(wdv, ~ australia)
analogy(lsa, ~ australia)
analogy(lsa2, ~ australia)
analogy(lsa3, ~ australia)
analogy(lsa3, ~ australia -animal)
analogy(lsa3, ~ amazon - commerce)
analogy(lsa3, ~ amazon - forest)
analogy(wdv, ~ australia - animal)
analogy(wdv, ~ amazon - commerce)
analogy(wdv, ~ amazon - forest - computer)

emb <- as.matrix(w2v)
vec <- emb["amazon",] - emb["commerce",]
#vec <- emb["japan",]# - emb["china",]
#vec <- emb["king",] - emb["queen",]
predict(w2v, vec, type = "nearest", top_n = 10)

predict(w2v, c("bus", "toilet"), type = "nearest", top_n = 10)

#synonyms(wdv, c("good", "bad", "new", "america", "japan", "bus", "toilet"))
synonyms(wdv, c("bus", "toilet"))
synonyms(wdv, c("amazon", "commerce"))
head(model.frame(~ japan - tokyo, as.data.frame.matrix(t(emb))))


lsa <- lsa(toks, dim = 200, verbose = TRUE)
synonyms(lsa, c("amazon", "commerce"))
analogy(lsa, ~ amazon - forest - computer)
analogy(lsa, ~ forest)
doc2vec(lsa)

dov <- doc2vec(tokens_group(toks), lsa)
plot(hclust(dist(as.matrix(dov))))
