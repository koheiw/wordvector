library(quanteda)
library(wordvector)
library(word2vec)
#library(quanteda.textstats)
options(wordvector_threads = 8)

corp <- data_corpus_news2014
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>%
   tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>%
   tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                 padding = TRUE) %>%
   tokens_tolower()

# word2vec -------------------------------------
wdv <- wordvector::word2vec(toks, dim = 50, type = "cbow", min_count = 5, iter = 5, 
                            verbose = TRUE)

analogy(wdv, ~ washington - america + france)
analogy(wdv, ~ berlin - germany + france, exclude = FALSE, n = 10)

# Word2vec (old) -------------------------------------

lis <- as.list(toks)
w2v <- word2vec::word2vec(lis, dim = 50, min_count = 0, type = "cbow", iter = 5, threads = 8)
emb <- as.matrix(w2v)
vec <- emb["berlin",] - emb["germany",] + emb["france",]
predict(w2v, vec, type = "nearest", top_n = 10)

# LSA -------------------------------------

lsa <- wordvector::lsa(toks, dim = 50, min_count = 0, verbose = TRUE)
analogy(wdv, ~ berlin - germany + france, exclude = FALSE, n = 10)
