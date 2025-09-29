library(quanteda)
library(doc2vec)
library(wordvector)

# doc2vec package -----------------------------

dat <- data.frame(doc_id = docnames(data_corpus_news2014),
                  text = data_corpus_news2014)

d2v <- paragraph2vec(dat, dim = 50, threads = 8, type = "PV-DM")
mat_d2v <- as.matrix(d2v, which = "docs", normalize = FALSE)

sim_d2v <- proxyC::simil(
    mat_d2v,
    mat_d2v["4263794",, drop = FALSE]
)

hist(rowSums(sim_d2v))
tail(sort(s <- rowSums(sim_d2v)))
print(tail(dat[order(s),]), max_ntoken = -1)

# wordvector package -------------------------

toks <- tokens(data_corpus_news2014)
dfmt <- dfm(toks, remove_padding = TRUE)
wdv <- textmodel_doc2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE, iter = 10,
                         tolower = FALSE)

mat_wdv <- as.matrix(wdv, layer = "documents", normalize = FALSE)
sim_wdv <- proxyC::simil(
    mat_wdv,
    mat_wdv["4263794",, drop = FALSE]
)

hist(rowSums(sim_wdv))
tail(sort(s <- rowSums(sim_wdv)))
print(tail(dat[order(s),]), max_ntoken = -1)

# -------------------------

# a <- rownames(as.matrix(d2v, which = "words", normalize = FALSE))
# b <- rownames(as.matrix(wdv, layer = "words", normalize = FALSE))
# length(a)
# length(b)
# length(intersect(a, b)) / length(union(a, b))

plot(rowSums(sim_wdv), rowSums(sim_d2v))
cor(rowSums(sim_wdv), rowSums(sim_d2v))


       