library(doc2vec)

dat <- data.frame(doc_id = docnames(data_corpus_news2014),
                  text = data_corpus_news2014)

d2v <- paragraph2vec(dat, threads = 8, type = "PV-DBOW")
mat_d2v <- as.matrix(d2v, which = "docs", normalize = FALSE)

sim_d2v <- proxyC::simil(
    mat_d2v,
    mat_d2v["4263794",, drop = FALSE]
)

hist(rowSums(sim_d2v))
tail(sort(s <- rowSums(sim_d2v)))
print(tail(dat[order(s),]), max_ntoken = -1)

# -------------------------

library(wordvector)

toks <- tokens(data_corpus_news2014, remove_punct = TRUE, remove_symbols = TRUE)
dfmt <- dfm(toks, remove_padding = TRUE)
wdv <- textmodel_doc2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE, iter = 10)

mat_wdv <- as.matrix(wdv, layer = "documents", normalize = FALSE)
sim_wdv <- proxyC::simil(
    mat_wdv,
    mat_wdv["4263794",, drop = FALSE]
)

hist(rowSums(sim_wdv))
tail(sort(s <- rowSums(sim_wdv)))
print(tail(dat[order(s),]), max_ntoken = -1)

# -------------------------

plot(rowSums(sim_wdv), rowSums(sim_d2v))
cor(rowSums(sim_wdv), rowSums(sim_d2v))

