library(quanteda)
library(Matrix)
library(doc2vec)
library(wordvector)

dat <- data.frame(doc_id = docnames(data_corpus_news2014),
                  text = as.character(data_corpus_news2014))
dat2 <- head(dat, 100)
dat2$doc_id <- paste0(dat2$doc_id, "_copy")
dat3 <- rbind(dat, dat2)
rownames(dat3) <- dat3$doc_id

# doc2vec package -----------------------------

d2v <- paragraph2vec(dat3, dim = 50, threads = 1, type = "PV-DM", trace = FALSE)
mat_d2v <- as.matrix(d2v, which = "docs", normalize = FALSE)
hist(mat_d2v["4362315",, drop = TRUE])

sim_d2v <- proxyC::simil(
    mat_d2v,
    mat_d2v["4362315",, drop = FALSE]
)

sim_d2v <- proxyC::simil(
    mat_d2v,
    mat_d2v["4102068",, drop = FALSE]
)

hist(rowSums(sim_d2v))
tail(sort(s <- rowSums(sim_d2v)))
print(tail(dat3[order(s),]))

sim_d2v_all <- proxyC::simil(
    mat_d2v,
    mat_d2v[tail(dat3$doc_id, 100),, drop = FALSE]
)
mean(diag(sim_d2v_all)) - mean(sim_d2v_all)

# wordvector package -------------------------

corp <- corpus(dat3)
toks <- tokens(corp)
dfmt <- dfm(toks, remove_padding = TRUE)
options(wordvector_threads = 8)
wdv <- textmodel_doc2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE, iter = 5,
                         tolower = FALSE, alpha = 0.05)
mat_wdv <- as.matrix(wdv, layer = "documents", normalize = FALSE)
hist(mat_wdv["4362315",, drop = TRUE])

sim_wdv <- proxyC::simil(
    mat_wdv,
    mat_wdv["4362315",, drop = FALSE]
)

sim_wdv <- proxyC::simil(
    mat_wdv,
    mat_wdv["4102068",, drop = FALSE]
)

hist(rowSums(sim_wdv))
tail(sort(s <- rowSums(sim_wdv)))
tail(dat3[order(s),])

sim_wdv_all <- proxyC::simil(
    mat_wdv,
    mat_wdv[tail(dat3$doc_id, 100),, drop = FALSE]
)
mean(diag(sim_wdv_all)) - mean(sim_wdv_all)

# -------------------------

# a <- rownames(as.matrix(d2v, which = "words", normalize = FALSE))
# b <- rownames(as.matrix(wdv, layer = "words", normalize = FALSE))
# length(a)
# length(b)
# length(intersect(a, b)) / length(union(a, b))

plot(rowSums(sim_wdv), rowSums(sim_d2v))
cor(rowSums(sim_wdv), rowSums(sim_d2v))


       