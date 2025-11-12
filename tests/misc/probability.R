library(quanteda)
library(wordvector)
options(wordvector_threads = 8)

seed <- LSX::seedwords("sentiment")

corp <- data_corpus_news2014
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>%
    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>%
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                  padding = TRUE) %>%
    tokens_tolower()

wov <- textmodel_word2vec(toks, dim = 100, type = "sg", min_count = 5, verbose = TRUE, iter = 10)
dov <- textmodel_doc2vec(toks, dim = 100, type = "dbow", min_count = 5, verbose = TRUE, iter = 10)

doc <- probability(dov, seed, layer = "documents")[,1]
print(toks[head(doc)])
print(toks[tail(doc)])

cor(probability(wov, seed, layer = "words", mode = "numeric")[,1],
    probability(dov, seed, layer = "words", mode = "numeric")[,1])

lss0 <- LSX::as.textmodel_lss(wov, seed, spatial = FALSE)
lss <- LSX:::as.textmodel_lss(dov, seed, spatial = FALSE)
pred0 <- predict(lss0, newdata = dfm(toks, remove_padding = TRUE), rescale = FALSE,
                 min_n = 40)
pred <- predict(lss, newdata = dfm(toks, remove_padding = TRUE), rescale = FALSE,
                 min_n = 40)
pred2 <- probability(dov, seed, layer = "documents", mode = "numeric")[,1] / length(seed)

dat <- data.frame(pred0, pred, pred2, pred3 = pred + pred2, text = corp[])

plot(dat[,1:4])
cor(dat[,1:4])
View(dat)
head(dat)
