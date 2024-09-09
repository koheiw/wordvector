library(quanteda)
library(wordvector)
library(LSX)

#data_corpus_guardian <- readRDS("/home/kohei/Dropbox/Public/data_corpus_guardian2016-10k.rds")
#corp <- data_corpus_guardian %>% 
corp <- data_corpus_inaugural %>% 
    corpus_reshape()
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_tolower()
ndoc(toks)

dfmt <- dfm(toks, remove_padding = TRUE) %>% 
    dfm_trim(min_termfreq = 5)
# lss0 <- textmodel_lss(dfmt, c("good" = 1, "bad" = -1), cache = TRUE)
# head(coef(lss0), 20)
# tail(coef(lss0), 20)
lss0 <- textmodel_lss(dfmt, "bad", cache = TRUE)
head(coef(lss0), 20)

set.seed(1234)
mod <- word2vec(toks, dim = 100, iter = 5, min_count = 5, type = "skip-gram",
                verbose = TRUE, threads = 4)
#lss <- as.textmodel_lss(t(as.matrix(mod)), c("good" = 1, "bad" = -1))
#head(coef(lss), 20)
#tail(coef(lss), 20)
lss <- as.textmodel_lss(t(as.matrix(mod)), "bad")
head(coef(lss), 20)

pred <- predict(mod, c("good", "bad"), type = "nearest")
pred
