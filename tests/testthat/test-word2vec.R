library(quanteda)
library(wordvector)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()
toks_grp <- tokens_group(toks)

wdv <- wordvector::word2vec(toks, dim = 50, iter = 20, min_count = 0, 
                            verbose = FALSE, sample = 0)

test_that("analogy works", {
    
    ana1 <- analogy(wdv, ~ us)
    expect_true(ana1$word[1] == "us")
    expect_true(ana1$similarity[1] == 1.0)
    expect_identical(attr(ana1, "weight"), 
                     c("us" = 1))
    
    ana2 <- analogy(wdv, ~ us - people)
    expect_true(ana2$word[1] == "us")
    expect_true(ana2$similarity[1] < 1.0)
    expect_identical(attr(ana2, "weight"), 
                     c("us" = 1, "people" = -1))
    
})

test_that("synonyms works", {
    
    syno1 <- synonyms(wdv, c("us"))
    expect_true(is.matrix(syno1))
    expect_identical(
        syno1[1,],
        c("us" = "us")
    )
    
    syno2 <- synonyms(wdv, c("us", "people"))
    expect_true(is.matrix(syno2))
    expect_identical(
        syno2[1,],
        c("us" = "us", "people" = "people")
    )
})
