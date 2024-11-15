library(quanteda)
library(wordvector)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()
toks_grp <- tokens_group(toks)

wov <- word2vec(toks, dim = 50, iter = 10, min_count = 0, sample = 0)
dov <- doc2vec(toks_grp, wov)
dov_nm <- doc2vec(toks_grp, min_count = 10, sample = 0)

test_that("word2vec words", {
    
    # wordvector
    expect_equal(
        class(wov), "textmodel_wordvector"
    )
    expect_equal(
        dim(wov$model), c(9280, 50)
    )
    expect_equal(
        wov$sample, 0
    )
    expect_equal(
        wov$min_count, 0L
    )
    
    # docvector with model
    expect_equal(
        dim(dov$model), c(59, 50)
    )
    expect_equal(
        class(dov), "textmodel_docvector"
    )
    expect_equal(
        dov$sample, 0
    )
    expect_equal(
        dov$min_count, 0L
    )
    
    # docvector without model
    expect_equal(
        dim(dov_nm$model), c(59, 50)
    )
    expect_equal(
        class(dov_nm), "textmodel_docvector"
    )
    expect_equal(
        dov_nm$sample, 0
    )
    expect_equal(
        dov_nm$min_count, 10L
    )
})

test_that("analogy works", {
    
    ana1 <- analogy(wov, ~ us)
    expect_true(ana1$word[1] == "us")
    expect_true(ana1$similarity[1] == 1.0)
    expect_identical(attr(ana1, "weight"), 
                     c("us" = 1))
    
    ana2 <- analogy(wov, ~ people - us)
    expect_true(ana2$word[1] != "us")
    expect_true(ana2$similarity[1] < 1.0)
    expect_identical(attr(ana2, "weight"), 
                     c("people" = 1, "us" = -1))
    
})

test_that("synonyms works", {
    
    syno1 <- synonyms(wov, c("us"))
    expect_true(is.matrix(syno1))
    expect_identical(
        syno1[1,],
        c("us" = "us")
    )
    
    syno2 <- synonyms(wov, c("us", "people"))
    expect_true(is.matrix(syno2))
    expect_identical(
        syno2[1,],
        c("us" = "us", "people" = "people")
    )
})

test_that("doc2vec works with different objects", {
    
    expect_equal(
        class(doc2vec(toks, wov)),
        "textmodel_docvector"
    )
    
    expect_equal(
        class(doc2vec(as.tokens_xptr(toks), wov)),
        "textmodel_docvector"
    )
    
    expect_error(
        doc2vec(toks, list),
        "The object for 'model' must be a trained textmodel_wordvector"
    )
})
