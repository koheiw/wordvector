library(quanteda)
library(wordvector)
options(wordvector_threads = 2)

corp <- head(data_corpus_inaugural, 59) %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE,
               concatenator = " ") %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_compound(data_dictionary_LSD2015, keep_unigrams = TRUE)

set.seed(1234)
wov <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)

test_that("textmodel_doc2vec works", {
    
    dov1 <- textmodel_doc2vec(toks, wov)
    expect_true(dov1$normalize)
    expect_equal(
        names(dov1),
        c("values", "dim", "concatenator", "docvars", "normalize", "call", "version")
    )
    expect_equal(
        dim(dov1$values), c(5234L, 50L)
    )
    expect_equal(class(dov1$values), c("matrix", "array"))
    expect_equal(
        class(dov1), "textmodel_docvector"
    )
    expect_output(
        print(dov1),
        paste(
            "",
            "Call:",
            "textmodel_doc2vec(x = toks, model = wov)",
            "",
            "50 dimensions; 5,234 documents.", sep = "\n"), fixed = TRUE
    )
    
    # normalize
    dov2 <- textmodel_doc2vec(toks, wov, normalize = FALSE)
    expect_false(identical(dov1$values, dov2$values))
    expect_false(dov2$normalize)
    
    # weights
    w <- abs(rnorm(nrow(wov$values)))
    dov3 <- textmodel_doc2vec(toks, wov, weights = w)
    expect_false(identical(dov1$values, dov3$values))
    
    # pattern
    dov4 <- textmodel_doc2vec(toks, wov, weights = 2.0, 
                              pattern = data_dictionary_LSD2015)
    expect_false(identical(dov1$values, dov4$values))
    
    dict <- dictionary(list(hard = "hard *"))
    dov5 <- textmodel_doc2vec(toks, wov, weights = 2.0, 
                              pattern = dict)
    expect_false(identical(dov1$values, dov5$values))
    
    # errors
    expect_error(
        textmodel_doc2vec(toks, wov, weights = c(0.1, 0.2, 0.2)),
        "The length of weights must be 5363"
    )
    
    w <- sample(c(1.0, NA), 5363, replace = TRUE)
    expect_error(
        textmodel_doc2vec(toks, wov, weights = w),
        "The value of weights cannot be NA"
    )
    
    expect_error(
        textmodel_doc2vec(toks, wov, weights = -0.1),
        "The value of weights must be between 0 and Inf"
    )

})

test_that("textmodel_doc2vec works with different objects", {
    
    expect_equal(
        class(textmodel_doc2vec(toks, wov)),
        "textmodel_docvector"
    )
    
    expect_equal(
        class(textmodel_doc2vec(as.tokens_xptr(toks), wov)),
        "textmodel_docvector"
    )
    
    expect_error(
        textmodel_doc2vec(toks, list),
        "The object for 'model' must be a trained textmodel_wordvector"
    )
})

test_that("grouped_data works", {
    
    dov_gp <- textmodel_doc2vec(toks, wov, group_data = TRUE)
    
    expect_identical(
        dim(dov_gp$values), c(59L, 50L)
    )
    expect_equal(
        class(dov_gp), "textmodel_docvector"
    )
    expect_equal(
        names(dov_gp),
        c("values", "dim", "concatenator", "docvars", "normalize", "call", "version")
    )
    
})

test_that("old and new produce similar results", {
    
    dfmt <- dfm(toks) %>% 
        dfm_group()
    dov0 <- textmodel_doc2vec(dfmt, wov, old = TRUE)
    dov1 <- textmodel_doc2vec(dfmt, wov)
    dov2 <- textmodel_doc2vec(dfmt, wov, normalize = FALSE)
    
    expect_false(identical(dov0$values, dov1$values))
    expect_false(identical(dov1$values, dov2$values))
    
    expect_lte(sd(dov0$values), 1)
    expect_lte(sd(dov1$values), 1)
    expect_gte(sd(dov2$values), 100)
    
    expect_equal(
        cor(t(dov0$values[1:5,]), t(dov0$values[1:5,])),
        cor(t(dov1$values[1:5,]), t(dov1$values[1:5,]))
    )
    
    expect_equal(
        cor(t(dov0$values[1:5,]), t(dov0$values[1:5,])),
        cor(t(dov2$values[1:5,]), t(dov2$values[1:5,]))
    )
    
})

test_that("textmodel_doc2vec returns zero for emptry documents (#17)", {
    toks <- tokens(c("Citizens of the United States", "")) %>% 
        tokens_tolower()
    dov <- textmodel_doc2vec(toks, wov)
    expect_true(all(dov$values[1,] != 0))
    expect_true(all(dov$values[2,] == 0))
})
