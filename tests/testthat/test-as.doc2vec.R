library(quanteda)
library(wordvector)
options(wordvector_threads = 2)

corp <- head(data_corpus_inaugural, 59) 

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE,
               concatenator = " ") %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_compound(data_dictionary_LSD2015, keep_unigrams = TRUE)

dfmt <- dfm(toks, remove_padding = TRUE) 

set.seed(1234)
wov <- textmodel_word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)

test_that("textmodel_doc2vec works", {
    
    dov1 <- as.textmodel_doc2vec(dfmt, wov)
    expect_false(dov1$normalize)
    expect_equal(
        names(dov1),
        c("values", "dim", "concatenator", "docvars", "normalize", "call", "version")
    )
    expect_equal(
        dim(dov1$values$word), c(5363L, 50L)
    )
    expect_equal(
        dim(dov1$values$doc), c(59L, 50L)
    )
    expect_equal(
        class(dov1), c("textmodel_doc2vec", "textmodel_wordvector")
    )
    expect_output(
        print(dov1),
        paste(
            "",
            "Call:",
            "as.textmodel_doc2vec(x = dfmt, model = wov)",
            "",
            "50 dimensions; 59 documents.", sep = "\n"), fixed = TRUE
    )
    
    # normalize
    dov2 <- as.textmodel_doc2vec(dfmt, wov, normalize = TRUE)
    expect_false(identical(dov1$values, dov2$values))
    expect_true(dov2$normalize)
    
    # weights
    w <- abs(rnorm(nrow(wov$values$word)))
    dov3 <- as.textmodel_doc2vec(dfmt, wov, weights = w)
    expect_false(identical(dov1$values, dov3$values))
    
    # pattern
    dov4 <- as.textmodel_doc2vec(dfmt, wov, weights = 2.0, 
                                 pattern = data_dictionary_LSD2015)
    expect_false(identical(dov1$values$doc, dov4$values$doc))
    
    dict <- dictionary(list(hard = "hard *"))
    dov5 <- as.textmodel_doc2vec(dfmt, wov, weights = 2.0, 
                                 pattern = dict)
    expect_false(identical(dov1$values$doc, dov5$values$doc))
    
    # errors
    expect_error(
        as.textmodel_doc2vec(dfmt, wov, weights = c(0.1, 0.2, 0.2)),
        "The length of weights must be 5363"
    )
    
    w <- sample(c(1.0, NA), 5363, replace = TRUE)
    expect_error(
        as.textmodel_doc2vec(dfmt, wov, weights = w),
        "The value of weights cannot be NA"
    )
    
    expect_error(
        as.textmodel_doc2vec(dfmt, wov, weights = -0.1),
        "The value of weights must be between 0 and Inf"
    )

})

test_that("as.textmodel_doc2vec works with different objects", {
    
    expect_equal(
        class(as.textmodel_doc2vec(dfmt, wov)),
        c("textmodel_doc2vec", "textmodel_wordvector")
    )
    
    expect_error(
        as.textmodel_doc2vec(toks, wov),
        "no applicable method for 'as.textmodel_doc2vec'"
    )
    
    expect_error(
        as.textmodel_doc2vec(dfmt, list()),
        "'model' must be a trained textmodel_word2vec"
    )
})

test_that("textmodel_doc2vec returns zero for emptry documents (#17)", {
    toks <- tokens(c("Citizens of the United States", "")) %>% 
        tokens_tolower()
    dfmt <- dfm(toks)
    dov <- as.textmodel_doc2vec(dfmt, wov)
    expect_true(all(dov$values$doc[1,] != 0))
    expect_true(all(dov$values$doc[2,] == 0))
})
