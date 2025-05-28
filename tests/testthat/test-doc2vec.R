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

test_that("textmodeldoc2vec works", {
    
    dov1 <- textmodel_doc2vec(toks, wov)
    expect_false(dov1$normalize)
    expect_equal(
        names(dov1),
        c("values", "dim", "concatenator", "docvars", "normalize", "call", "version")
    )
    expect_equal(
        dim(dov1$values), c(5234L, 50L)
    )
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
    dov2 <- textmodel_doc2vec(toks, wov, normalize = TRUE)
    expect_false(identical(dov1$values, dov2$values))
    expect_true(dov2$normalize)
    
    # weight
    w <- abs(rnorm(nrow(wov$values)))
    dov3 <- textmodel_doc2vec(toks, wov, weight = w)
    expect_false(identical(dov1$values, dov3$values))
    
    # pattern
    dov4 <- textmodel_doc2vec(toks, wov, weight = 2.0, 
                              pattern = data_dictionary_LSD2015)
    expect_false(identical(dov1$values, dov4$values))
    
    dict <- dictionary(list(hard = "hard *"))
    dov5 <- textmodel_doc2vec(toks, wov, weight = 2.0, 
                              pattern = dict)
    expect_false(identical(dov1$values, dov5$values))
    
    # errors
    expect_error(
        textmodel_doc2vec(toks, wov, weight = c(0.1, 0.2, 0.2)),
        "The length of weight must be 5363"
    )
    
    w <- sample(c(1.0, NA), 5363, replace = TRUE)
    expect_error(
        textmodel_doc2vec(toks, wov, weight = w),
        "The value of weight cannot be NA"
    )
    
    expect_error(
        textmodel_doc2vec(toks, wov, weight = -0.1),
        "The value of weight must be between 0 and Inf"
    )

})


test_that("textmodel_doc2vec returns zero for emptry documents (#17)", {
    toks <- tokens(c("Citizens of the United States", "")) %>% 
        tokens_tolower()
    dov <- textmodel_doc2vec(toks, wov)
    expect_true(all(dov$values[1,] != 0))
    expect_true(all(dov$values[2,] == 0))
})

