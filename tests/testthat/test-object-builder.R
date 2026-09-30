
test_that("dots works with object builders", {
    
    # word2vec
    mat <- matrix(rnorm(9), nrow = 3, dimnames = list(c("a", "b", "c")))
    wov1 <- as.textmodel_word2vec(mat, frequency = c("a" = 10, "b" = 20, "c" = 5), 
                                  tolower = FALSE, dim = 10, xxx = "none")
    expect_equal(
        wov1$frequency,
        c("a" = 10, "b" = 20, "c" = 5)
    )
    expect_false(
        wov1$tolower
    )
    expect_equal(
        wov1$dim,
        3
    )
    expect_null(
        wov1$xxx
    )
    
    # doc2vec
    mat <- matrix(rnorm(9), nrow = 3, dimnames = list(c("doc1", "doc2", "doc3")))
    dov1 <- as.textmodel_doc2vec(mat, concatenator = "+", tolower = FALSE, dim = 10,
                                 xxx = "none")
    
    expect_equal(
        dov1$concatenator,
        "+"
    )
    expect_false(
        dov1$tolower
    )
    expect_equal(
        dov1$dim,
        3
    )
    expect_null(
        wov1$xxx
    )
    
})
