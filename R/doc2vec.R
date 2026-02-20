#' Doc2vec model
#' 
#' Train a doc2vec model (Le & Mikolov, 2014) using a [quanteda::tokens] object.
#' @export
#' @param type the architecture of the model; either "dm" (distributed memory) or 
#'   "dbow" (distributed bag-of-words).
#' @param window the size of the window for context words. Ignored when `type = "dbow"` as
#'   its context window is the entire document (sentence or paragraph).
#' @inheritParams textmodel_word2vec
#' @return 
#' Returns a textmodel_doc2vec object with matrices for word and document vector values in `values`.
#' Other elements are the same as [wordvector::textmodel_word2vec].
#' @references 
#'   Le, Q. V., & Mikolov, T. (2014). Distributed Representations of Sentences and 
#'   Documents (No. arXiv:1405.4053). arXiv. https://doi.org/10.48550/arXiv.1405.4053
textmodel_doc2vec <- function(x, dim = 50, type = c("dm", "dbow"), 
                              min_count = 5, window = 5, 
                              iter = 10, alpha = 0.05, model = NULL, 
                              use_ns = TRUE, ns_size = 5, sample = 0.001, tolower = TRUE,
                              include_data = FALSE, verbose = FALSE, ...) {
    UseMethod("textmodel_doc2vec")
}

#' @export
#' @method textmodel_doc2vec tokens
textmodel_doc2vec.tokens <- function(x, dim = 50, type = c("dm", "dbow"), 
                                     min_count = 5, window = 5, 
                                     iter = 10, alpha = 0.05, model = NULL, 
                                     use_ns = TRUE, ns_size = 5, sample = 0.001, tolower = TRUE,
                                     include_data = FALSE, verbose = FALSE, ...) {
    
    type <- match.arg(type)
    wordvector(x, dim, type, TRUE, min_count, window, iter, alpha, model, 
               use_ns, ns_size, sample, tolower, include_data, verbose, ...)
    
}

#' @rdname as.matrix
#' @export
as.matrix.textmodel_doc2vec <- function(x, normalize = TRUE, 
                                        layer = c("documents", "words"), ...) {
    
    x <- upgrade_pre06(x)
    normalize <- check_logical(normalize)
    layer <- match.arg(layer)
    
    if (layer == "words") {
        result <- x$values$word
    } else {
        # TODO: add grouping by docid
        result <- x$values$doc
    }
    if (is.null(result))
        stop("x does not have the layer for ", layer)
    if (normalize) {
        v <- sqrt(rowSums(result ^ 2) / ncol(result))
        result <- result / v
    }
    return(result) 
}

