
#' @export
as.matrix.textmodel_docvector <- function(x, ...){
    return(x$values) 
}

#' Create distributed representation of documents
#' 
#' Create distributed representation of documents as weighted word vectors.
#' @param x a [quanteda::tokens] object.
#' @param model a textmodel_wordvector object.
#' @param group_data if `TRUE`, apply `dfm_group(x)` before creating document vectors.
#' @param ... passed to `[textmodel_word2vec]` when `model = NULL`.
#' @returns Returns a textmodel_docvector object with elements inherited from `model` 
#'   or passed via `...` plus:
#'   \item{values}{a matrix for document vectors.}
#'   \item{call}{the command used to execute the function.}
#' @export
textmodel_doc2vec <- function(x, model = NULL, group_data = FALSE,
                              ...) {
    UseMethod("textmodel_doc2vec")
}

#' @export
#' @method textmodel_doc2vec tokens
textmodel_doc2vec.tokens <- function(x, model = NULL, group_data = FALSE, 
                                     ...) {

    result <- textmodel_doc2vec(dfm(x, remove_padding = TRUE), model = model,
                                group_data = group_data, ...)
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}

#' @export
#' @method textmodel_doc2vec dfm
textmodel_doc2vec.dfm <- function(x, model = NULL, group_data = FALSE, 
                                     ...) {
    if (is.null(model)) {
        model <- textmodel_word2vec(x, ...)
    } else {
        if (!identical(class(model), "textmodel_wordvector"))
            stop("The object for 'model' must be a trained textmodel_wordvector")
    }
    if (group_data)
        x <- dfm_group(x)
    
    wov <- as.matrix(model)
    x <- dfm_match(x, rownames(wov))
    
    l <- rowSums(x) == 0
    dov <- Matrix::tcrossprod(x, t(wov)) # NOTE: consider using proxyC::prod
    dov <- dov / sqrt(Matrix::rowSums(dov ^ 2) / ncol(dov))
    dov[l,] <- 0
    
    result <- list(
        "values" = dov,
        "dim" = ncol(dov),
        "concatenator" = meta(x, field = "concatenator", type = "object"), 
        "docvars" = docvars(x),
        "call" = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE), 
        "version" = utils::packageVersion("wordvector")
    )
    class(result) <- "textmodel_docvector"
    return(result)
}

doc2vec <- function(...) {
    .Deprecated("textmodel_doc2vec")
    textmodel_doc2vec(...)
}

