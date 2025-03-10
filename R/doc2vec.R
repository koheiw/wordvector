
#' @export
as.matrix.textmodel_docvector <- function(x, ...){
    return(x$values) 
}

#' Create distributed representation of documents
#' 
#' Create distributed representation of documents as weighted word vectors.
#' @param x a [quanteda::tokens] object.
#' @param model a textmodel_wordvector object.
#' @param ... passed to `[word2vec]` when `model = NULL`.
#' @returns Returns a textmodel_docvector object with elements inherited from `model` 
#'   or passed via `...` plus:
#'   \item{values}{a matrix for document vectors.}
#'   \item{call}{the command used to execute the function.}
#' @export
textmodel_doc2vec <- function(x, model = NULL, ...) {
    UseMethod("textmodel_doc2vec")
}

#' @export
#' @method textmodel_doc2vec tokens
textmodel_doc2vec.tokens <- function(x, model = NULL, ...) {
    if (is.null(model)) {
        model <- textmodel_word2vec(x, ...)
    } else {
        if (!identical(class(model), "textmodel_wordvector"))
            stop("The object for 'model' must be a trained textmodel_wordvector")
    }
    result <- model
    wov <- as.matrix(model)
    dfmt <- dfm_match(dfm(x, remove_padding = TRUE), rownames(wov))
    empty <- rowSums(dfmt) == 0
    dov <- Matrix::tcrossprod(dfmt, t(wov)) # NOTE: consider using proxyC::prod
    result$values <- dov / sqrt(Matrix::rowSums(dov ^ 2) / ncol(dov))
    result$values[empty,] <- 0
    class(result) <- "textmodel_docvector"
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}

doc2vec <- function(...) {
    .Deprecated("textmodel_doc2vec")
    textmodel_doc2vec(...)
}

