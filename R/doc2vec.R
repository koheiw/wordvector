
#' @export
as.matrix.textmodel_docvector <- function(x, ...){
    return(x$vectors) 
}

#' Create distributed representation of documents

#' @param x a [quanteda::tokens] object.
#' @param model a textmodel_wordvector object.
#' @param ... passed to `[word2vec]` when `model = NULL`.
#' @returns Returns a textmodel_docvector object with elements inherited from `model` 
#'   or passed via `...` plus:
#'   \item{vectors}{a matrix for document vectors.}
#'   \item{call}{the command used to execute the function.}
#' @export
doc2vec <- function(x, model = NULL, ...) {
    UseMethod("doc2vec")
}

#' @export
#' @method doc2vec tokens
doc2vec.tokens <- function(x, model = NULL, ...) {
    if (is.null(model)) {
        model <- word2vec(x, ...)
    } else {
        if (!identical(class(model), "textmodel_wordvector"))
            stop("The object for 'model' must be a trained textmodel_wordvector")
    }
    result <- model
    wov <- as.matrix(model)
    dfmt <- dfm_match(dfm(x, remove_padding = TRUE), rownames(wov))
    dov <- Matrix::tcrossprod(dfmt, t(wov)) # NOTE: consider using proxyC::prod
    result$vectors <- dov / sqrt(Matrix::rowSums(dov ^ 2) / ncol(dov))
    class(result) <- "textmodel_docvector"
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}
