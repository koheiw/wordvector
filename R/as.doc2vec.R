#' Create a doc2vec model
#' 
#' Create a doc2vec model as weighted word vectors.
#' @param x a [quanteda::tokens] or [quanteda::dfm] object.
#' @param model a textmodel_wordvector object.
#' @param compound if `TRUE`, compound multi-word expressions in `x` based on `model` 
#'   internally. Only applies when `x` is a [quanteda::tokens] object.
#' @param group_data if `TRUE`, apply `dfm_group(x)` before creating document vectors.
#' @param ... additional arguments passed to the underlying function.
#' @details
#' For Japanese or Chinese texts, `model$concatenator` must be empty (""). 
#' The value is inherited from the tokens object on which the model was trained. 
#' It triggers tokenization of words in `model` and compounding of characters in `x` 
#' before creating document vectors.
#' @returns Returns a textmodel_doc2vec object with the following elements:
#'   \item{values}{a list of matrices for word and document vectors.}
#'   \item{dim}{the size of the document vectors.}
#'   \item{concatenator}{the concatenator in `x`.}
#'   \item{docvars}{document variables copied from `x`.}
#'   \item{call}{the command used to execute the function.}
#'   \item{version}{the version of the wordvector package.}
#' @export
as.textmodel_doc2vec <- function(x, model, 
                                 compound = TRUE, group_data = FALSE, ...) {
    UseMethod("as.textmodel_doc2vec")
}

#' @export
#' @method as.textmodel_doc2vec tokens
as.textmodel_doc2vec.tokens <- function(x, model, compound = TRUE, 
                                        group_data = FALSE, ...) {
    
    wov <- as.matrix(model, FALSE, layer = "words")
    compound <- check_logical(compound)
    
    if (compound) {
        conc <- model$concatenator
        if (identical(conc, "")) {
            p <- as.list(tokens(rownames(wov)))
        } else {
            p <- phrase(rownames(wov), conc)
        }
        x <- tokens_compound(x, p, valuetype = "fixed", join = FALSE, 
                             concatenator = conc)
    }
    x <- dfm(x, tolower = model$tolower)
    result <- as.textmodel_doc2vec(x, model = model, 
                                   group_data = group_data, ...)
    result$call = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}

#' @export
#' @method as.textmodel_doc2vec dfm
as.textmodel_doc2vec.dfm <- function(x, model, compound = TRUE, 
                                     group_data = FALSE, ...) {
    
    model <- upgrade_pre06(model)
    model <- check_model(model, c("word2vec", "doc2vec", "lsa"))
    conc <- meta(x, field = "concatenator", type = "object")

    wov <- as.matrix(model, normalize = FALSE, layer = "words")
    if (group_data)
        x <- dfm_group(x)
    x <- dfm_match(x, rownames(wov))
    dov <- as.matrix(Matrix::tcrossprod(x, t(wov)))
    dov <- normalize(dov)
    
    result <- build_doc2vec(
        docname = docnames(x),
        model = model,
        values = list("word" = wov, "doc" = dov),
        weights = NULL,
        frequency = featfreq(x),
        concatenator = conc, 
        docvars = x@docvars,
        normalize = FALSE,
        call = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE),
        ...
    )
    return(result)
}

#' @export
#' @method as.textmodel_doc2vec matrix
as.textmodel_doc2vec.matrix <- function(x, ...) {
    
    if (is.null(rownames(x)))
        stop("x must have rownames for documents")
    if (!is.numeric(x) || any(is.na(x)))
        stop("x must be a numeric matrix without NA")
    colnames(x) <- NULL
    
    result <- build_doc2vec(
        docname = rownames(x),
        values = list("doc" = x),
        weights = NULL,
        dim = ncol(x),
        normalize = FALSE,
        call = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE),
        ...
    )
    class(result) <- c("textmodel_doc2vec", "textmodel_wordvector")
    return(result)
}

