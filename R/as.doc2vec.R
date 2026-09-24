#' Create a doc2vec model
#' 
#' Create a doc2vec model as weighted word vectors.
#' @param x a [quanteda::tokens] or [quanteda::dfm] object.
#' @param model a textmodel_wordvector object.
#' @param normalize if `TRUE`, normalized word vectors before creating document vectors.
#' @param compound if `TRUE`, compound multi-word expressions in `x` based on `model` 
#'   internally. Only applies when `x` is a [quanteda::tokens] object.
#' @param group_data if `TRUE`, apply `dfm_group(x)` before creating document vectors.
#' @param ... additional arguments passed to [quanteda::object2id].
#' @details
#' @returns Returns a textmodel_doc2vec object with the following elements:
#'   \item{values}{a list of matrices for word and document vectors.}
#'   \item{dim}{the size of the document vectors.}
#'   \item{concatenator}{the concatenator in `x`.}
#'   \item{docvars}{document variables copied from `x`.}
#'   \item{normalize}{if the document vectors are normalized.}
#'   \item{call}{the command used to execute the function.}
#'   \item{version}{the version of the wordvector package.}
#' @export
as.textmodel_doc2vec <- function(x, model, normalize = FALSE, 
                                 compound = TRUE, group_data = FALSE, ...) {
    UseMethod("as.textmodel_doc2vec")
}

#' @export
#' @method as.textmodel_doc2vec tokens
as.textmodel_doc2vec.tokens <- function(x, model = NULL, normalize = FALSE, 
                                        compound = TRUE, group_data = FALSE, ...) {
    
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
    x <- dfm(x, remove_padding = TRUE, tolower = model$tolower)
    result <- as.textmodel_doc2vec(x, model = model, normalize = normalize, 
                                   group_data = group_data, ...)
    result$call = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}

#' @export
#' @method as.textmodel_doc2vec dfm
as.textmodel_doc2vec.dfm <- function(x, model = NULL, normalize = FALSE, 
                                     compound = TRUE, group_data = FALSE, ...) {
    
    model <- upgrade_pre06(model)
    model <- check_model(model, c("word2vec", "doc2vec", "lsa"))
    conc <- meta(x, field = "concatenator", type = "object")

    wov <- as.matrix(model, normalize, layer = "words")
    if (group_data)
        x <- dfm_group(x)
    x <- dfm_match(x, rownames(wov))
    
    l <- rowSums(x) == 0
    dov <- as.matrix(Matrix::tcrossprod(x, t(wov))) # NOTE: consider using proxyC::prod
    dov <- dov / sqrt(rowSums(dov ^ 2) / ncol(dov))
    dov[l,] <- 0
    
    result <- list(
        "values" = list("word" = wov, "doc" = dov),
        "weights" = model$weights,
        "dim" = model$dim,
        "tolower" = model$tolower,
        "concatenator" = conc, 
        "docvars" = x@docvars,
        "normalize" = normalize,
        "call" = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE), 
        "version" = utils::packageVersion("wordvector")
    )
    class(result) <- c("textmodel_doc2vec", "textmodel_wordvector")
    return(result)
}

