#' @export
lsa <- function(x, dim = 50, min_count = 5L, engine = "RSpectra", weight = "count", verbose = FALSE, ...) {
    x <- tokens_trim(x, min_termfreq = min_count, termfreq_type = "count")
    x <- dfm(x, remove_padding = TRUE)
    if (engine %in% c("RSpectra", "irlba", "rsvd")) {
        if (verbose)
            cat(sprintf("Performing SVD by %s...\n", engine))
        svd <- get_svd(x, dim, engine, weight, ...)
        emb <- svd$v
        rownames(emb) <- featnames(x)
    }
    result <- list(
        model = emb,
        min_count = min_count,
        concatenator = meta(x, field = "concatenator", type = "object"),
        call = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    )
    class(result) <- "textmodel_wordvector"
    return(result)
}

get_svd <- function(x, k, engine, weight = "count", ...) {
    x <- quanteda::dfm_weight(x, scheme = weight)
    if (engine == "RSpectra") {
        result <- RSpectra::svds(as(x, "dgCMatrix"), k = k, nu = 0, nv = k, ...)
    } else if (engine == "rsvd") {
        result <- rsvd::rsvd(as(x, "dgCMatrix"), k = k, nu = 0, nv = k, ...)
    } else {
        result <- irlba::irlba(as(x, "dgCMatrix"), nv = k, right_only = TRUE, ...)
    }
    return(result)
}
