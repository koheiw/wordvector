#' @export
lsa <- function(x, dim = 50, min_count = 5L, engine = c("RSpectra", "irlba", "rsvd"), 
                weight = "count", verbose = FALSE, ...) {
    UseMethod("lsa")   
}

#' @export
lsa.tokens <- function(x, dim = 50, min_count = 5L, engine = c("RSpectra", "irlba", "rsvd"), 
                       weight = "count", verbose = FALSE, ...) {
    
    engine <- match.arg(engine)
    x <- tokens_trim(x, min_termfreq = min_count, termfreq_type = "count")
    x <- dfm(x, remove_padding = TRUE)
    if (engine %in% c("RSpectra", "irlba", "rsvd")) {
        if (verbose) {
            cat(sprintf("Performing SVD into %d dimensions\n", dim))
            cat(sprintf("...using %s\n", engine))
        }
        svd <- get_svd(x, dim, engine, weight, ...)
        if (verbose)
            cat("...complete\n")
        wov <- svd$v
        rownames(wov) <- featnames(x)
    }
    result <- list(
        model = wov,
        dim = dim,
        min_count = min_count,
        frequency = featfreq(x),
        engine = engine,
        weight = weight,
        concatenator = meta(x, field = "concatenator", type = "object"),
        call = try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    )
    class(result) <- "textmodel_wordvector"
    return(result)
}

get_svd <- function(x, k, engine, weight = "count", reduce = FALSE, ...) {
    if (reduce) {
        x <- quanteda::dfm_weight(x, weights = 1 / sqrt(featfreq(x)))
    } else {
        if (weight == "sqrt") {
            x@x <- sqrt(x@x)
        } else {
            x <- quanteda::dfm_weight(x, scheme = weight)
        }
    }
    if (engine == "RSpectra") {
        result <- RSpectra::svds(as(x, "dgCMatrix"), k = k, nu = 0, nv = k, ...)
    } else if (engine == "rsvd") {
        result <- rsvd::rsvd(as(x, "dgCMatrix"), k = k, nu = 0, nv = k, ...)
    } else {
        result <- irlba::irlba(as(x, "dgCMatrix"), nv = k, right_only = TRUE, ...)
    }
    return(result)
}
