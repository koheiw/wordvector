// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// cpp_w2v
Rcpp::List cpp_w2v(Rcpp::List texts_, Rcpp::CharacterVector types_, std::string modelFile, uint16_t size, uint8_t window, uint16_t expTableSize, uint8_t expValueMax, float sample, bool withHS, uint8_t negative, uint8_t threads, uint8_t iterations, float alpha, bool withSG, bool verbose, bool normalize);
RcppExport SEXP _wordvector_cpp_w2v(SEXP texts_SEXP, SEXP types_SEXP, SEXP modelFileSEXP, SEXP sizeSEXP, SEXP windowSEXP, SEXP expTableSizeSEXP, SEXP expValueMaxSEXP, SEXP sampleSEXP, SEXP withHSSEXP, SEXP negativeSEXP, SEXP threadsSEXP, SEXP iterationsSEXP, SEXP alphaSEXP, SEXP withSGSEXP, SEXP verboseSEXP, SEXP normalizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type texts_(texts_SEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type types_(types_SEXP);
    Rcpp::traits::input_parameter< std::string >::type modelFile(modelFileSEXP);
    Rcpp::traits::input_parameter< uint16_t >::type size(sizeSEXP);
    Rcpp::traits::input_parameter< uint8_t >::type window(windowSEXP);
    Rcpp::traits::input_parameter< uint16_t >::type expTableSize(expTableSizeSEXP);
    Rcpp::traits::input_parameter< uint8_t >::type expValueMax(expValueMaxSEXP);
    Rcpp::traits::input_parameter< float >::type sample(sampleSEXP);
    Rcpp::traits::input_parameter< bool >::type withHS(withHSSEXP);
    Rcpp::traits::input_parameter< uint8_t >::type negative(negativeSEXP);
    Rcpp::traits::input_parameter< uint8_t >::type threads(threadsSEXP);
    Rcpp::traits::input_parameter< uint8_t >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< float >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< bool >::type withSG(withSGSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type normalize(normalizeSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_w2v(texts_, types_, modelFile, size, window, expTableSize, expValueMax, sample, withHS, negative, threads, iterations, alpha, withSG, verbose, normalize));
    return rcpp_result_gen;
END_RCPP
}
// w2v_dictionary
std::vector<std::string> w2v_dictionary(SEXP ptr);
RcppExport SEXP _wordvector_w2v_dictionary(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(w2v_dictionary(ptr));
    return rcpp_result_gen;
END_RCPP
}
// w2v_embedding
Rcpp::NumericMatrix w2v_embedding(SEXP ptr, Rcpp::StringVector x);
RcppExport SEXP _wordvector_w2v_embedding(SEXP ptrSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(w2v_embedding(ptr, x));
    return rcpp_result_gen;
END_RCPP
}
// w2v_nearest
Rcpp::DataFrame w2v_nearest(SEXP ptr, std::string x, std::size_t top_n, float min_distance);
RcppExport SEXP _wordvector_w2v_nearest(SEXP ptrSEXP, SEXP xSEXP, SEXP top_nSEXP, SEXP min_distanceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< std::string >::type x(xSEXP);
    Rcpp::traits::input_parameter< std::size_t >::type top_n(top_nSEXP);
    Rcpp::traits::input_parameter< float >::type min_distance(min_distanceSEXP);
    rcpp_result_gen = Rcpp::wrap(w2v_nearest(ptr, x, top_n, min_distance));
    return rcpp_result_gen;
END_RCPP
}
// w2v_nearest_vector
Rcpp::List w2v_nearest_vector(SEXP ptr, const std::vector<float>& x, std::size_t top_n, float min_distance);
RcppExport SEXP _wordvector_w2v_nearest_vector(SEXP ptrSEXP, SEXP xSEXP, SEXP top_nSEXP, SEXP min_distanceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< const std::vector<float>& >::type x(xSEXP);
    Rcpp::traits::input_parameter< std::size_t >::type top_n(top_nSEXP);
    Rcpp::traits::input_parameter< float >::type min_distance(min_distanceSEXP);
    rcpp_result_gen = Rcpp::wrap(w2v_nearest_vector(ptr, x, top_n, min_distance));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_wordvector_cpp_w2v", (DL_FUNC) &_wordvector_cpp_w2v, 16},
    {"_wordvector_w2v_dictionary", (DL_FUNC) &_wordvector_w2v_dictionary, 1},
    {"_wordvector_w2v_embedding", (DL_FUNC) &_wordvector_w2v_embedding, 2},
    {"_wordvector_w2v_nearest", (DL_FUNC) &_wordvector_w2v_nearest, 4},
    {"_wordvector_w2v_nearest_vector", (DL_FUNC) &_wordvector_w2v_nearest_vector, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_wordvector(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
