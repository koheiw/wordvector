/**
 * @file
 * @brief
 * @author Max Fomichev
 * @date 15.02.2017
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/
#include <Rcpp.h>
#ifndef WORD2VEC_WORD2VEC_HPP
#define WORD2VEC_WORD2VEC_HPP

#include <cassert>
#include <string>
#include <vector>
#include <queue>
#include <functional>
#include <cmath>
#include <stdexcept>

typedef std::vector<std::string> types_t;
typedef std::vector<unsigned int> text_t;
typedef std::vector<text_t> texts_t;
typedef std::vector<size_t> frequency_t;

namespace w2v {
    
    /**
     * @brief corpus stores tokens
     */    
    class corpus_t final {
    public:
        texts_t texts;
        types_t types;
        frequency_t frequency;
        size_t totalWords;
        size_t trainWords;
        
        // constructors
        corpus_t(): texts() {}
        corpus_t(texts_t _texts, types_t _types): 
                 texts(_texts), types(_types) {}

        void setWordFreq() {
            
            frequency = frequency_t(types.size(), 0);
            totalWords = 0;
            trainWords = 0;
            for (size_t h = 0; h < texts.size(); h++) {
                text_t text = texts[h];
                for (size_t i = 0; i < text.size(); i++) {
                    totalWords++;
                    auto &word = text[i];
                    if (word < 0 || types.size() < word)
                        throw std::range_error("invalid token object");
                    if (word == 0) // padding
                        continue;
                    frequency[word - 1]++;
                    trainWords++;
                }
            }
            // Rcpp::Rcout << "trainWords: " << trainWords << "\n";
            // Rcpp::Rcout << "totalWords: " << totalWords << "\n";
            // Rcpp::Rcout << "frequency.size(): " << frequency.size() << "\n";
            // Rcpp::Rcout << "words.size(): " << words.size() << "\n";
        }
    };
    
    /**
     * @brief settings structure holds all training parameters
     */
    struct settings_t final {
        uint16_t size = 100; //< word vector size
        uint16_t window = 5; //< skip length between words
        uint16_t expTableSize = 1000; //< exp(x) / (exp(x) + 1) values lookup table size
        uint16_t expValueMax = 6; //< max value in the lookup table
        float sample = 1e-3f; //< threshold for occurrence of words
        bool withHS = false; //< use hierarchical softmax instead of negative sampling
        uint16_t negative = 5; //< negative examples number
        uint16_t threads = 1; //< train threads number
        uint16_t iterations = 5; //< train iterations
        float alpha = 0.05f; //< starting learn rate
        bool freeze = false;
        int type = 1; //< 1:CBOW 2:Skip-Gram 3:CBOW (doc2vec) 4:Skip-Gram (doc2vec)
        uint32_t random = 1234; // < random number seed
        bool verbose = false; // print progress
        settings_t() = default;
    };


    class word2vec_t final {
    protected:
        
        // vocabulary
        std::vector<std::string> m_vocabulary;
        std::size_t m_vocabularySize = 0;
        
        // word vector
        std::size_t m_vectorSize = 0;
        std::vector<float> m_pjLayerValues;
        std::vector<float> m_bpWeights;
        
        // document vector
        std::size_t m_corpusSize = 0;
        std::vector<float> m_docValues;
        
        mutable std::string m_errMsg;
        
    public:
        
        // constructor
        word2vec_t() {};
        word2vec_t(std::vector<std::string> vocabulary_,
                   std::size_t vectorSize_,
                   std::vector<float> pjLayerValues_,
                   std::vector<float> bpWeights_): 
                   m_vocabulary(vocabulary_),
                   m_vocabularySize(vocabulary_.size()),
                   m_vectorSize(vectorSize_),
                   m_pjLayerValues(pjLayerValues_),
                   m_bpWeights(bpWeights_) {}
    
        // virtual destructor
        virtual ~word2vec_t() = default;
        
        const std::vector<float> &values() {return m_pjLayerValues;}  // TODO: change to wordValues
        const std::vector<float> &weights() {return m_bpWeights;}
        const std::vector<float> &docValues() {return m_docValues;} 
        
        // @returns m_corpusSize size (number of documents)
        std::size_t corpusSize() const noexcept {return m_corpusSize;}
        // @returns vector size of model
        std::size_t vectorSize() const noexcept {return m_vectorSize;}
        // @returns m_vocabularySize size (number of unique words)
        std::size_t vocabularySize() const noexcept {return m_vocabularySize;}
        // @returns vector size of model
        std::vector<std::string> vocabulary() const noexcept {return m_vocabulary;}
        // @returns error message
        std::string errMsg() const noexcept {return m_errMsg;}
        
        // train model
        bool train(const settings_t &_settings,
                   const corpus_t &_corpus,
                   const word2vec_t &_model) noexcept;
        
        // normalize by factors
        void normalizeValues() {
            for(std::size_t i = 0; i < m_vocabularySize; i += m_vectorSize) {
                float ss = 0.0f;
                for(std::size_t j = 0; j < m_vectorSize; ++j) {
                    ss += m_pjLayerValues[i + j] * m_pjLayerValues[i + j];
                }
                if (ss <= 0.0f) 
                    throw std::runtime_error("failed to normalize pjLayerValues");
                float d = std::sqrt(ss / m_vectorSize);
                for(std::size_t j = 0; j < m_vectorSize; ++j) {
                    m_pjLayerValues[i + j] = m_pjLayerValues[i + j] / d;
                }
            }
        }
        
    };
}
#endif // WORD2VEC_WORD2VEC_HPP
