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
#include <unordered_map>
#include <queue>
#include <functional>
#include <cmath>
#include <stdexcept>

typedef std::vector<std::string> words_t;
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
        words_t words;
        frequency_t frequency;
        size_t totalWords;
        size_t trainWords;
        
        // constructors
        corpus_t(): texts() {}
        corpus_t(texts_t _texts, words_t _words): 
                 texts(_texts), words(_words) {}

        void setWordFreq() {
            
            frequency = frequency_t(words.size(), 0);
            totalWords = 0;
            trainWords = 0;
            for (size_t h = 0; h < texts.size(); h++) {
                text_t text = texts[h];
                for (size_t i = 0; i < text.size(); i++) {
                    totalWords++;
                    auto &word = text[i];
                    //Rcpp::Rcout << i << ": " << word << "\n"; 
                    if (word < 0 || words.size() < word)
                        throw std::range_error("setWordFreq: invalid words");
                    if (word == 0) // padding
                        continue;
                    // if (words[word - 1].empty()) {
                    //     word = 0; // remove and pad
                    //     continue;
                    // }
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
        uint16_t minWordFreq = 5; ///< discard words that appear less than minWordFreq times
        uint16_t size = 100; ///< word vector size
        uint16_t window = 5; ///< skip length between words
        uint16_t expTableSize = 1000; ///< exp(x) / (exp(x) + 1) values lookup table size
        uint16_t expValueMax = 6; ///< max value in the lookup table
        float sample = 1e-3f; ///< threshold for occurrence of words
        bool withHS = false; ///< use hierarchical softmax instead of negative sampling
        uint16_t negative = 5; ///< negative examples number
        uint16_t threads = 1; ///< train threads number
        uint16_t iterations = 5; ///< train iterations
        float alpha = 0.05f; ///< starting learn rate
        int type = 1; ///< 1:CBOW 2:Skip-Gram
        uint32_t random = 1234; /// < random number seed
        settings_t() = default;
    };


    class word2vec_t final {
    protected:
        //using map_t = std::unordered_map<std::string, std::vector<float>>;
        
        // word vector
        std::vector<float> m_trainMatrix; // NOTE: rename to m_layer or m_wordvector?
        //map_t m_map;
        uint16_t m_vectorSize = 0;
        std::size_t m_vocaburarySize = 0;
        //std::size_t m_mapSize = 0;
        mutable std::string m_errMsg;
        
    public:
        /// type of callback function to be called on train data file parsing progress events
        using vocabularyProgressCallback_t = std::function<void(float)>;
        /// type of callback function to be called on train data file parsed event
        using vocabularyStatsCallback_t = std::function<void(std::size_t, std::size_t, std::size_t)>;
        /// type of callback function to be called on training progress events
        using trainProgressCallback_t = std::function<void(float, float)>;

        /// constructs a model
        //word2vec_t(): m_map(), m_errMsg() {}
        /// virtual destructor
        virtual ~word2vec_t() = default;
        
        /// direct access to the word-vector map
        //const map_t &map() {return m_map;} // NOTE: consider removing
        
        const std::vector<float> &trainMatrix() {return m_trainMatrix;} 
        
        /// @returns vector size of model
        uint16_t vectorSize() const noexcept {return m_vectorSize;}
        /// @returns model size (number of stored vectors)
        //std::size_t modelSize() const noexcept {return m_mapSize;}
        /// @returns m_vocaburarySize size (number of unique words)
        std::size_t vocaburarySize() const noexcept {return m_vocaburarySize;}
        /// @returns error message
        std::string errMsg() const noexcept {return m_errMsg;}
        
        // train model
        bool train(const settings_t &_settings,
                   const corpus_t &_corpus,
                   trainProgressCallback_t _trainProgressCallback) noexcept;
        
        /// normalise vectors
        // inline void normalize() {
        //     for(auto &it : m_map) {
        //         // normalize vector
        //         auto &vec = it.second;
        //         float ss = 0.0f;
        //         for (auto const &v : vec) {
        //             ss += v * v;
        //         }
        //         if (ss <= 0.0f) 
        //             throw std::runtime_error("failed to normalize vectors");
        //         float d = std::sqrt(ss / vec.size());
        //         for (auto &v : vec) {
        //             v = v / d;
        //         }
        //     } 
        // }
        
        // normalise vectors
        inline void normalize() {
            for(std::size_t i = 0; i < m_vocaburarySize; i += m_vectorSize) {
                float ss = 0.0f;
                for(std::size_t j = 0; j < m_vectorSize; ++j) {
                    ss += m_trainMatrix[i + j] * m_trainMatrix[i + j];
                }
                if (ss <= 0.0f) 
                    throw std::runtime_error("failed to normalize vectors");
                float d = std::sqrt(ss / m_vectorSize);
                for(std::size_t j = 0; j < m_vectorSize; ++j) {
                    m_trainMatrix[i + j] = m_trainMatrix[i + j] / d;
                }
            } 
        }

    };
    
    class progress_t {
    public:
        void iteration(int iter, double msec, float alpha) {
            Rprintf(" ......iteration %d elapsed time: %.2f seconds (alpha: %.4f)\n", 
                    iter, msec / 1000, alpha);
        }
    };
}
#endif // WORD2VEC_WORD2VEC_HPP
