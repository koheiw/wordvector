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
#include <memory>
#include <functional>
#include <cmath>
#include <stdexcept>

typedef std::vector<std::string> types_t;
typedef std::vector<unsigned int> words_t;
typedef std::vector<unsigned int> text_t;
// typedef std::vector<int> words_t;
// typedef std::vector<int> text_t;
typedef std::vector<text_t> texts_t;
typedef std::vector<size_t> frequency_t;

namespace w2v {
    
    /**
     * @brief corpus stores tokens and stopwords     
     */    
    class corpus_t final {
    public:
        texts_t texts;
        types_t types;
        frequency_t frequency;
        size_t totalWords;
        size_t trainWords;
        
        // Constructors
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
                    //Rcpp::Rcout << i << ": " << word << "\n"; 
                    if (word < 0 || types.size() < word)
                        throw std::range_error("setWordFreq: invalid types");
                    if (word == 0) // padding
                        continue;
                    // if (types[word - 1].empty()) {
                    //     word = 0; // remove and pad
                    //     continue;
                    // }
                    frequency[word - 1]++;
                    trainWords++;
                }
            }
            Rcpp::Rcout << "trainWords: " << trainWords << "\n";
            Rcpp::Rcout << "totalWords: " << totalWords << "\n";
            Rcpp::Rcout << "frequency.size(): " << frequency.size() << "\n";
            Rcpp::Rcout << "types.size(): " << types.size() << "\n";
        }
    };
    
    /**
     * @brief trainSettings structure holds all training parameters
     */
    struct trainSettings_t final {
        uint16_t minWordFreq = 5; ///< discard words that appear less than minWordFreq times
        uint16_t size = 100; ///< word vector size
        uint8_t window = 5; ///< skip length between words
        uint16_t expTableSize = 1000; ///< exp(x) / (exp(x) + 1) values lookup table size
        uint8_t expValueMax = 6; ///< max value in the lookup table
        float sample = 1e-3f; ///< threshold for occurrence of words
        bool withHS = false; ///< use hierarchical softmax instead of negative sampling
        uint8_t negative = 5; ///< negative examples number
        uint8_t threads = 12; ///< train threads number
        uint8_t iterations = 5; ///< train iterations
        float alpha = 0.05f; ///< starting learn rate
        bool withSG = false; ///< use Skip-Gram instead of CBOW
        uint32_t random = 1234; /// < random number seed
        // TODO: remove
        std::string wordDelimiterChars = " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r";
        std::string endOfSentenceChars = ".\n?!";
        trainSettings_t() = default;
    };

    /**
     * @brief base class of a vector itself and vector operations
     *
     * w2vBase class implements basic vector operations such us "+", "-", "=" and "=="
    */
    class vector_t: public std::vector<float> {
    public:
        vector_t(): std::vector<float>() {}
        /// Constructs an empty vector with size _size
        explicit vector_t(std::size_t _size): std::vector<float>(_size, 0.0f) {}

        /// Constructs a vector from std::vector
        explicit vector_t(const std::vector<float> &_vector): std::vector<float>(_vector) {}

        /// @returns summarized w2vBase object
        vector_t &operator+=(const vector_t &_from) {
            if (this != &_from) {
                assert(size() == _from.size());

                for (std::size_t i = 0; i <  size(); ++i) {
                    (*this)[i] += _from[i];
                }
                float med = 0.0f;
                for (auto const &i:(*this)) {
                    med += i * i;
                }
                if (med <= 0.0f) {
                    throw std::runtime_error("word2vec: can not create vector");
                }
                med = std::sqrt(med / this->size());
                for (auto &i:(*this)) {
                    i /= med;
                }
            }

            return *this;
        }

        /// @returns substracted w2vBase object
        vector_t &operator-=(const vector_t &_from) {
            if (this != &_from) {
                assert(size() == _from.size());

                for (std::size_t i = 0; i < size(); ++i) {
                    (*this)[i] -= _from[i];
                }
                float med = 0.0f;
                for (auto const &i:(*this)) {
                    med += i * i;
                }
                if (med <= 0.0f) {
                    throw std::runtime_error("word2vec: can not create vector");
                }
                med = std::sqrt(med / this->size());
                for (auto &i:(*this)) {
                    i /= med;
                }
            }

            return *this;
        }

        friend vector_t operator+(vector_t _what, const vector_t &_with) {
            _what += _with;
            return _what;
        }

        friend vector_t operator-(vector_t _what, const vector_t &_with) {
            _what -= _with;
            return _what;
        }
    };

    /**
     * @brief base class of a vectors model
     *
     * Model is a storage of pairs key&vector, it implements some basic functionality related to vectors storage -
     * model size, vector size, get vector by key, calculate distance between two vectors and find nearest vectors
     * to a specified vector
    */
    template <class key_t>
    class model_t {
    protected:
        using map_t = std::unordered_map<key_t, vector_t>;

        map_t m_map;
        uint16_t m_vectorSize = 0;
        std::size_t m_mapSize = 0;
        mutable std::string m_errMsg;

        const std::string wrongFormatErrMsg = "model: wrong model file format";

    private:
        struct nearestCmp_t final {
            inline bool operator()(const std::pair<key_t, float> &_left,
                                   const std::pair<key_t, float> &_right) const noexcept {
                return _left.second > _right.second;
            }
        };

    public:
        /// constructs a model
        model_t(): m_map(), m_errMsg() {}
        /// virtual destructor
        virtual ~model_t() = default;

        /// Direct access to the word-vector map
        const map_t &map() {return m_map;}

        /**
         * Vector access by key value
         * @param _key key value uniquely identifying vector in model
         * @returns pointer to the vector or nullptr if no such key found
        */
        inline const vector_t *vector(const key_t &_key) const noexcept {
            auto const &i = m_map.find(_key);
            if (i != m_map.end()) {
                return &i->second;
            }

            return nullptr;
        }

        /// @returns vector size of model
        inline uint16_t vectorSize() const noexcept {return m_vectorSize;}
        /// @returns model size (number of stored vectors)
        inline std::size_t modelSize() const noexcept {return m_mapSize;}
        /// @returns error message
        inline std::string errMsg() const noexcept {return m_errMsg;}
    };

    /**
     * @brief storage model of pairs key&vector where key type is std::string (word)
     *
     * Model is derived from model_t class and implements save/load methods and train model method
    */
    class w2vModel_t: public model_t<std::string> {
    public:
        /// type of callback function to be called on train data file parsing progress events
        using vocabularyProgressCallback_t = std::function<void(float)>;
        /// type of callback function to be called on train data file parsed event
        using vocabularyStatsCallback_t = std::function<void(std::size_t, std::size_t, std::size_t)>;
        /// type of callback function to be called on training progress events
        using trainProgressCallback_t = std::function<void(float, float)>;

    public:
        /// Constructs w2vModel object
        w2vModel_t(): model_t<std::string>() {}

        /**
         * Trains model
         * @param _trainSettings trainSettings_t structure with training parameters
         * @param _trainFile file name of train corpus data
         * @param _stopWordsFile file name with stop words
         * @param _vocabularyProgressCallback callback function reporting train corpus data parsing progress,
         * nullptr if progress statistic is not needed
         * @param _vocabularyStatsCallback callback function reporting train corpus statistic,
         * nullptr if train data corpus statistic is not needed
         * @param _trainProgressCallback callback function reporting training progress,
         * nullptr if training progress statistic is not needed
         * @returns true on successful completion or false otherwise
        */
        bool train(const trainSettings_t &_trainSettings,
                   const corpus_t &_corpus,
                   trainProgressCallback_t _trainProgressCallback) noexcept;

        /**
         * Normalise vectors
         */
        inline void normalize() {
            for(auto &kv : m_map) {
                // normalize vector
                auto &v = kv.second;
                float med = 0.0f;
                for (auto const &j:v) {
                    med += j * j;
                }
                if (med <= 0.0f) {
                    throw std::runtime_error("failed to normalize vectors");
                }
                med = std::sqrt(med / v.size());
                for (auto &j:v) {
                    j /= med;
                }
            } 
        }
    };
}
#endif // WORD2VEC_WORD2VEC_HPP
