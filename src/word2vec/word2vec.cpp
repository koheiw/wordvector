/**
 * @file
 * @brief
 * @author Max Fomichev
 * @date 15.02.2017
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/
#include <Rcpp.h>
#include "word2vec.hpp"
#include "trainer.hpp"

namespace w2v {
    bool w2vModel_t::train(const trainSettings_t &_trainSettings,
                           const corpus_t &_corpus,
                           trainProgressCallback_t _trainProgressCallback) noexcept {
        try {
            // store tokens
            std::shared_ptr<corpus_t> corpus(new corpus_t(_corpus));
            
            m_vectorSize = _trainSettings.size;
            m_mapSize = corpus->types.size();
            
            // train model
            std::vector<float> _trainMatrix;
            trainer_t(std::make_shared<trainSettings_t>(_trainSettings),
                      corpus,
                      _trainProgressCallback)(_trainMatrix);

            std::size_t wordIndex = 0;
            for (auto const &i : corpus->types) {
                //Rcpp::Rcout << i << "\n";
                auto &v = m_map[i];
                v.resize(m_vectorSize);
                std::copy(&_trainMatrix[wordIndex * m_vectorSize],
                          &_trainMatrix[(wordIndex + 1) * m_vectorSize],
                          &v[0]);
                wordIndex++;
            }

            return true;
        } catch (const std::exception &_e) {
            m_errMsg = _e.what();
        } catch (...) {
            m_errMsg = "unknown error";
        }

        return false;
    }
}
