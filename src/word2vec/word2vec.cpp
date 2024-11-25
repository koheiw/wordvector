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
    bool word2vec_t::train(const settings_t &_settings,
                           const corpus_t &_corpus,
                           trainProgressCallback_t _trainProgressCallback) noexcept {
        try {
            // store tokens
            std::shared_ptr<corpus_t> corpus(new corpus_t(_corpus));
            
            m_vectorSize = _settings.size;
            m_mapSize = corpus->types.size();
            
            // train model
            //std::vector<float> _trainMatrix;
            trainer_t(std::make_shared<settings_t>(_settings),
                      corpus,
                      _trainProgressCallback)(trainMatrix);

            std::size_t wordIndex = 0;
            for (auto const &type : corpus->types) {
                //Rcpp::Rcout << type << "\n";
                auto &vec = m_map[type];
                vec.resize(m_vectorSize);
                std::copy(&trainMatrix[wordIndex * m_vectorSize],
                          &trainMatrix[(wordIndex + 1) * m_vectorSize],
                          &vec[0]);
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
