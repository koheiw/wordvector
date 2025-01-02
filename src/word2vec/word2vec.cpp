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
                           const corpus_t &_corpus) noexcept {
        try {
            // store tokens
            std::shared_ptr<corpus_t> corpus(new corpus_t(_corpus));
            
            m_vectorSize = _settings.size;
            m_vocaburarySize = corpus->words.size();
            
            // train model
            //std::vector<float> _trainMatrix;
            trainer_t(std::make_shared<settings_t>(_settings),
                      corpus)(m_trainMatrix);

            return true;
        } catch (const std::exception &_e) {
            m_errMsg = _e.what();
        } catch (...) {
            m_errMsg = "unknown error";
        }

        return false;
    }
}
