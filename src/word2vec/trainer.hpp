/**
 * @file
 * @brief trainer class of word2vec model
 * @author Max Fomichev
 * @date 20.12.2016
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/

#ifndef WORD2VEC_TRAINER_H
#define WORD2VEC_TRAINER_H

#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>

#include "word2vec.hpp"
#include "trainThread.hpp"

namespace w2v {
    /**
     * @brief trainer class of word2vec model
     *
     * trainer class is responsible for train-specific data instantiation, train threads control and
     * train process itself.
    */
    class trainer_t {
    private:
        std::size_t m_matrixSize = 0;
        std::vector<std::unique_ptr<trainThread_t>> m_threads;
        int m_iter = 0;
        uint32_t m_random = 1234; // random seed
        bool m_verbose = false;
        
    public:
        /**
         * Constructs a trainer object
         * @param _settings trainSattings object
         * @param _vocabulary vocabulary object
         * @param _fileMapper fileMapper object related to a train data set file
        */
        trainer_t(const std::shared_ptr<settings_t> &_settings,
                  const std::shared_ptr<corpus_t> &_corpus); 

        /**
         * Runs training process
         * @param[out] _trainMatrix train model matrix
        */
        void operator()(std::vector<float> &_trainMatrix) noexcept;
    };
}

#endif // WORD2VEC_TRAINER_H
