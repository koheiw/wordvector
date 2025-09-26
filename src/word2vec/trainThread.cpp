/**
 * @file
 * @brief trainThread trains a word2vec model from the specified part of train data set file
 * @author Max Fomichev
 * @date 20.12.2016
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/

#include "trainThread.hpp"

namespace w2v {
    
    trainThread_t::trainThread_t(const std::pair<std::size_t, std::size_t> &_range, 
                                 const data_t &_data) :
            m_range(_range),
            m_data(_data), m_randomGenerator(m_data.settings->random),
            //m_rndWindowShift(0, static_cast<short>((m_data.settings->window - 1))), // NOTE: to delete
            m_rndWindow(1, static_cast<short>((m_data.settings->window))), // NOTE: added
            m_downSampling(), m_nsDistribution(), m_hiddenLayerValues(), m_hiddenLayerErrors(),
            m_thread() {

        if (!m_data.settings) {
            throw std::runtime_error("train settings are not initialized");
        }

        if (m_data.settings->sample < 1.0f) {
            m_downSampling.reset(new downSampling_t(m_data.settings->sample,
                                                    m_data.corpus->trainWords));
        }

        if (m_data.settings->negative > 0) {
            m_nsDistribution.reset(new nsDistribution_t(m_data.corpus->frequency));
        }

        if (m_data.settings->withHS && !m_data.huffmanTree) {
            throw std::runtime_error("Huffman tree object is not initialized");
        }

        m_hiddenLayerErrors.reset(new std::vector<float>(m_data.settings->size));
        m_hiddenLayerValues.reset(new std::vector<float>(m_data.settings->size)); // not used in SG
        m_docLayerErrors.reset(new std::vector<float>(m_data.settings->size));
        m_docLayerValues.reset(new std::vector<float>(m_data.settings->size)); // not used in SG
        
        if (!m_data.corpus) {
            throw std::runtime_error("corpus object is not initialized");
        }
        
    }

    void trainThread_t::worker(int &_iter, float &_alpha) noexcept {
        
        for (auto g = 1; g <= m_data.settings->iterations; ++g) {
            
            std::size_t threadProcessedWords = 0;
            std::size_t prvThreadProcessedWords = 0;
            
            auto wordsPerAllThreads = m_data.settings->iterations * m_data.corpus->trainWords;
            auto wordsPerAlpha = wordsPerAllThreads / 10000;
            
            float alpha = 0;
            for (std::size_t h = m_range.first; h <= m_range.second; ++h) {
                
                // calculate alpha
                if (threadProcessedWords - prvThreadProcessedWords > wordsPerAlpha) { // next 0.01% processed
                    *m_data.processedWords += threadProcessedWords - prvThreadProcessedWords;
                    prvThreadProcessedWords = threadProcessedWords;

                    float ratio = static_cast<float>(*(m_data.processedWords)) / wordsPerAllThreads;
                    alpha = m_data.settings->alpha * (1 - ratio);
                    if (alpha < m_data.settings->alpha * 0.0001f) {
                        alpha = m_data.settings->alpha * 0.0001f;
                    }
                    (*m_data.alpha) = alpha;
                }
                
                text_t text = m_data.corpus->texts[h];
                //std::cout << "text: " <<  text.size() << "\n";
                
                // read sentence
                std::vector<unsigned int> sentence;
                sentence.reserve(text.size());
                for (size_t i = 0; i < text.size(); ++i) {

                    auto &word = text[i];
                    // ignore padding
                    if (word == 0) { 
                        //std::cout << "padding: " << word << "\n";
                        continue; 
                    }
                    // ignore infrequent words
                    // if (m_data.corpus->frequency[word - 1] < m_data.settings->minWordFreq) {
                    //     //std::cout << "infrequent: " << word << "\n";
                    //     continue;
                    // }
                    
                    threadProcessedWords++;
                    if (m_data.settings->sample < 1.0f) {
                        if ((*m_downSampling)(m_data.corpus->frequency[word - 1], m_randomGenerator)) {
                            //std::cout << "downsample: " << word << "\n";
                            continue; // skip this word
                        }
                    }
                    sentence.push_back(word - 1); // zero-based index of words
                }
                
                //std::cout << "sentence = " <<  sentence.size() << "\n";
                if (m_data.settings->type == 1) {
                    cbow(sentence);
                } else if (m_data.settings->type == 2) {
                    skipGram(sentence);
                } else if (m_data.settings->type == 3) {
                    cbow2(sentence, h);
                }
            }
            // for progress message
            if (m_range.first == 0) {
                _iter = g;
                _alpha = alpha;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // std::cout << "trainThread_t::worker()\n";
        // std::cout << m_data.bpWeights << "\n";
        // for (size_t i = 0; i < 10; i++){
        //     std::cout << (*m_data.bpWeights)[i] << ", ";
        // }
        // std::cout << "\n";
    }

    inline void trainThread_t::cbow(const std::vector<unsigned int> &_text) noexcept {
        
        std::size_t K = m_data.settings->size;
        if (_text.size() == 0)
            return;
        for (std::size_t i = 0; i < _text.size(); ++i) {
            // hidden layers initialized with 0 values for each target word
            std::memset(m_hiddenLayerValues->data(), 0, m_hiddenLayerValues->size() * sizeof(float));
            std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));

            int window = m_rndWindow(m_randomGenerator);
            std::size_t from = std::max(0, (int)i - window);
            std::size_t to = std::min((int)_text.size(), (int)i + window);
            std::size_t cw = 0;
            for (std::size_t j = from; j < to; ++j) {
                if (j == i)
                    continue;
                auto shift = _text[j] * K;
                for (std::size_t k = 0; k < K; ++k) {
                    (*m_hiddenLayerValues)[k] += (*m_data.pjLayerValues)[k + shift];
                }
                cw++;
            }
            
            if (cw == 0)
                continue;
            for (std::size_t k = 0; k < K; ++k) {
                (*m_hiddenLayerValues)[k] /= cw;
            }
            
            if (m_data.settings->withHS) {
                hierarchicalSoftmax(_text[i], *m_hiddenLayerErrors, *m_hiddenLayerValues, 0);
            } else {
                negativeSampling(_text[i], *m_hiddenLayerErrors, *m_hiddenLayerValues, 0);
            }
            
            // hidden -> in
            for (std::size_t j = from; j < to; ++j) {
                if (j == i)
                    continue;
                auto shift = _text[j] * K;
                for (std::size_t k = 0; k < K; ++k) {
                    (*m_data.pjLayerValues)[k + shift] += (*m_hiddenLayerErrors)[k];
                }
            }
        }
    }

    inline void trainThread_t::cbow2(const std::vector<unsigned int> &_text, std::size_t _docIndex) noexcept {
        
        std::size_t K = m_data.settings->size;
        if (_text.size() == 0)
            return;
        for (std::size_t i = 0; i < _text.size(); ++i) {
            // hidden layers initialized with 0 values for each target word
            std::memset(m_hiddenLayerValues->data(), 0, m_hiddenLayerValues->size() * sizeof(float));
            std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));
            std::memset(m_docLayerValues->data(), 0, m_docLayerValues->size() * sizeof(float));
            std::memset(m_docLayerErrors->data(), 0, m_docLayerErrors->size() * sizeof(float));
            
            int window = m_rndWindow(m_randomGenerator);
            std::size_t from = std::max(0, (int)i - window);
            std::size_t to = std::min((int)_text.size(), (int)i + window);
            std::size_t cw = 0;
            for (std::size_t j = from; j < to; ++j) {
                if (j == i)
                    continue;
                auto shift = _text[j] * K;
                for (std::size_t k = 0; k < K; ++k) {
                    (*m_hiddenLayerValues)[k] += (*m_data.pjLayerValues)[k + shift];
                }
                cw++;
            }
            
            if (cw == 0)
                continue;
            for (std::size_t k = 0; k < K; ++k) {
                (*m_hiddenLayerValues)[k] /= cw;
            }
            
            auto docShift = _docIndex * K;
            if (m_data.settings->withHS) {
                hierarchicalSoftmax2(_text[i], *m_hiddenLayerErrors, *m_hiddenLayerValues, 0,
                                               *m_docLayerErrors, *m_data.docValues, docShift);
            } else {
                negativeSampling2(_text[i], *m_hiddenLayerErrors, *m_hiddenLayerValues, 0,
                                            *m_docLayerErrors, *m_data.docValues, docShift);
            }
            
            // hidden -> in
            for (std::size_t j = from; j < to; ++j) {
                if (j == i)
                    continue;
                auto shift = _text[j] * K;
                for (std::size_t k = 0; k < K; ++k) {
                    (*m_data.pjLayerValues)[k + shift] += (*m_hiddenLayerErrors)[k];
                    (*m_data.docValues)[k + docShift] += (*m_docLayerErrors)[k];
                }
            }
        }
    }
    
    inline void trainThread_t::skipGram(const std::vector<unsigned int> &_text) noexcept {
        
        std::size_t K = m_data.settings->size;
        if (_text.size() == 0)
            return;
        for (std::size_t i = 0; i < _text.size(); ++i) {
            int window = m_rndWindow(m_randomGenerator);
            std::size_t from = std::max(0, (int)i - window);
            std::size_t to = std::min((int)_text.size(), (int)i + window);
            for (std::size_t j = from; j < to; j++) {
                if (j == i)
                    continue;
                
                // hidden layer initialized with 0 values for each context word
                std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));
                
                // shift to the selected word vector in the matrix
                auto shift = _text[j] * K;
                if (m_data.settings->withHS) {
                    hierarchicalSoftmax(_text[i], *m_hiddenLayerErrors, *m_data.pjLayerValues, shift);
                } else {
                    negativeSampling(_text[i], *m_hiddenLayerErrors, *m_data.pjLayerValues, shift);
                }
                for (std::size_t k = 0; k < m_data.settings->size; ++k) {
                    (*m_data.pjLayerValues)[k + shift] += (*m_hiddenLayerErrors)[k];
                }
            }
        }
    }

    inline void trainThread_t::hierarchicalSoftmax(std::size_t _word,
                                                   std::vector<float> &_hiddenLayerErrors,
                                                   std::vector<float> &_hiddenLayerValues,
                                                   std::size_t _hiddenLayerShift) noexcept {
        
        std::size_t K = m_data.settings->size;
        auto huffmanData = m_data.huffmanTree->huffmanData(_word);
        for (std::size_t i = 0; i < huffmanData->huffmanCode.size(); ++i) {
            auto shift = huffmanData->huffmanPoint[i] * K;
            
            // propagate hidden -> output
            float f = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                f += _hiddenLayerValues[k + _hiddenLayerShift] * (*m_data.bpWeights)[k + shift];
            }
            float prob = 0;
            if (f < -m_data.settings->expValueMax) {
                //continue;
                prob = 0.0f;
            } else if (f > m_data.settings->expValueMax) {
                //continue;
                prob = 1.0f;
            } else {
                auto r = (f + m_data.settings->expValueMax) * (m_data.expTable->size() / m_data.settings->expValueMax / 2);
                prob = (*m_data.expTable)[static_cast<std::size_t>(r)];
            }
            
            // compute gradient x alpha
            auto gxa = (1.0f - static_cast<float>(huffmanData->huffmanCode[i]) - prob) * (*m_data.alpha);
            // propagate errors output -> hidden
            for (std::size_t k = 0; k < K; ++k) {
                _hiddenLayerErrors[k] += gxa * (*m_data.bpWeights)[k + shift];
            }
            // learn weights hidden -> output
            for (std::size_t k = 0; k < K; ++k) {
                (*m_data.bpWeights)[k + shift] += gxa * _hiddenLayerValues[k + _hiddenLayerShift];
            }
        }
    }

    inline void trainThread_t::hierarchicalSoftmax2(std::size_t _word,
                                                   std::vector<float> &_wordLayerErrors,
                                                   std::vector<float> &_wordLayerValues,
                                                   std::size_t _wordLayerShift,
                                                   std::vector<float> &_docLayerErrors,
                                                   std::vector<float> &_docLayerValues, 
                                                   std::size_t _docLayerShift) noexcept {
        
        std::size_t K = m_data.settings->size;
        auto huffmanData = m_data.huffmanTree->huffmanData(_word);
        for (std::size_t i = 0; i < huffmanData->huffmanCode.size(); ++i) {
            auto shift = huffmanData->huffmanPoint[i] * K;
            
            // propagate hidden -> output
            float f = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                f += _wordLayerValues[k + _wordLayerShift] * (*m_data.bpWeights)[k + shift];
                f += _docLayerValues[k + _docLayerShift] * (*m_data.docWeights)[k + _docLayerShift];
            }
            float prob = 0;
            if (f < -m_data.settings->expValueMax) {
                //continue;
                prob = 0.0f;
            } else if (f > m_data.settings->expValueMax) {
                //continue;
                prob = 1.0f;
            } else {
                auto r = (f + m_data.settings->expValueMax) * (m_data.expTable->size() / m_data.settings->expValueMax / 2);
                prob = (*m_data.expTable)[static_cast<std::size_t>(r)];
            }
            
            // compute gradient x alpha
            auto gxa = (1.0f - static_cast<float>(huffmanData->huffmanCode[i]) - prob) * (*m_data.alpha);
            // propagate errors output -> hidden
            for (std::size_t k = 0; k < K; ++k) {
                _wordLayerErrors[k] += gxa * (*m_data.bpWeights)[k + shift];
                _docLayerErrors[k] += gxa * (*m_data.docWeights)[k + _docLayerShift];
            }
            // learn weights hidden -> output
            for (std::size_t k = 0; k < K; ++k) {
                (*m_data.bpWeights)[k + shift] += gxa * _wordLayerValues[k + _wordLayerShift];
                (*m_data.docWeights)[k + _docLayerShift] += gxa * _docLayerValues[k + _docLayerShift];
            }
        }
    }

    inline void trainThread_t::negativeSampling(std::size_t _word,
                                                std::vector<float> &_hiddenLayerErrors,
                                                std::vector<float> &_hiddenLayerValues,
                                                std::size_t _hiddenLayerShift) noexcept {
        
        std::size_t K = m_data.settings->size;
        for (std::size_t i = 0; i < static_cast<std::size_t>(m_data.settings->negative) + 1; ++i) {
            std::size_t target = 0;
            bool label = false;
            if (i == 0) {
                // positive case
                target = _word;
                label = true;
            } else {
                // negative case
                target = (*m_nsDistribution)(m_randomGenerator);
                if (target == _word) {
                    continue;
                }
            }
            auto shift = target * K;
            
            // propagate hidden -> output
            float f = 0.0f;
            // predict likelihood of _word using logistic regression
            for (std::size_t k = 0; k < K; ++k) {
                f += _hiddenLayerValues[k + _hiddenLayerShift] * (*m_data.bpWeights)[k + shift];
            }
            //std::cout << f << "\n";
            float prob = 0;
            if (f < -m_data.settings->expValueMax) {
                prob = 0.0f;
            } else if (f > m_data.settings->expValueMax) {
                prob = 1.0f;
            } else {
                auto r = (f + m_data.settings->expValueMax) * (m_data.expTable->size() / m_data.settings->expValueMax / 2);
                prob = (*m_data.expTable)[static_cast<std::size_t>(r)];
            }
            
            // compute gradient x alpha
            auto gxa = (static_cast<float>(label) - prob) * (*m_data.alpha); // gxa >= 0 in the positive case
            //std::cout << i << ": " << _word << ", " <<  target << ", " << gxa << "\n";
            // propagate errors output -> hidden
            for (std::size_t k = 0; k < K; ++k) {
                _hiddenLayerErrors[k] += gxa * (*m_data.bpWeights)[k + shift]; // added to pjLayerValues
            }
            // learn weights hidden -> output
            for (std::size_t k = 0; k < K; ++k) {
                (*m_data.bpWeights)[k + shift] += gxa * _hiddenLayerValues[k + _hiddenLayerShift];
            }
        }
    }

    

    inline void trainThread_t::negativeSampling2(std::size_t _word,
                                                std::vector<float> &_wordLayerErrors,
                                                std::vector<float> &_wordLayerValues,
                                                std::size_t _wordLayerShift,
                                                std::vector<float> &_docLayerErrors,
                                                std::vector<float> &_docLayerValues,
                                                std::size_t _docLayerShift
                                                ) noexcept {
        
        std::size_t K = m_data.settings->size;
        for (std::size_t i = 0; i < static_cast<std::size_t>(m_data.settings->negative) + 1; ++i) {
            std::size_t target = 0;
            bool label = false;
            if (i == 0) {
                // positive case
                target = _word;
                label = true;
            } else {
                // negative case
                target = (*m_nsDistribution)(m_randomGenerator);
                if (target == _word) {
                    continue;
                }
            }
            auto shift = target * K;
            
            // propagate hidden -> output
            float f = 0.0f;
            // predict likelihood of _word using logistic regression
            for (std::size_t k = 0; k < K; ++k) {
                f += _wordLayerValues[k + _wordLayerShift] * (*m_data.bpWeights)[k + shift];
                f += _docLayerValues[k + _docLayerShift] * (*m_data.docWeights)[k + _docLayerShift];
            }
            //std::cout << f << "\n";
            float prob = 0;
            if (f < -m_data.settings->expValueMax) {
                prob = 0.0f;
            } else if (f > m_data.settings->expValueMax) {
                prob = 1.0f;
            } else {
                auto r = (f + m_data.settings->expValueMax) * (m_data.expTable->size() / m_data.settings->expValueMax / 2);
                prob = (*m_data.expTable)[static_cast<std::size_t>(r)];
            }
            
            // compute gradient x alpha
            auto gxa = (static_cast<float>(label) - prob) * (*m_data.alpha); // gxa >= 0 in the positive case
            //std::cout << i << ": " << _word << ", " <<  target << ", " << gxa << "\n";
            // propagate errors output -> hidden
            for (std::size_t k = 0; k < K; ++k) {
                _wordLayerErrors[k] += gxa * (*m_data.bpWeights)[k + shift]; // added to pjLayerValues
                _docLayerErrors[k] += gxa * (*m_data.docWeights)[k + _docLayerShift];
            }
            // learn weights hidden -> output
            for (std::size_t k = 0; k < K; ++k) {
                (*m_data.bpWeights)[k + shift] += gxa * _wordLayerValues[k + _wordLayerShift];
                (*m_data.docWeights)[k + _docLayerShift] += gxa * _docLayerValues[k + _docLayerShift];
            }
        }
    }
}
