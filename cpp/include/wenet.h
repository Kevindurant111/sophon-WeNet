#include <sndfile.h>
#include <armadillo>
#include <string>
#include <cassert>
#include "processor.h"
#include "wrapper.h"
#include "bmruntime_cpp.h"
#include "utils.hpp"
#include "util.h"

class WeNet {
    int sample_frequency;
    int num_mel_bins;
    int frame_shift;
    int frame_length;
    int decoding_chunk_size;
    int subsampling_rate;
    int context;
    std::vector<std::string> dict;

    int batch_size;
    int beam_size;

    const char* model;
    std::shared_ptr<bmruntime::Context> ctx;
    std::shared_ptr<bmruntime::Network> net;
    std::vector<bmruntime::Tensor *> inputs;
    std::vector<bmruntime::Tensor *> outputs;
    std::string result;

    arma::fmat feats;
    TimeStamp *m_ts;

    int pre_process(const char* file_path);
    int inference();

    public:
    
    WeNet(std::shared_ptr<bmruntime::Context> ctx): ctx(ctx) {};
    int Init(const std::vector<std::string>& dict, int sample_frequency, int num_mel_bins, int frame_shift, int frame_length, int decoding_chunk_size, int subsampling_rate, int context);
    std::string Recognize(const char* file_path);
    void enableProfile(TimeStamp *ts);
};