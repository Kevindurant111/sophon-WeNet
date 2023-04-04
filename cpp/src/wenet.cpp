#include "wenet.h"

using namespace bmruntime;

int WeNet::Init(const std::vector<std::string>& dict, int sample_frequency, int num_mel_bins, int frame_shift, int frame_length, int decoding_chunk_size, int subsampling_rate, int context) {
    this->dict = dict;
    this->sample_frequency = sample_frequency;
    this->sample_frequency = sample_frequency;
    this->num_mel_bins = num_mel_bins;
    this->frame_shift = frame_shift;
    this->frame_length = frame_length;
    this->decoding_chunk_size = decoding_chunk_size;
    this->subsampling_rate = subsampling_rate;
    this->context = context;

    // create Network
    std::vector<const char *> network_names;
    ctx->get_network_names(&network_names);
    net = std::make_shared<Network>(*ctx, network_names[0], 0); // use stage[0]
    assert(net->info()->input_num == 6);

    // Initialize the memory space required for the input and output tensors
    inputs = net->Inputs();
    outputs = net->Outputs();
    const bm_tensor_t * log_probs_bm_tensor_t = outputs[0]->tensor();
    batch_size = log_probs_bm_tensor_t->shape.dims[0];
    beam_size = log_probs_bm_tensor_t->shape.dims[2];
    assert(batch_size == 1 && "Streaming inference only supports batch size = 1!");

    return 0;
}

int WeNet::pre_process(const char* file_path) {
    LOG_TS(m_ts, "wenet load file");
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(file_path, SFM_READ, &sfinfo);

    if (sndfile == NULL) {
        std::cerr << "Error opening WAV file: " << sf_strerror(sndfile) << std::endl;
        return 1;
    }

    // std::cout << "Sample rate: " << sfinfo.samplerate << std::endl;
    // std::cout << "Channels: " << sfinfo.channels << std::endl;
    // std::cout << "Frames: " << sfinfo.frames << std::endl;

    // Read the samples into a buffer
    const int buffer_size = sfinfo.frames * sfinfo.channels;
    double buffer[buffer_size];
    sf_read_double(sndfile, buffer, buffer_size);
    // sf_count_t samples_read = sf_read_double(sndfile, buffer, buffer_size);
    // std::cout << "Samples read: " << samples_read << std::endl;

    // Close the file
    sf_close(sndfile);

    arma::fmat waveform = arma::fmat(sfinfo.channels, sfinfo.frames);
    if(sfinfo.channels == 1) {
        for (int i = 0; i < sfinfo.frames; i++) {
            waveform(0, i) = buffer[i] * (1 << 15);
        }
    }
    else if(sfinfo.channels == 2) {
        for (int i = 0; i < sfinfo.frames; i += 2) {
            waveform(0, i / 2) = buffer[i] * (1 << 15);
            waveform(0, i / 2) = buffer[i + 1] * (1 << 15);
        }       
    }
    else {
        std::cerr << "The number of channels in the wav file is not normal!" << std::endl;
        return 1;
    }
    LOG_TS(m_ts, "wenet load file");

    LOG_TS(m_ts, "wenet fbank");
    feats = fbank(waveform, num_mel_bins, frame_length, frame_shift, sample_frequency, 0.0, 0.0, true, true, false);
    // todo:spec_sub
    LOG_TS(m_ts, "wenet fbank");

    return 0;
}

int WeNet::inference() {
    void* att_cache = calloc(inputs[1]->num_elements(), sizeof(float));
    void* cnn_cache = calloc(inputs[2]->num_elements(), sizeof(float));
    void* cache_mask = calloc(inputs[4]->num_elements(), sizeof(float));
    void* offset = calloc(inputs[5]->num_elements(), sizeof(int));

    void* log_probs = calloc(outputs[0]->num_elements(), sizeof(float));
    void* log_probs_idx = calloc(outputs[1]->num_elements(), sizeof(int));
    void* chunk_out = calloc(outputs[2]->num_elements(), sizeof(float));
    void* chunk_out_lens = calloc(outputs[3]->num_elements(), sizeof(int));

    // inference
    int num_frames = feats.n_rows;
    int stride = subsampling_rate * decoding_chunk_size;
    int decoding_window = (decoding_chunk_size - 1) * subsampling_rate + context;
    
    result = "";
    for(int cur = 0; cur < num_frames - context + 1; cur += stride) {
        int end = std::min(cur + decoding_window, num_frames);
        arma::fmat chunk_xs = feats.submat(cur, 0, end - 1, feats.n_cols - 1);
        // pad if needed
        if((int)chunk_xs.n_rows < decoding_window) {
            arma::fmat pad_zeros(decoding_window - chunk_xs.n_rows, chunk_xs.n_cols, arma::fill::zeros);
            chunk_xs = arma::join_cols(chunk_xs, pad_zeros);
        }
        
        void* chunk_xs_ptr = fmat_to_sys_mem(chunk_xs);
        int chunk_lens = chunk_xs.n_rows;
        int* chunk_lens_ptr = (int*) malloc(sizeof(int));
        if (chunk_lens_ptr != nullptr) {
            *chunk_lens_ptr = chunk_lens;
        }
        else {
            std::cerr << "Failed to request memory space" << std::endl;
            exit(1);
        }
        void* void_chunk_lens_ptr = chunk_lens_ptr;

        inputs[0]->CopyFrom(void_chunk_lens_ptr);
        inputs[1]->CopyFrom(att_cache);
        inputs[2]->CopyFrom(cnn_cache);
        inputs[3]->CopyFrom(chunk_xs_ptr);
        inputs[4]->CopyFrom(cache_mask);
        inputs[5]->CopyFrom(offset);
        LOG_TS(m_ts, "wenet inference");
        auto status = net->Forward();
        LOG_TS(m_ts, "wenet inference");
        assert(BM_SUCCESS == status);
    
        outputs[0]->CopyTo(log_probs);
        outputs[1]->CopyTo(log_probs_idx);
        outputs[2]->CopyTo(chunk_out);
        outputs[3]->CopyTo(chunk_out_lens);
        outputs[4]->CopyTo(offset);
        outputs[5]->CopyTo(att_cache);
        outputs[6]->CopyTo(cnn_cache);
        outputs[7]->CopyTo(cache_mask);

        LOG_TS(m_ts, "wenet postprocess");
        std::vector<std::string> hyps = ctc_decoding(log_probs, log_probs_idx, chunk_out_lens, beam_size, batch_size, dict, "ctc_prefix_beam_search");
        LOG_TS(m_ts, "wenet postprocess");
        std::cout << hyps[0] << std::endl;
        result += hyps[0];

        std::free(chunk_lens_ptr);
        std::free(chunk_xs_ptr);
    }
    std::free(att_cache);
    std::free(cnn_cache);
    std::free(cache_mask);
    std::free(offset);

    std::free(log_probs);
    std::free(log_probs_idx);
    std::free(chunk_out);
    std::free(chunk_out_lens);
    
    return 0;
}

std::string WeNet::Recognize(const char* file_path) {
    LOG_TS(m_ts, "wenet preprocess");
    pre_process(file_path);
    LOG_TS(m_ts, "wenet preprocess");

    inference();
    return result;
}

void WeNet::enableProfile(TimeStamp *ts) {
    m_ts = ts;
}