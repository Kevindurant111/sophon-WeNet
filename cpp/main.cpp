#include <sndfile.h>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "processor.h"
#include "bmruntime_cpp.h"
#include "wrapper.h"

using namespace bmruntime;

int main(int argc, char** argv) {
    std::string model = "/data/WeNet/models/BM1684/wenet_encoder_fp32.bmodel";
    int sample_frequency = 16000;
    int num_mel_bins = 80;
    int frame_shift = 10;
    int frame_length = 25;

    int decoding_chunk_size = 16;
    int subsampling_rate = 4;
    int context = 7;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <wav_file_path>" << std::endl;
        return 1;
    }

    const char* file_path = argv[1];
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(file_path, SFM_READ, &sfinfo);

    if (sndfile == NULL) {
        std::cerr << "Error opening WAV file: " << sf_strerror(sndfile) << std::endl;
        return 1;
    }

    std::cout << "Sample rate: " << sfinfo.samplerate << std::endl;
    std::cout << "Channels: " << sfinfo.channels << std::endl;
    std::cout << "Frames: " << sfinfo.frames << std::endl;

    // Read the samples into a buffer
    const int buffer_size = sfinfo.frames * sfinfo.channels;
    double buffer[buffer_size];
    sf_count_t samples_read = sf_read_double(sndfile, buffer, buffer_size);

    std::cout << "Samples read: " << samples_read << std::endl;

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

    auto feats = fbank(waveform, num_mel_bins, frame_length, frame_shift, sample_frequency, 0.0, 0.0, true, true);
    // todo:spec_sub

    // load model
    Context ctx;
    bm_status_t status = ctx.load_bmodel(model.c_str());
    assert(BM_SUCCESS == status);

    // create Network
    std::vector<const char *> network_names;
    ctx.get_network_names(&network_names);
    Network wenet(ctx, network_names[0], 0); // use stage[0]
    assert(wenet.info()->input_num == 6);

    // Initialize the memory space required for the input and output tensors
    auto &inputs = wenet.Inputs();
    auto &outputs = wenet.Outputs();
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
        // inputs[0]->Reshape({1, {1}});                    // chunk_lens, int32
        // inputs[1]->Reshape({5, {1, 12, 4, 80, 128}});    // att_cache, fp32
        // inputs[2]->Reshape({4, {1, 12, 256, 7}});        // cnn_cache, fp32
        // inputs[3]->Reshape({3, {1, 67, 80}});            // chunk_xs, fp32
        // inputs[4]->Reshape({3, {1, 1, 80}});             // cache_mask, fp32
        // inputs[5]->Reshape({2, {1, 1}});                 // offset, int32
        inputs[0]->CopyFrom(void_chunk_lens_ptr);
        inputs[1]->CopyFrom(att_cache);
        inputs[2]->CopyFrom(cnn_cache);
        inputs[3]->CopyFrom(chunk_xs_ptr);
        inputs[4]->CopyFrom(cache_mask);
        inputs[5]->CopyFrom(offset);
    
        status = wenet.Forward();
        assert(BM_SUCCESS == status);
    
        outputs[0]->CopyTo(log_probs);
        outputs[1]->CopyTo(log_probs_idx);
        outputs[2]->CopyTo(chunk_out);
        outputs[3]->CopyTo(chunk_out_lens);
        outputs[4]->CopyTo(offset);
        outputs[5]->CopyTo(att_cache);
        outputs[6]->CopyTo(cnn_cache);
        outputs[7]->CopyTo(cache_mask);
        float *data = static_cast<float*>(chunk_out);
        for(int i = 0; i < 80; i++){
              std::cout << data[i] << ' ';
        }
        std::cout << std::endl;

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