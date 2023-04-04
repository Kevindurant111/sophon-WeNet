#include <sndfile.h>
#include <armadillo>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <iostream>
#include "bmruntime_cpp.h"
#include "util.h"
#include "utils.hpp"
#include "wenet.h"

using namespace bmruntime;

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string data_lists_file = "/data/work/sophon-WeNet/datasets/aishell_S0764/aishell_S0764.list";
    auto data_map = read_data_lists(data_lists_file);
    std::string dict_file = "/data/WeNet/config/lang_char.txt";
    std::vector<std::string> dict = read_dict(dict_file);

    std::string model = "/data/WeNet/models/BM1684/wenet_encoder_fp32.bmodel";
    int sample_frequency = 16000;
    int num_mel_bins = 80;
    int frame_shift = 10;
    int frame_length = 25;

    int decoding_chunk_size = 16;
    int subsampling_rate = 4;
    int context = 7;

    std::string result_file = "/data/work/sophon-WeNet/cpp/result.txt";
    // Check if file exists
    std::FILE* file_exists = std::fopen(result_file.c_str(), "r");
    if (file_exists) {
        // File exists, delete it
        std::fclose(file_exists);
        std::remove(result_file.c_str());
    }

    // load model
    auto ctx = std::make_shared<Context>();
    bm_status_t status = ctx->load_bmodel(model.c_str());
    assert(BM_SUCCESS == status);

    WeNet wenet(ctx);
    wenet.Init(dict, sample_frequency, num_mel_bins, frame_shift, frame_length, decoding_chunk_size, subsampling_rate, context);

    // profiling
    TimeStamp wenet_ts;
    wenet.enableProfile(&wenet_ts);

    for (const auto& pair : data_map) {
        const char* file_path = pair.second.c_str();
        auto result = wenet.Recognize(file_path);

        std::cout << "Key: " << pair.first << " , Result: " << result << std::endl;

        // Open file for writing
        std::ofstream result_file_stream(result_file, std::ios::app);

        if (!result_file_stream.is_open()) {
            // Failed to open file, handle error
            std::cerr << "Failed to open file " << result_file << " for writing." << std::endl;
            return 1;
        }
        result_file_stream << pair.first + " " + result + "\n";
        result_file_stream.close();
    }
    // print speed
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    wenet_ts.calbr_basetime(base_time);
    wenet_ts.build_timeline("wenet test");
    wenet_ts.show_summary("wenet test");
    wenet_ts.clear();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "程序运行时间为 " << duration.count() << " 毫秒" << std::endl;
    
    return 0;
}