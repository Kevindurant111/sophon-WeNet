#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <system_error>
#include <thread>
#include <map>
#include "ctcdecode.h"


std::vector<std::string> read_dict(const std::string& dict_file);

std::vector<std::string> ctc_decoding(void* log_probs, 
    void* log_probs_idx, 
    void* chunk_out_lens, 
    int beam_size, 
    int batch_size, 
    const std::vector<std::string> &vocabulary, 
    const std::string& mode
);

std::map<std::string, std::string> read_data_lists(const std::string& data_lists_file);