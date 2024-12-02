#include <stdio.h>
#include<iostream>
#include <filesystem>

namespace fs = std::filesystem;
std::string path_to_decoder = "/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/jpeglib-implementation/libjpeg_install/build/djpeg";
std::string path_to_py = "/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/jpeglib-implementation/process_ppm.py";
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Please provide the name of the image file to be decompressed." << std::endl;
        return 1;
    }
    std::string imagePath = argv[1];
    fs::path file_path(imagePath);
    file_path.replace_extension(".ppm");
    std::string filename = file_path.filename().string();
    std::string ppm_output = "../testing/jpeglib_output_ppm/" + filename;
    std::string command = path_to_decoder + " -outfile "+ ppm_output + " " + imagePath;
    int ret_code = system(command.c_str());
    if (ret_code != 0) {
        throw std::runtime_error("Command execution failed with code: " + std::to_string(ret_code));
    }
    std::string python_command = "python3 " + path_to_py + " " + ppm_output + " " + "../testing/jpeglib_output_arrays";
    ret_code = system(python_command.c_str());
    if (ret_code != 0) {
        throw std::runtime_error("Command execution failed with code: " + std::to_string(ret_code));
    }
    return 0;
}