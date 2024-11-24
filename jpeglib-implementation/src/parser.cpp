#include "parser.h"

JPEGParser::JPEGParser(std::string& imagePath) {
    fs::path file_path(imagePath);
    this->filename = file_path.filename().string();
    std::ifstream input(imagePath, std::ios::binary);
    this->readBytes = std::vector<uint8_t>((std::istreambuf_iterator<char>(input)),
                                           (std::istreambuf_iterator<char>()));
    input.close();
}

void JPEGParser::decode() {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    // Feed JPEG data to the decompression structure
    jpeg_mem_src(&cinfo, this->readBytes.data(), this->readBytes.size());

    // Read header and start decompression
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    // Ensure image dimensions match
    this->width = cinfo.output_width;
    this->height = cinfo.output_height;
    int numChannels = cinfo.output_components;

    if (numChannels != 3) {
        throw std::runtime_error("Unexpected number of channels! Only RGB images are supported.");
    }

    // Allocate memory for decompressed data
    size_t rowStride = this->width * numChannels;
    std::vector<uint8_t> buffer(rowStride);
    this->decompressedData.resize(this->width * this->height * numChannels);

    size_t currentRow = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW rowPointer[1] = {buffer.data()};
        jpeg_read_scanlines(&cinfo, rowPointer, 1);

        // Copy the row data to the decompressedData buffer
        std::copy(buffer.begin(), buffer.end(),
                  this->decompressedData.begin() + currentRow * rowStride);
        currentRow++;
    }

    // Stop decompression
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    // Deinterleave the RGB data
    size_t totalPixels = this->width * this->height;
    std::vector<uint8_t> redChannel(totalPixels);
    std::vector<uint8_t> greenChannel(totalPixels);
    std::vector<uint8_t> blueChannel(totalPixels);

    for (size_t i = 0; i < totalPixels; ++i) {
        redChannel[i] = this->decompressedData[i * numChannels];
        greenChannel[i] = this->decompressedData[i * numChannels + 1];
        blueChannel[i] = this->decompressedData[i * numChannels + 2];
    }

    // Concatenate the channels back to the decompressedData vector in channel-major order
    this->decompressedData.clear();
    this->decompressedData.insert(this->decompressedData.end(), redChannel.begin(), redChannel.end());
    this->decompressedData.insert(this->decompressedData.end(), greenChannel.begin(), greenChannel.end());
    this->decompressedData.insert(this->decompressedData.end(), blueChannel.begin(), blueChannel.end());
}

void JPEGParser::write() {
    // Output directory and file path
    fs::path output_dir = "../testing/jpeglib_output_arrays";
    fs::path full_path = output_dir / this->filename;
    full_path.replace_extension(".array");

    // Open file for writing
    std::ofstream outfile(full_path);
    if (!outfile) {
        throw std::runtime_error("Failed to open output file for writing");
    }

    // Write dimensions
    outfile << this->height << " " << this->width << std::endl;

    // Calculate total number of pixels
    size_t totalPixels = this->width * this->height;

    // Validate decompressed data size
    if (this->decompressedData.size() != totalPixels * 3) {
        throw std::runtime_error("Decompressed data size mismatch!");
    }

    // Write red channel (first third of the data)
    for (size_t i = 0; i < totalPixels; ++i) {
        outfile << static_cast<int>(this->decompressedData[i]) << " ";
    }
    outfile << std::endl;

    // Write green channel (second third of the data)
    for (size_t i = 0; i < totalPixels; ++i) {
        outfile << static_cast<int>(this->decompressedData[totalPixels + i]) << " ";
    }
    outfile << std::endl;

    // Write blue channel (final third of the data)
    for (size_t i = 0; i < totalPixels; ++i) {
        outfile << static_cast<int>(this->decompressedData[2 * totalPixels + i]) << " ";
    }
    outfile << std::endl;

    // Close the file
    outfile.close();
}