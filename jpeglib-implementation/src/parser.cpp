#include "parser.h"

JPEGParser::JPEGParser(std::string& imagePath){
    // Extract the file name of the image file from the file path
    fs::path file_path(imagePath);
    this->filename = file_path.filename().string();
    std::ifstream input(imagePath, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
    this->readBytes = bytes;
    input.close();
    // JPEGParser::extract(bytes);
}

void JPEGParser::extract() {        
    Stream* stream = new Stream(this->readBytes);

    while (true) {
        uint16_t marker = stream->getMarker();

        if (marker == MARKERS[0]) {
            // Skip this marker.
            continue;
        } else if (marker == MARKERS[3]) { // Start of Frame
            uint16_t tableSize = stream->getMarker(); // Read size of the frame
            stream->getNBytes(this->startOfFrame, (int)tableSize - 2);

            // Parse dimensions from the Start of Frame header
            Stream* frame = new Stream(this->startOfFrame);
            frame->getByte(); // Skip precision
            this->height = frame->getMarker();
            this->width = frame->getMarker();

            delete frame; // Clean up allocated Stream object
            break; // Stop parsing after extracting dimensions
        } else if (marker == MARKERS[5]) {
            // No need to parse image data for this task, stop processing.
            break;
        }
    }

    delete stream; // Clean up allocated Stream object  
}

void JPEGParser::decode() {
    // Initialize JPEG compression object
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // Set up in-memory destination
    unsigned char* mem_buffer = nullptr; // Pointer to hold JPEG data
    unsigned long mem_size = 0;          // Size of the JPEG data
    jpeg_mem_dest(&cinfo, &mem_buffer, &mem_size);

    // Set image properties
    cinfo.image_width = this->width;
    cinfo.image_height = this->height;
    cinfo.input_components = 3; // Assume RGB
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 100, TRUE);

    // Start compression
    jpeg_start_compress(&cinfo, TRUE);

    // Write scanlines
    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = const_cast<unsigned char*>(&readBytes.data()[cinfo.next_scanline * width * 3]);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // Finish compression
    jpeg_finish_compress(&cinfo);

    // Copy data to output buffer
    imageData.assign(mem_buffer, mem_buffer + mem_size);

    // Clean up
    jpeg_destroy_compress(&cinfo);
    free(mem_buffer); // Release memory allocated by libjpeg

}

void JPEGParser::write() {
    // Writing the decoded channels to a file instead of displaying using opencv
    fs::path output_dir = "../testing/jpeglib_output_arrays"; // Change the directory name here for future CUDA implementations
    fs::path full_path = output_dir / this->filename;
    full_path.replace_extension(".array");
    std::ofstream outfile(full_path);
    outfile << this->height << " " << this->width << std::endl;
    for (size_t i = 0; i < this->imageData.size(); ++i) {
        outfile << static_cast<int>(this->imageData[i]); // Cast byte to int for numeric output
        if (i < this->imageData.size() - 1) {
            outfile << " "; // Add a space after each value except the last
        }
    }
    outfile << std::endl;
    outfile.close();
}