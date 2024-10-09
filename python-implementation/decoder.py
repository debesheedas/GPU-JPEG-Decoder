from struct import unpack
from huffman_table import HuffmanTable

marker_mapping = {
    0xffd8: "Start Marker",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        f = open(image_file, "rb")
        self.img_data = f.read()
        self.data_chunks = self.extract_chunks()
        for k,v in self.data_chunks.items():
            print(k,v)
            print() 
    
    def extract_chunks(self):
        data_chunks = {}
        data = self.img_data
        i = 0
        # start_ptr_header = i
        while i < len(data):
            marker, = unpack(">H", data[i:i+2])
            if marker == 0xffd8:
                data_chunks[marker_mapping[marker]] = marker
            elif marker == 0xffe0:
                start_ptr_header = i
                start_marker = marker
            elif marker == 0xffdb:
                if marker_mapping[start_marker] not in data_chunks:
                    data_chunks[marker_mapping[start_marker]] = data[start_ptr_header:i]
                    start_ptr_header = i
                    start_marker = marker
                else:
                    data_chunks["Quant_Table_1"] = data[start_ptr_header:i]
                    start_ptr_header = i
                    start_marker = marker
            elif marker == 0xffc0:
                data_chunks["Quant_Table_2"] = data[start_ptr_header:i]
                start_ptr_header = i
                start_marker = marker
            elif marker == 0xffc4:
                if "Start of Frame" not in data_chunks:
                    data_chunks[marker_mapping[start_marker]] = data[start_ptr_header:i]
                    start_marker = marker
                    start_ptr_header = i
                elif "Huffman_Table_1" not in data_chunks:
                    data_chunks["Huffman_Table_1"] = data[start_ptr_header:i]
                    start_marker = marker
                    start_ptr_header = i
                elif "Huffman_Table_2" not in data_chunks:
                    data_chunks["Huffman_Table_2"] = data[start_ptr_header:i]
                    start_marker = marker
                    start_ptr_header = i
                elif "Huffman_Table_3" not in data_chunks:
                    data_chunks["Huffman_Table_3"] = data[start_ptr_header:i]
                    start_marker = marker
                    start_ptr_header = i
            elif marker == 0xffda:
                data_chunks["Huffman_Table_4"] = data[start_ptr_header:i]
                start_ptr_header = i
                start_marker = marker
            elif marker == 0xffd9:
                #TODO: pass the last part
                break
            i = i + 1
        return data_chunks
    
    def decode(self):
        huffman_tables = []
        for i in range(1,5):
            huffman_tables.append(HuffmanTable(self.data_chunks[f'Huffman_Table_{i}']))
        print(huffman_tables)            


    # def decode(self):
    #     data = self.img_data
    #     while(True):
    #         marker, = unpack(">H", data[0:2])
    #         print(marker_mapping.get(marker))
    #         if marker == 0xffd8:
    #             data = data[2:]
    #         elif marker == 0xffd9:
    #             return
    #         elif marker == 0xffda:
    #             data = data[-2:]
    #         else:
    #             lenchunk, = unpack(">H", data[2:4])
    #             data = data[2+lenchunk:]            
    #         if len(data)==0:
    #             break        

if __name__ == "__main__":
    img = JPEG('profile.jpg')

    img.decode()    

# OUTPUT:
# Start of Image
# Application Default Header
# Quantization Table
# Quantization Table
# Start of Frame
# Huffman Table
# Huffman Table
# Huffman Table
# Huffman Table
# Start of Scan
# End of Image