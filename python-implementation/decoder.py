from struct import unpack
from huffman_table import HuffmanTable
from huffman_tree import HuffmanTree, HuffmanTreeNode
class JPEG:
    def __init__(self, image_file):
        f = open(image_file, "rb")
        self.img_data = f.read()
        self.quantMapping = []
        self.width = 0
        self.height = 0

        # Extracting the data parts of the JPEG image.
        self.data_chunks = self.extract_chunks()

    # 0xFF is a special character, if it is in the image it is followed by 0x00 for distinction. 
    # When decoding, we need to remove these 00's.
    def remove_byte_stuffing(self, data):
        datapro = []
        i = 0
        while(True):
            b,bnext = unpack("BB",data[i:i+2])  
            if (b == 0xff):
                if (bnext != 0x00):
                    break
                datapro.append(data[i])
                i+=2
            else:
                datapro.append(data[i])
                i+=1
        return datapro,i
    
    def start_of_scan(self, data, header_len):
        data,lenchunk = self.remove_byte_stuffing(data[header_len:])
        oldlumdccoeff, oldCbdccoeff, oldCrdccoeff = 0, 0, 0
        for y in range(self.height//8):
            for x in range(self.width//8):
                # TODO: build matrix      
                pass 
        return lenchunk+header_len
    
    def parse_sof(self, data):
        hdr, self.height, self.width, components = unpack(">BHHB",data[0:6])
        print("size %ix%i" % (self.width,  self.height))

        for i in range(components):
            id, samp, QtbId = unpack("BBB",data[6+i*3:9+i*3])
            self.quantMapping.append(QtbId)      

    def extract_chunks(self):
        data_chunks = {}
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            if marker == 0xFFD8:
                data = data[2:]
            elif marker == 0xFFD9:
                return data_chunks
            else:
                (len_chunk,) = unpack(">H", data[2:4]) # Length of the segment
                len_chunk += 2 # Adding the inital FFDA part for complete header length
                chunk = data[4:len_chunk]
                if marker == 0xFFC4:
                    if 'Huffman_Tables' not in data_chunks:
                        data_chunks['Huffman_Tables'] = []
                    data_chunks['Huffman_Tables'].append(chunk)
                elif marker == 0xFFDB:
                    if 'Quant_Tables' not in data_chunks:
                        data_chunks["Quant_Tables"] = []
                    data_chunks["Quant_Tables"].append(chunk)
                elif marker == 0xFFC0:
                    data_chunks['Start_Of_Frame'] = chunk
                    self.parse_sof(data_chunks['Start_Of_Frame'])
                elif marker == 0xFFDA:
                    len_chunk = self.start_of_scan(data, len_chunk)
                    data_chunks['Image'] = data
                data = data[len_chunk:]
            if len(data) == 0:
                break
        
    def print_8x8_matrix(self, binary_data):
        # Unpack the binary data into 64 unsigned 8-bit integers (64 "B" format characters)
        unpacked_data = unpack("64B", binary_data)
        
        # Format and print the 8x8 matrix
        print("8x8 Matrix (Decimal Values):")
        for i in range(8):
            # Print 8 values per row (i * 8 to (i+1) * 8)
            row = unpacked_data[i * 8:(i + 1) * 8]
            print(" ".join(f"{value:2d}" for value in row)) 


    def decode(self):
        # Step 1: Extracting Huffman Tables
        huffman_tables = []
        for i in range(4):
            huffman_tables.append(HuffmanTable(self.data_chunks['Huffman_Tables'][i]))

        root = HuffmanTreeNode(internal=True)
        tree =  HuffmanTree(self.data_chunks['Huffman_Tables'][0], root)
        for item in tree.elements:
            tree.add_to_tree(root, item, item.length)
        tree.decode_tree(tree.root, '')
        for item,val in tree.codes.items():
            print(item, ' ' , val)

        # Step 2: Extracting Quantization Values
        print(self.data_chunks['Quant_Tables'][0][1:]) # 0 for luminance
        print(len(self.data_chunks['Quant_Tables'][0][1:]))
        print(self.print_8x8_matrix(self.data_chunks['Quant_Tables'][0][1:]))
        print(self.data_chunks['Quant_Tables'][1][1:]) # 1 for others

if __name__ == "__main__":
    img = JPEG('profile.jpg')
    img.decode()    