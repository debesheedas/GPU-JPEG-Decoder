from struct import unpack

class Stream:
    def __init__(self, data):
        self.data= data
        self.pos = 0

    def GetBit(self):
        b = self.data[self.pos >> 3]
        s = 7-(self.pos & 0x7)
        self.pos+=1
        return (b >> s) & 1

    def GetBitN(self, l):
        val = 0
        for i in range(l):
            val = val*2 + self.GetBit()
        return val

class HuffmanTable:
    def __init__(self, data):
        self.root=[]
        self.elements = []
        self.header, self.lengths, self.elements = self.extract_table(data)
        self.GetHuffmanBits(self.lengths, self.elements)

    def extract_table(self, data):
        offset = 0
        header, = unpack("B",data[offset:offset+1])
        offset += 1

        # Extract the 16 bytes containing length data
        lengths = unpack("BBBBBBBBBBBBBBBB", data[offset:offset+16]) 
        offset += 16

        # Extract the elements after the initial 16 bytes
        elements = []
        for i in lengths:
            elements += (unpack("B"*i, data[offset:offset+i]))
            offset += i 

        print("Header: ",header)
        print("lengths: ", lengths)
        print("Elements: ", len(elements))
        data = data[offset:]
        return header, lengths, elements
    
    def BitsFromLengths(self, root, element, pos):
        if isinstance(root,list):
            if pos==0:
                if len(root)<2:
                    root.append(element)
                    return True                
                return False
            for i in [0,1]:
                if len(root) == i:
                    root.append([])
                if self.BitsFromLengths(root[i], element, pos-1) == True:
                    return True
        return False
    
    def GetHuffmanBits(self,  lengths, elements):
        ii = 0
        for i in range(len(lengths)):
            for j in range(lengths[i]):
                self.BitsFromLengths(self.root, elements[ii], i)
                ii+=1

    def Find(self,st):
        r = self.root
        while isinstance(r, list):
            r=r[st.GetBit()]
        return  r 

    def GetCode(self, st):
        while(True):
            res = self.Find(st)
            if res == 0:
                return 0
            elif ( res != -1):
                return res