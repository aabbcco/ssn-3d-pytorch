import numpy as np


class write:

    def tobcd(self, data, pointtype, filename):
        
        def generate_header(pointshape, pointtype):
            strs = (
                '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS ')
            if pointtype is 'xyz':
                strs += ('x y z \nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
                formats="%.6f %.6f %.6f"
            elif pointtype is 'xyzl':
                strs += ('x y z label \nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
                formats="%.6f %.6f %.6f %u"
            elif pointtype is 'xyzrgb':
                strs += ('x y z rgb \nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
                formats="%.6f %.6f %.6f %u"
            else:
                raise Exception('unsupported point type')
            strs += ('\nWIDTH ' + str(pointshape)+'\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0' + \
                '\nPOINTS ' + str(pointshape)+'\nDATA ascii')
            return formats, strs
        
        formats,header = generate_header(data.shape[0],pointtype)
            
        if pointtype == 'xyz' or pointtype == 'xyzl':
            np.savetxt(filename, data, fmt=formats, header=header,comments='')
        elif pointtype == 'xyzrgb':
            data[:, 3] = (data[:, 3].astype('uint32') << 16 | data[:, 4].astype('uint32') << 8 | data[:, 5].astype('uint32'))
            np.savetxt(filename, data[:,:4], fmt=formats, header=header,comments='')
            

        

    def toply(self, data, pointtype, filename):

        raise Exception('not supported this moment.plz use plyfile')

        def generate_header(self, pointshape, pointtype):
            strs = 'ply\nformat ascii 1.0\ncomment made by pointcloud io module\n'
            strs += 'element vertex '+str(int(pointshape))+'\n'
            strs += 'property float32 x\nproperty float32 y\n property float32 z\n'
            if 'rgb' in pointtype:
                strs += 'property red uint8\nproperty green uint8\nproperty blue uint8\n'



class _PcdHeadPraser():
    def __init__(self,file=None):
        self.line=0
        if file is not None:
            self.Prase_Head(file)
        else:
            self.fields=[]
            self.width=0
            self.height=0
            self.viewpoint=''
            self.points=0
            self.dataformat=''
            self.fp=None
        super().__init__()
        
    
    def _error(self,message='Prase error! '):
        raise Exception(message+str(self.line))

    def Prase_Head(self,file):
        try:
            self.fp=open(file)
        except OSError:
            print('no such file named',file)
        for i in range(11):
            self.line+=1
            str=(self.fp.readline()).split('\n')[0]
            if i in [2,6,7,8,9,10]:
                obj=str.split(' ')[0]
                if obj=='FIELDS':
                    self.fields=str.split(' ')[1:]
                elif obj=='WIDTH':
                    self.width=int((str.split(' ')[-1]))
                elif obj=='HEIGHT':
                    self.height=int((str.split(' ')[-1]))
                elif obj=='POINTS':
                    self.points=int((str.split(' ')[-1]))
                elif obj=='DATA':
                    self.dataformat=(str.split(' ')[-1])

        self.fp.close()
        

def __pcd_data_formatter(data,header,asdict=False):

    def unpack_rgb(rgb):
        rgbi=(np.expand_dims(rgb,-1)).astype(int)
        r=(np.right_shift(rgbi,16))&0xff
        g=(np.right_shift(rgbi,8))&0xff
        b=rgbi&0xff
        datas=np.concatenate((r,g,b),axis=-1)
        return datas
    
    
    pointer=0
    if asdict:
        element={}
        element['xyz']=data[:,:3]
        if 'rgb' in header.fields:
            element['rgb']=unpack_rgb(data[:,3+pointer])
            pointer+=1
        if 'label' in header.fields:
            element['lebel']=(data[:,3+pointer]).astype(int)
            point+=1

        return element

    else:
        datas=data[:,0:3]
        if 'rgb' in header.fields:
            datas=np.concatenate((datas,unpack_rgb(data[:,3+pointer])),axis=-1)
            pointer+=1
        if 'label' in header.fields:
            datas=np.concatenate((datas,np.expand_dims(data[:,3+pointer],axis=-1)),axis=-1)
            pointer+=1

        return datas
            

    


    
            




def read(filename):

    def __read_ply(filename):
        raise Exception('not implemented yet')

    if filename[-4:] == '.pcd':
        header=_PcdHeadPraser(filename)
        rawdata=np.loadtxt(filename,skiprows=11)
        datas=__pcd_data_formatter(rawdata,header)
        return datas

    elif filename[-4:] == '.ply':
        return __read_ply(filename)
    else:
        print(filename[-4:])
        raise Exception(' invild file or file not supported')



write = write()


if __name__ == "__main__":
    XYZ = './repack.pcd'
    RGB = './repack_rgb.pcd'
    LAB = './repack_l.pcd'
    data = read(XYZ)
    datargb = read(RGB)
    datal=read(LAB)
    print(data.shape,datargb.shape,datal.shape)
    write.tobcd(data, 'xyz', 'repack_new.pcd')
    write.tobcd(datal, 'xyzl', 'repack_l_new.pcd')
    write.tobcd(datargb, 'xyzrgb', 'repack_rgb_new.pcd')
