import numpy as np
import OpenEXR, Imath


def load_exr(fname):
    dmap_exr_in = OpenEXR.InputFile(fname)
    dw = dmap_exr_in.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    ystr = dmap_exr_in.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
    y = np.frombuffer(ystr, dtype=np.float32)
    y.shape = (size[1], size[0])
    return y


def save_exr(fname, image, compression=Imath.Compression(Imath.Compression.DWAA_COMPRESSION)):
    dmap_header = OpenEXR.Header(image.shape[1], image.shape[0])
    dmap_header['channels'] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    dmap_header['compression'] = compression
    dmap_exrfile = OpenEXR.OutputFile(fname, dmap_header)
    dmap_exrfile.writePixels({"Y": image})
    dmap_exrfile.close()
