import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

class SimCLRFinetuneTrainDataTransform(Pipeline):
    def __init__(self, DATA_PATH, input_height, batch_size, num_threads, device_id):
        super(SimCLRFinetuneTrainDataTransform, self).__init__(batch_size, num_threads, device_id, seed = 12)

        self.COPIES = 1

        self.input_height = input_height
        self.input = ops.FileReader(file_root = DATA_PATH, random_shuffle = True, seed = 12)

        self.coin = ops.CoinFlip(probability=0.5)
        self.uniform = ops.Uniform(range = [0.7,1.3]) #-1 to 1
        #read image (I think that has to be cpu, do a mixed operation to decode into gpu)
        self.decode = ops.ImageDecoder(device = 'mixed', output_type = types.RGB)
        self.crop = ops.RandomResizedCrop(size = [self.input_height, self.input_height], device = "gpu")
        self.flip = ops.Flip(vertical = self.coin(), horizontal = self.coin(), device = "gpu")
        self.colorjit_gray = ops.ColorTwist(brightness = self.uniform(), contrast = self.uniform(), hue = self.uniform(), saturation = self.uniform(), device = "gpu")
        self.blur = ops.GaussianBlur(window_size = int(0.1*self.input_height), device = "gpu", dtype = types.FLOAT)
        self.swapaxes = ops.Transpose(perm = [2,0,1], device = "gpu")

        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")

    def define_graph(self):
        jpegs, labels = self.input(name = 'Reader')
        image = self.decode(jpegs)
        image = self.crop(image)
        image = self.flip(image)
        image = self.colorjit_gray(image)
        image = self.blur(image)
        image = self.swapaxes(image)

        labels = labels.gpu()
        labels = self.to_int64(labels)
        return (image, labels)
