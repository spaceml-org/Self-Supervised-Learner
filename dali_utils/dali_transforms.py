import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from torchvision.datasets import ImageFolder


class SimCLRTransform(Pipeline):
    def __init__(
        self,
        DATA_PATH,
        input_height,
        batch_size,
        copies,
        stage,
        num_threads,
        device_id,
        seed=1729,
    ):
        super(SimCLRTransform, self).__init__(
            batch_size, num_threads, device_id, seed=seed
        )

        # this lets our pytorch compat function find the length of our dataset
        self.num_samples = len(ImageFolder(DATA_PATH))

        self.copies = copies
        self.input_height = input_height
        self.stage = stage

        self.input = ops.FileReader(file_root=DATA_PATH, random_shuffle=True, seed=seed)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")
        self.to_int32_cpu = ops.Cast(dtype=types.INT32, device="cpu")

        self.coin = ops.random.CoinFlip(probability=0.5)
        self.uniform = ops.random.Uniform(range=[0.4, 1.5])
        self.blur_amt = ops.random.Uniform(
            values=[float(i) for i in range(1, int(0.1 * self.input_height), 2)]
        )
        self.angles = ops.random.Uniform(range=[0, 360])
        self.cast = ops.Cast(dtype=types.FLOAT, device="gpu")
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self.crop = ops.RandomResizedCrop(
            size=self.input_height,
            minibatch_size=batch_size,
            random_area=[0.5, 1.0],
            device="gpu",
        )
        self.resize = ops.Resize(
            resize_x=self.input_height, resize_y=self.input_height, device="gpu"
        )
        self.flip = ops.Flip(vertical=self.coin(), horizontal=self.coin(), device="gpu")
        self.colorjit_gray = ops.ColorTwist(
            brightness=self.uniform(),
            contrast=self.uniform(),
            hue=self.uniform(),
            saturation=self.uniform(),
            device="gpu",
        )
        self.blur = ops.GaussianBlur(
            window_size=self.to_int32_cpu(self.blur_amt()), device="gpu"
        )
        self.rotate = ops.Rotate(
            angle=self.angles(),
            keep_size=True,
            interp_type=types.DALIInterpType.INTERP_LINEAR,
            device="gpu",
        )
        self.swapaxes = ops.Transpose(perm=[2, 0, 1], device="gpu")

    def train_transform(self, image):
        image = self.rotate(image)
        image = self.flip(image)
        image = self.colorjit_gray(image)
        # image = self.blur(image)
        image = self.crop(image)
        image = self.cast(image)
        image = self.swapaxes(image)
        return image

    def val_transform(self, image):
        image = self.resize(image)
        image = self.cast(image)
        image = self.swapaxes(image)
        return image

    def define_graph(self):
        jpegs, label = self.input()
        jpegs = self.decode(jpegs)

        if self.stage == "train":
            self.transform = self.train_transform
        else:
            self.transform = self.val_transform

        batch = ()
        for i in range(self.copies):
            batch += (self.transform(jpegs),)

        if self.stage is not "inference":
            label = label.gpu()
            label = self.to_int64(label)
            batch += (label,)
        return batch
