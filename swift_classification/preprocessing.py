import cv2
from skimage import transform
from skimage.feature import hog


class Decode(object):
    """Decode per-item byte arrays to recover image matrices.

    Args:
        -color_type (int): Flag specifying the color type of a
        loaded image. See OpenCV docs for specific flag options:

        (https://docs.opencv.org/2.4/modules/highgui/doc/
        reading_and_writing_images_and_video.html#imread)
    """

    def __init__(self, flags):
        self.flags = flags

    def __call__(self, bytestream):
        return cv2.imdecode(bytestream, flags=self.flags)


class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.dim = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.dim = output_size

    def __call__(self, img):
        return transform.resize(img, self.dim)


class HOG(object):
    def __init__(self):
        pass  # Using defaults params for now, may change later

    def __call__(self, img):
        return hog(img)
