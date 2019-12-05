from data.FishEyeGenerator import FishEyeGenerator
import numpy as np
import os
import cv2

DATA_SET_DIR = "E:/DataSets/CityScape/"

dst_dir = "F:\\Code\\Github\\FisheyeSeg\\DataSets\\CityScape\\val_rotate10\\"
dst_annot_dir = "F:\\Code\\Github\\FisheyeSeg\\DataSets\\CityScape\\val_rotate10_annot\\"
VAL_DIR = "F:\\Code\\Github\\FisheyeSeg\\DataSets\\CityScape\\val\\"
VAL_ANNOT_DIR = "F:\\Code\\Github\\FisheyeSeg\\DataSets\\CityScape\\valannot\\"

class FESetsGenerator:

    def __init__(self, dst_shape, focal_len=350):
        self._generator = FishEyeGenerator(focal_len, dst_shape)

        self._F_RAND_FLAG = False
        self._F_RANGE = [200,400]

        self._EXT_RAND_FLAG = False
        self._EXT_PARAM_RANGE = [5, 5, 10, 0.3, 0.3, 0.4]
        self._generator.set_ext_param_range(self._EXT_PARAM_RANGE)

    def set_ext_param_range(self,ext_param):
        for i in range(6):
            self._EXT_PARAM_RANGE[i] = ext_param[i]
        self._generator.set_ext_param_range(self._EXT_PARAM_RANGE)

    def rand_ext_params(self):
        self._EXT_RAND_FLAG = True

    def set_ext_params(self,ext_params):
        self._generator.set_ext_params(ext_params)
        self._EXT_RAND_FLAG = False


    def set_f(self, focal_len):
        self._generator.set_f(focal_len)
        self._F_RAND_FLAG = False

    def rand_f(self, f_range=[200,400]):
        self._F_RANGE = f_range
        self._F_RAND_FLAG = True


    def generate(self,src_dir, src_annot_dir, dst_dir, dst_annot_dir, prefix):

        image_list = sorted([image for image in os.listdir(src_dir) if image.endswith(".png")])
    
        count =0
        for image in image_list:
            src_image = cv2.imread(src_dir+image)
            src_annot_image = cv2.imread(src_annot_dir+image, 0)

            if self._F_RAND_FLAG:
                self._generator.rand_f(self._F_RANGE)
            if self._EXT_RAND_FLAG:
                self._generator.rand_ext_params()

            result1 = self._generator.transFromColor(src_image)
            cv2.imwrite(dst_dir+prefix+image, result1)
            print("Image", count, "Done!")
            result2 = self._generator.transFromGray(src_annot_image)
            cv2.imwrite(dst_annot_dir+prefix+image, result2)
            print("Image annot", count, "Done!")
            count += 1
        print("ALL Done!")
    
def test():
    DT = FESetsGenerator([640,640], focal_len=350)
    DT.set_ext_param_range([10,10,10,0,0,0])
    DT.rand_ext_params()
    # DT.rand_f()
    DT.generate(VAL_DIR, VAL_ANNOT_DIR, dst_dir, dst_annot_dir, prefix='r')

if __name__ == "__main__":
    test()

