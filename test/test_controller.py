import unittest
import os

from itertools import product

from control import Controller
from algorithm import detect_tile_region


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = Controller()
        self.controller.click_on_rel_pos(self.controller.start_button_rel_coord)

    # def test_tile_clicking(self):
    #     tile_ids = tuple(product(range(self.controller.tile_rows), range(self.controller.tile_cols)))
    #     self.controller.click_on_tiles(tile_ids, interval=0.0)

    def test_img_capturing(self):
        img = self.controller.take_screenshot()
        img.save("../local/raw_test.png")
        region = detect_tile_region(img, self.controller.tile_region_area)
        region.save("../local/test.png")

    def tearDown(self) -> None:
        self.controller.shut_down()



if __name__ == '__main__':
    unittest.main()
