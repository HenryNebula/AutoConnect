import unittest
import numpy as np
from strategy import check_reachability, Position


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        map = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 0]
            ]
        )
        self.padded_map = np.pad(map, pad_width=1, constant_values=1)
        self.max_row, self.max_col = self.padded_map.shape

    def test_padded_shape(self):
        self.assertEqual(self.max_row, 6)
        self.assertEqual(self.max_col, 6)

    def test_start_equals_end(self):
        self.assertEqual(True, check_reachability(self.padded_map, Position(0, 0), Position(0, 0)))
        self.assertEqual(False, check_reachability(self.padded_map,
                                                   Position(self.max_row, self.max_col),
                                                   Position(self.max_row, self.max_col)))

    def test_turns(self):
        self.assertEqual(True,
                         check_reachability(self.padded_map, Position(1, 2), Position(3, 1)),
                         msg="Test reachability in one turn.")
        self.assertEqual(True,
                         check_reachability(self.padded_map, Position(1, 2), Position(4, 1)),
                         msg="Test reachability in two turn.")
        self.assertEqual(True,
                         check_reachability(self.padded_map, Position(3, 1), Position(3, 4)),
                         msg="Test reachability in zero turn.")

    def test_unreachable(self):
        self.assertEqual(False,
                         check_reachability(self.padded_map, Position(1, 2), Position(4, 4)),
                         msg="Not reachable (three turns).")


if __name__ == '__main__':
    unittest.main()
