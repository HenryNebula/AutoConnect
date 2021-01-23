import time
from pathlib import Path
import threading
from datetime import datetime
from typing import Tuple
import logging

import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image

from pywinauto.application import Application
from pywinauto.win32structures import RECT
from apscheduler.schedulers import SchedulerNotRunningError
from apscheduler.schedulers.background import BackgroundScheduler
import keyboard

import algorithm
import strategy

EVENT_TIMEOUT = 3


class Controller:
    def __init__(self, agent: strategy.BaseAgent, patience=10, log_dir="log/", data_dir="data/env/"):
        project_root_dir = Path(__file__).parent
        self.app = Application()
        flashplayer_path = (project_root_dir / Path("static/flashplayer.exe")).resolve()
        game_swf_path = (project_root_dir / Path("static/game.swf")).resolve()
        self.app.start(f"{flashplayer_path} {game_swf_path}")
        self.window = self.app["Adobe Flash Player"].wrapper_object()
        self.window_rect = self.window.rectangle()

        self.agent = agent
        self.patience = patience
        self.log_dir = project_root_dir / Path(log_dir)

        if data_dir is not None:
            data_dir = project_root_dir / Path(data_dir)
            self.pos_data_dir = data_dir / Path("pos/")
            self.neg_data_dir = data_dir / Path("neg/")
            for dir_ in [self.log_dir, self.pos_data_dir, self.neg_data_dir]:
                if not dir_.exists():
                    dir_.mkdir(parents=True)
        else:
            self.pos_data_dir, self.neg_data_dir = None, None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / Path(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
                logging.StreamHandler()
            ]
        )

        self.region_coordinates = None
        self.start_button_rel_coord = (450, 425)
        self.tile_size = 39
        self.tile_rows = 8
        self.tile_cols = 12
        self.tile_region_area = self.tile_rows * self.tile_cols * self.tile_size ** 2

        self.load_game()
        self.exit_event = threading.Event()
        self.end_signal_received = False
        keyboard.add_hotkey("ctrl+c", self.set_end_signal_received)

        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.check_game_over, "interval", seconds=2, next_run_time=datetime.now())
        logging.getLogger('apscheduler.executors.default').setLevel(logging.ERROR)
        self.scheduler.start()

    def set_end_signal_received(self):
        logging.info("End signal (Ctrl+C) received, will shutdown the controller after executing the current strategy.")
        self.end_signal_received = True

    def load_game(self):
        self.click_on_rel_pos(coords=self.start_button_rel_coord)
        time.sleep(0.5)
        img = self.take_screenshot()
        if self.region_coordinates is None:
            try:
                region, self.region_coordinates = algorithm.detect_tile_region(img, area=self.tile_region_area)
            except AssertionError:
                logging.warning("Fail to detect the tile region.")

    def click_on_rel_pos(self, coords: Tuple[int, int], explicit_release=False):
        self.window.press_mouse_input(button='left',
                                      coords=coords,
                                      pressed='',
                                      absolute=False)
        if explicit_release:
            self.window.release_mouse_input(button='left',
                                            coords=coords,
                                            pressed='',
                                            absolute=False)

    def tile_id_to_rel_pos(self, tile_id):
        row, col = tile_id  # row and col starts from 1 due to padding
        x = (col - 1) * self.tile_size + self.region_coordinates[0]
        y = (row - 1) * self.tile_size + self.region_coordinates[1]
        return x, y

    def click_on_tiles(self, tile_ids: Tuple[Tuple[int, int]], interval=0.):
        for tile_id in tile_ids:
            x, y = self.tile_id_to_rel_pos(tile_id)
            # move to center
            x += self.tile_size // 2
            y += self.tile_size // 2
            self.click_on_rel_pos((x, y))
            if interval != 0:
                time.sleep(interval)

    def execute_strategy(self):
        logging.info("Reset all control parameters and start executing strategies.")
        clicked_pairs_cnt = 0
        missed_pairs_cnt = 0
        img = self.take_screenshot()
        base_tiles = None
        no_move_rounds = 0

        while no_move_rounds <= self.patience:

            if self.region_coordinates is None:
                return
            else:
                region, _ = algorithm.detect_tile_region(img, area=self.tile_region_area,
                                                         crop_coordinates=self.region_coordinates)

            if self.end_signal_received:
                return

            tiles = algorithm.crop_tiles(region, self.tile_size)
            if base_tiles is None:
                base_tiles = tiles

            # if bg_mask is None:
            #     bg_mask = np.zeros_like(algorithm.generate_background_mask(tiles, pad=True))
            # elif bg_mask is not None and np.all(bg_mask == 0) and missed_pairs_cnt < clicked_pairs_cnt:
            #     # only generate the mask using algorithms after the first run under all-zero initialization
            #     bg_mask = algorithm.generate_background_mask(tiles, pad=True)

            bg_mask = algorithm.generate_background_mask(tiles, pad=True)
            if missed_pairs_cnt == clicked_pairs_cnt:
                bg_mask = np.zeros_like(bg_mask)
            elif not algorithm.check_background_consistency(bg_mask, clicked_pairs_cnt-missed_pairs_cnt):
                logging.info("Switch to slower background detection algorithm.")
                bg_mask = algorithm.generate_background_mask_v2(base_tiles, tiles, pad=True)
                algorithm.check_background_consistency(bg_mask, clicked_pairs_cnt-missed_pairs_cnt)

            strategy: Tuple[Tuple[Tuple[int, int]]] = self.agent.generate_strategy(bg_mask, tiles)

            # check exit event first in case the strategy is empty
            res = self.exit_event.wait(EVENT_TIMEOUT)
            if not res:
                logging.info("Exit event triggered.")
                return
            for step in strategy:
                res = self.exit_event.wait(EVENT_TIMEOUT)
                if not res:
                    logging.info("Exit event triggered.")
                    return
                self.click_on_tiles(step)

            new_img = self.take_screenshot()

            for step in strategy:
                clicked_pairs_cnt += 1
                eliminated, conf = self.check_step_elimination(img, new_img, step)
                logging.info(f"Click on pair {step[:2]} : {'NOT' if not eliminated else ''} eliminated ({conf}). ")
                if not eliminated:
                    missed_pairs_cnt += 1

            img = new_img

            if strategy:
                logging.info(f"Execution summary: hit rate "
                             f"{clicked_pairs_cnt - missed_pairs_cnt} / {clicked_pairs_cnt}.")
            else:
                logging.info(f"Experienced {no_move_rounds} / {self.patience} no move rounds.")
                no_move_rounds += 1

            self.exit_event.clear()

    def take_screenshot(self, coords=None):
        self.window.set_focus()
        rect = self.window_rect if coords is None else RECT(*coords)
        img = self.window.capture_as_image(rect)
        return img

    def check_step_elimination(self, old_screenshot, new_screenshot, step):
        sim = -float("inf")
        old_concat_tiles = Image.new("RGB", size=(self.tile_size * 2, self.tile_size))
        new_concat_tiles = Image.new("RGB", size=(self.tile_size * 2, self.tile_size))

        for i, tile_id in enumerate(step):
            left, upper = self.tile_id_to_rel_pos(tile_id)
            right, buttom = left + self.tile_size, upper + self.tile_size
            new_tile = new_screenshot.crop((left, upper, right, buttom))
            old_tile = old_screenshot.crop((left, upper, right, buttom))
            sim = max(sim, structural_similarity(np.array(new_tile), np.array(old_tile),
                                                 win_size=self.tile_size,
                                                 multichannel=True))
            old_concat_tiles.paste(old_tile, (i * self.tile_size, 0))
            new_concat_tiles.paste(new_tile, (i * self.tile_size, 0))

        eliminated = sim < 0.5
        if self.pos_data_dir is not None and self.neg_data_dir is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if eliminated:
                old_concat_tiles.save((self.pos_data_dir / Path(f"{ts}.bmp")).resolve())

            # whether eliminated or not, the new tiles are always not a pair
            new_concat_tiles.save((self.neg_data_dir / Path(f"{ts}.bmp")).resolve())

        return eliminated, sim

    def check_game_over(self):
        img = self.take_screenshot()

        for p in ["static/restart.png", "static/next_level.png"]:
            pattern = Image.open(Path(__file__).parent / Path(p))
            matched_coords = algorithm.pattern_matching(img, pattern)

            if matched_coords is not None:
                x, y, width, height = matched_coords
                logging.info("Restart menu detected.")
                self.exit_event.clear()
                time.sleep(2*EVENT_TIMEOUT)
                logging.info("Restart a new round.")

                self.click_on_rel_pos((x + width // 2, y + height // 2), explicit_release=True)
                time.sleep(2 * EVENT_TIMEOUT)
                self.load_game()
                self.exit_event.set()
                self.execute_strategy()
                break

        if not self.exit_event.is_set():
            self.exit_event.set()

    def run(self):
        self.execute_strategy()
        while not self.end_signal_received:
            pass
        self.shut_down()

    def shut_down(self):
        logging.info("Shutting down the controller ...")
        try:
            self.scheduler.shutdown(wait=True)
        except SchedulerNotRunningError:
            logging.warning("First run of the scheduler has not started.")

        self.app.kill()

    def __del__(self):
        self.shut_down()
