"""
Engine chay Function (YAML): load functions, thuc thi tung step.
Step types: match_click, sleep, click_position, wait_until_match, key_press.
"""
import os
import time
import logging
import random
import yaml
import pyautogui
import cv2 as cv
from vision import Vision

log = logging.getLogger("kha_lastz")


def load_functions(functions_dir="functions"):
    """Load tat ca file YAML trong functions_dir. Tra ve dict: ten_function -> { description, steps }."""
    result = {}
    if not os.path.isdir(functions_dir):
        return result
    for fname in os.listdir(functions_dir):
        if not fname.endswith(".yaml") and not fname.endswith(".yml"):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(functions_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        result[name] = {
            "description": data.get("description", ""),
            "steps": data.get("steps", []),
        }
    return result


def collect_templates(functions_dict):
    """Lay danh sach duong dan template tu tat ca steps de tao vision cache."""
    templates = set()
    for fn in functions_dict.values():
        for step in fn["steps"]:
            if step.get("type") == "match_click" and step.get("template"):
                templates.add(step["template"])
            if step.get("type") == "wait_until_match" and step.get("template"):
                templates.add(step["template"])
    return list(templates)


def build_vision_cache(template_paths):
    """Tao dict template_path -> Vision(...)."""
    cache = {}
    for path in template_paths:
        try:
            cache[path] = Vision(path)
        except Exception as e:
            log.info("[bot_engine] Failed to load template {}: {}".format(path, e))
    return cache


class FunctionRunner:
    """Chay 1 function: giu state step hien tai, xu ly tung step theo type."""

    def __init__(self, vision_cache):
        self.vision_cache = vision_cache
        self.functions = {}
        self.state = "idle"  # idle | running
        self.function_name = None
        self.steps = []
        self.step_index = 0
        self.step_start_time = None
        self.step_click_count = 0
        self.wincap = None

    def load(self, functions_dict):
        self.functions = functions_dict

    def start(self, function_name, wincap):
        if function_name not in self.functions:
            log.info("[Runner] Function not found: {}".format(function_name))
            return False
        self.function_name = function_name
        self.steps = self.functions[function_name]["steps"]
        self.step_index = 0
        self.step_start_time = time.time()
        self.step_click_count = 0
        self.wincap = wincap
        self.state = "running"
        log.info("[Runner] Started function: {}".format(function_name))
        return True

    def stop(self):
        self.state = "idle"
        self.function_name = None

    def update(self, screenshot, wincap):
        """Tra ve 'running' | 'done' | 'idle'. Neu running thi xu ly step hien tai."""
        if self.state != "running" or screenshot is None or self.step_index >= len(self.steps):
            if self.state == "running" and self.step_index >= len(self.steps):
                self.state = "idle"
                log.info("[Runner] Finished function: {}".format(self.function_name))
                return "done"
            return "idle" if self.state == "idle" else "running"

        self.wincap = wincap
        step = self.steps[self.step_index]
        step_type = step.get("type", "")

        now = time.time()
        if self.step_start_time is None:
            self.step_start_time = now

        if step_type == "match_click":
            template = step.get("template")
            threshold = step.get("threshold", 0.75)
            one_shot = step.get("one_shot", True)
            timeout_sec = step.get("timeout_sec") or 999
            max_clicks = step.get("max_clicks") or 999999
            click_interval_sec = step.get("click_interval_sec") or 0
            click_random_offset = step.get("click_random_offset") or 0  # pixel, ± xung quanh tam

            vision = self.vision_cache.get(template)
            if not vision:
                self._advance_step()
                return "running"
            points = vision.find(screenshot, threshold=threshold, debug_mode=None)
            if points:
                center = list(points[0])
                if click_random_offset > 0:
                    center[0] += random.randint(-click_random_offset, click_random_offset)
                    center[1] += random.randint(-click_random_offset, click_random_offset)
                sx, sy = wincap.get_screen_position(tuple(center))
                pyautogui.click(sx, sy)
                self.step_click_count += 1
                # Log de thay auto click (moi 10 lan hoac lan dau hoac one_shot)
                if one_shot or self.step_click_count <= 1 or self.step_click_count % 10 == 0:
                    log.info("[Runner] match_click {} (count {})".format(template, self.step_click_count))
                if not one_shot and click_interval_sec > 0:
                    time.sleep(click_interval_sec)
                if one_shot:
                    self._advance_step()
                    return "running"
                if self.step_click_count >= max_clicks:
                    log.info("[Runner] match_click {} reached max_clicks={}".format(template, max_clicks))
                    self._advance_step()
                    return "running"
            # timeout for this step
            if now - self.step_start_time >= timeout_sec:
                log.info("[Runner] match_click {} timeout (template not found in {}s)".format(template, timeout_sec))
                self._advance_step()
            return "running"

        if step_type == "sleep":
            duration = step.get("duration_sec", 0)
            if now - self.step_start_time >= duration:
                self._advance_step()
            return "running"

        if step_type == "click_position":
            ox = step.get("offset_x", 0.15)
            oy = step.get("offset_y", 0.15)
            px = int(wincap.w * ox)
            py = int(wincap.h * oy)
            sx, sy = wincap.get_screen_position((px, py))
            pyautogui.click(sx, sy)
            log.info("[Runner] click_position ({}, {})".format(sx, sy))
            self._advance_step()
            return "running"

        if step_type == "wait_until_match":
            template = step.get("template")
            threshold = step.get("threshold", 0.75)
            timeout_sec = step.get("timeout_sec") or 30
            vision = self.vision_cache.get(template)
            if not vision:
                self._advance_step()
                return "running"
            points = vision.find(screenshot, threshold=threshold, debug_mode=None)
            if points:
                self._advance_step()
                return "running"
            if now - self.step_start_time >= timeout_sec:
                self._advance_step()
            return "running"

        if step_type == "key_press":
            key = step.get("key", "")
            if key:
                pyautogui.press(key)
            self._advance_step()
            return "running"

        # unknown type -> skip
        self._advance_step()
        return "running"

    def _advance_step(self):
        self.step_index += 1
        self.step_start_time = time.time()
        self.step_click_count = 0


def load_config(config_path="config.yaml"):
    """Load config.yaml. Tra ve dict co key_bindings, schedules."""
    if not os.path.isfile(config_path):
        return {"key_bindings": {}, "schedules": []}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"key_bindings": {}, "schedules": []}
