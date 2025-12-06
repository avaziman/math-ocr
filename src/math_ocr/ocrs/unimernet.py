import argparse
from pathlib import Path
from typing import override

import numpy as np
import torch
import unimernet.tasks as tasks
from PIL.Image import Image
from unimernet.common.config import Config
from unimernet.processors import load_processor

from math_ocr.math_ocr import MathOCR, MathResult

PACKAGE_ROOT_PATH = Path(__file__).parents[1]


class UnimernetMathOCR(MathOCR):
    def __init__(self):
        config_path = PACKAGE_ROOT_PATH / "configs/unimertest/demo.yaml"
        self.processor = ImageProcessor(config_path)

    @override
    def recognize_math(self, image: Image) -> MathResult:
        latex_code = self.processor.process_single_image(image)
        return MathResult(text=latex_code, confidence=1.0)


class ImageProcessor:
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model, self.vis_processor = self.load_model_and_processor()

    def _patch_relative_path(self, config: Config) -> Config:
        MATH_OCR_PREFIX = "MATH_OCR"

        def patch(old_path: Path) -> str:
            return str(old_path).replace(MATH_OCR_PREFIX, str(PACKAGE_ROOT_PATH))

        config.config.model.pretrained = patch(config.config.model.pretrained)
        config.config.model.tokenizer_config.path = patch(
            config.config.model.tokenizer_config.path
        )
        config.config.model.model_config.model_name = patch(
            config.config.model.model_config.model_name
        )

        return config

    def load_model_and_processor(self):
        args = argparse.Namespace(cfg_path=self.cfg_path, options=None)

        cfg = Config(args)
        cfg = self._patch_relative_path(cfg)

        task = tasks.setup_task(cfg)
        model = task.build_model(cfg).to(self.device)
        vis_processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )

        return model, vis_processor

    def process_single_image(self, image: Image) -> str:
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image)
        # Convert RGB to BGR
        if len(open_cv_image.shape) == 3:
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
        image = self.vis_processor(image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image})
        pred = output["pred_str"][0]

        return pred
