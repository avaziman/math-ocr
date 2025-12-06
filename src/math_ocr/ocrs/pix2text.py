from typing import Any, override

from PIL.Image import Image
from pix2text import LatexOCR

from math_ocr.math_ocr import MathOCR, MathResult


class Pix2TextMathOCR(MathOCR):
    def __init__(self):
        self.latex_ocr = LatexOCR(model="mfr-1.5")

    @override
    def recognize_math(self, image: Image) -> MathResult:
        result: dict[str, Any] = self.latex_ocr.recognize(image)
        return MathResult(text=result["text"], confidence=result["score"])
