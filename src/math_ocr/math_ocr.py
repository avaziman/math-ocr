from PIL.Image import Image
from abc import ABC, abstractmethod
from pydantic import BaseModel


class MathResult(BaseModel):
    text: str
    confidence: float


class MathOCR(ABC):
    @abstractmethod
    def recognize_math(self, image: Image) -> MathResult:
        pass
