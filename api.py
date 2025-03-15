from openai import OpenAI
from models.language_model import LanguageModel
from models.vision_model import VisionModel
from typing import Union, Dict, Generator

class APIClient:
    def __init__(self):
        self.client = OpenAI(
            api_key="your api key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.language_model = LanguageModel(self.client)
        self.vision_model = VisionModel(self.client)
    
    def llm_chat(self, messages: list, tools: list = None) -> Union[Dict, Generator[str, None, None]]:
        """语言模型对话"""
        return self.language_model.chat(messages, tools)
    
    def sendPicture(self, image_path: str):
        """视觉模型分析图片"""
        return self.vision_model.chat([], image_path)
