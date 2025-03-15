from abc import ABC, abstractmethod
from typing import Dict, List, Any, Generator

class BaseModel(ABC):
    """基础模型类"""
    def __init__(self, client):
        self.client = client
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict:
        """模型对话接口"""
        pass
    
    def format_messages(self, messages: List[Dict[str, Any]], system_prompt: str) -> List[Dict]:
        """格式化消息"""
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        }
        
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            })
            
        return [system_message] + formatted_messages