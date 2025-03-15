from .base_model import BaseModel
from typing import Dict, List, Any, Generator
import base64
import os

class VisionModel(BaseModel):
    """视觉模型"""
    def __init__(self, client):
        super().__init__(client)
        self.model_name = "qwen-vl-plus"
        self.system_prompt = "You are a helpful assistant. Answer in Chinese."
    
    def encode_image(self, image_path: str) -> str:
        """Base64编码图片"""
        try:
            print(f"[DEBUG] Encoding image from path: {image_path}")
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                print("[DEBUG] Image encoded successfully")
                return encoded
        except Exception as e:
            print(f"[DEBUG] Error encoding image: {str(e)}")
            raise
    
    def chat(self, messages: List[Dict[str, Any]], image_path: str) -> Generator[str, None, None]:
        """视觉模型对话"""
        try:
            print(f"[DEBUG] Starting vision chat with image: {image_path}")
            
            # Handle file:// prefix
            if image_path.startswith("file://"):
                image_path = image_path[7:]
                print(f"[DEBUG] Removed file:// prefix, new path: {image_path}")
            
            # Verify file exists
            if not os.path.exists(image_path):
                print(f"[DEBUG] Image file not found: {image_path}")
                yield f"错误：找不到图片文件 {image_path}"
                return
            
            # Get image format
            image_format = image_path.lower().split('.')[-1]
            if image_format == 'jpg':
                image_format = 'jpeg'
            print(f"[DEBUG] Image format: {image_format}")
            
            # Encode image
            try:
                base64_image = self.encode_image(image_path)
                print("[DEBUG] Image encoded successfully")
            except Exception as e:
                print(f"[DEBUG] Image encoding failed: {str(e)}")
                yield f"错误：图片编码失败 - {str(e)}"
                return
            
            # Prepare messages
            messages = [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": self.system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        },
                        {"type": "text", "text": "请详细描述这张图片的内容。"}
                    ]
                }
            ]
            
            print("[DEBUG] Creating completion")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            
            # Process response
            print("[DEBUG] Processing completion chunks")
            has_content = False
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    if content:
                        has_content = True
                        print(f"[DEBUG] Got content chunk: {content}")
                        yield content
            
            if not has_content:
                print("[DEBUG] No content generated")
                yield "抱歉，我无法解析这张图片。"
                    
        except Exception as e:
            print(f"[DEBUG] Error in vision chat: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            yield f"图片处理出错: {str(e)}"