from abc import ABC, abstractmethod
from typing import Any, Dict, List, Generator
import json
import requests
from bs4 import BeautifulSoup

class ImageAnalysisAgent:
    def __init__(self, client):
        self.client = client
        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'analyze_image',
                    'description': 'Analyze the content of an image',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'image_path': {
                                'type': 'string',
                                'description': 'The path to the image file to analyze'
                            }
                        },
                        'required': ['image_path']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'description': '获取杭州天气信息',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'city_code': {
                                'type': 'string',
                                'description': '城市代码，默认为杭州(101210101)'
                            }
                        },
                        'required': ['city_code']
                    }
                }
            }
        ]
        self.function_map = {
            'analyze_image': self.vision_analysis,
            'get_weather': self.weather_analysis
        }
    
    def get_tools(self) -> List[Dict]:
        """获取可用的工具列表"""
        return self.tools
    
    def vision_analysis(self, image_path: str) -> Generator[str, None, None]:
        """使用视觉模型分析图片"""
        try:
            for chunk in self.client.sendPicture(image_path):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    def weather_analysis(self, city_code: str = "101210101") -> Generator[str, None, None]:
        """获取并解析天气信息"""
        try:
            print("[DEBUG] Starting weather analysis for city_code:", city_code)
            # 获取网页内容
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            url = f"https://www.weather.com.cn/weather/{city_code}.shtml"
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'
            
            print("[DEBUG] Got weather page response")
            
            # 将HTML内容发送给语言模型解析
            response_gen = self.client.llm_chat(
                messages=[{
                    'role': 'user',
                    'content': f"""请分析这个天气网页的HTML内容，提取出今天的天气信息，包括温度、天气状况等关键信息。
                    仅返回关键信息，用通俗易懂的语言描述。
                    HTML内容如下：
                    {response.text}"""
                }]
            )
            
            print("[DEBUG] Sent request to language model")
            
            # 处理响应结果
            for chunk in response_gen:
                print(f"[DEBUG] Processing chunk: {chunk}")
                if isinstance(chunk, dict):
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        print(f"[DEBUG] Yielding dict content: {content}")
                        yield content
                elif isinstance(chunk, str):
                    print(f"[DEBUG] Yielding string content: {chunk}")
                    yield chunk
            
        except Exception as e:
            print(f"[DEBUG] Error in weather_analysis: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            yield f"获取天气信息失败: {str(e)}"

    def execute_tool(self, tool_call: Dict) -> Dict[str, Any]:
        """执行工具调用"""
        try:
            print("[DEBUG] Starting tool execution with:", json.dumps(tool_call, ensure_ascii=False))
            function_name = tool_call['function']['name']
            arguments = tool_call['function']['arguments']
            
            print(f"[DEBUG] Function name: {function_name}")
            print(f"[DEBUG] Raw arguments: {arguments}")
            
            # 规范化参数处理
            try:
                # 确保arguments是字典
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                        print(f"[DEBUG] Successfully parsed arguments: {json.dumps(arguments, ensure_ascii=False)}")
                    except json.JSONDecodeError as e:
                        print(f"[DEBUG] Failed to parse arguments: {str(e)}")
                        arguments = {}
                
                # 添加默认参数
                if function_name == "get_weather" and "city_code" not in arguments:
                    arguments["city_code"] = "101210101"
                    print("[DEBUG] Added default city_code")
                
                # 获取对应的函数
                if function := self.function_map.get(function_name):
                    print(f"[DEBUG] Found function {function_name} in function_map")
                    
                    # 执行函数并收集结果
                    result = []
                    print(f"[DEBUG] Starting function execution with args: {json.dumps(arguments, ensure_ascii=False)}")
                    
                    # 直接调用函数
                    generator = function(**arguments)
                    print("[DEBUG] Function called successfully, processing generator")
                    
                    # 处理生成器返回的内容
                    for chunk in generator:
                        if chunk:
                            print(f"[DEBUG] Received chunk: {chunk}")
                            result.append(str(chunk))
                    
                    if not result:
                        print("[DEBUG] No output generated from function")
                        return {
                            "status": "error",
                            "message": "No output generated"
                        }
                    
                    print(f"[DEBUG] Function completed with {len(result)} chunks")
                    print(f"[DEBUG] Results: {result}")
                    
                    return {
                        "status": "success",
                        "data": result,
                        "type": "vision_analysis" if function_name == "analyze_image" else "weather_analysis"
                    }
                
                print(f"[DEBUG] Function {function_name} not found in function_map")
                return {
                    "status": "error",
                    "message": f"Unknown function: {function_name}"
                }
                
            except Exception as e:
                print(f"[DEBUG] Error during function execution: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                return {
                    "status": "error",
                    "message": f"Tool execution error: {str(e)}"
                }
                
        except Exception as e:
            print(f"[DEBUG] Outer error in execute_tool: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"Tool execution error: {str(e)}"
            }
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """处理图片分析请求"""
        try:
            # 使用语言模型进行意图理解
            response = self.client.llm_chat(
                messages=[{
                    'role': 'user',
                    'content': f'请分析这张图片的内容 {image_path}'
                }],
                tools=self.tools
            )
            
            # 处理模型响应
            if 'message' in response and 'tool_calls' in response['message']:
                return self.execute_tool(response['message']['tool_calls'][0])
            
            return {
                "status": "error",
                "message": "No tool calls returned from model"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }