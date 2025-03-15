from .base_model import BaseModel
from typing import Dict, List, Any, Generator, Union
import json

class LanguageModel(BaseModel):
    """通用语言模型"""
    def __init__(self, client):
        super().__init__(client)
        self.model_name = "qwen-plus"
        self.system_prompt = "You are a helpful AI assistant. Answer in Chinese."
    
    def _process_stream(self, completion) -> Generator[Union[Dict, str], None, None]:
        """处理流式响应"""
        has_content = False
        try:
            for chunk in completion:
                if hasattr(chunk.choices[0], 'delta'):
                    delta = chunk.choices[0].delta
                    
                    # 处理工具调用
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_calls = []
                        for tool_call in delta.tool_calls:
                            function = tool_call.function
                            # 直接访问 arguments 属性
                            arguments = function.arguments if hasattr(function, 'arguments') else ""
                            tool_calls.append({
                                "function": {
                                    "name": function.name if function else "",
                                    "arguments": arguments
                                }
                            })
                        yield {
                            "message": {
                                "tool_calls": tool_calls
                            }
                        }
                        return  # 结束生成器
                    
                    # 处理普通内容
                    elif hasattr(delta, 'content') and delta.content is not None:
                        has_content = True
                        yield delta.content
            
            # 如果没有内容，返回默认消息
            if not has_content:
                yield "抱歉，我现在无法回答这个问题。"
                
        except Exception as e:
            yield f"Stream processing error: {str(e)}"

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict] = None) -> Generator[Union[Dict, str], None, None]:
        """语言模型对话"""
        try:
            formatted_messages = self.format_messages(messages, self.system_prompt)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                tools=tools,
                stream=True
            )
            
            # 用于累积工具调用数据
            current_tool_call = None
            argument_parts = []
            content_buffer = []
            
            for chunk in completion:
                if hasattr(chunk.choices[0], 'delta'):
                    delta = chunk.choices[0].delta
                    print(f"[DEBUG] Processing delta: {delta}")
                    
                    # 检查工具调用
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            # 如果有函数名，开始新的工具调用
                            if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                current_tool_call = {
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": ""
                                    }
                                }
                                print(f"[DEBUG] Started new tool call: {tool_call.function.name}")
                            
                            # 累积参数部分
                            if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                argument_parts.append(tool_call.function.arguments)
                                print(f"[DEBUG] Accumulated argument part: {tool_call.function.arguments}")
                    
                    # 检查普通内容
                    elif hasattr(delta, 'content') and delta.content is not None:
                        content_buffer.append(delta.content)
                        yield delta.content
                        print(f"[DEBUG] Yielded content: {delta.content}")
            
            # 处理累积的内容
            if content_buffer:
                # 如果有普通内容，已经通过yield发送了
                pass
            # 如果有工具调用，作为最后一个chunk发送
            elif current_tool_call:
                # 合并所有参数部分
                full_arguments = "".join(argument_parts)
                print(f"[DEBUG] Combined arguments: {full_arguments}")
                
                # 尝试解析JSON以验证格式
                try:
                    json.loads(full_arguments)
                    current_tool_call["function"]["arguments"] = full_arguments
                    print(f"[DEBUG] Final valid tool call: {current_tool_call}")
                    yield {
                        "message": {
                            "tool_calls": [current_tool_call]
                        }
                    }
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] JSON parse error: {str(e)}")
                    yield {
                        "status": "error",
                        "message": f"Invalid JSON format in arguments: {str(e)}"
                    }
        
        except Exception as e:
            print(f"[DEBUG] Error in chat: {str(e)}")
            yield f"Language model error: {str(e)}"