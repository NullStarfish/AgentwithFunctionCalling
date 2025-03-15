from api import APIClient
from agent import ImageAnalysisAgent
import os
from typing import List, Dict, Any, Generator, Union
import json

def analyze_local_image(image_path: str):
    """分析本地图片文件"""
    # 验证文件是否存在
    if not os.path.exists(image_path):
        return {"status": "error", "message": "File not found"}
    
    # 验证文件是否为图片
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        return {"status": "error", "message": "Invalid image file"}
    
    try:
        # 初始化客户端和代理
        api_client = APIClient()
        image_agent = ImageAnalysisAgent(api_client)
        
        # 处理图片
        result = image_agent.process(f"file://{os.path.abspath(image_path)}")
        return result
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ChatSession:
    """对话会话管理类"""
    def __init__(self):
        self.api_client = APIClient()
        self.image_agent = ImageAnalysisAgent(self.api_client)
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """添加消息到历史记录"""
        self.conversation_history.append({
            'role': role,
            'content': content
        })

    def handle_response(self) -> Generator[str, None, None]:
        """处理模型响应"""
        try:
            print("[DEBUG] Calling llm_chat with tools")
            response = self.api_client.llm_chat(
                messages=self.conversation_history,
                tools=self.image_agent.get_tools()
            )
            
            print("[DEBUG] Got response type:", type(response))
            
            # 处理工具调用响应
            if isinstance(response, dict):
                print("[DEBUG] Processing dict response")
                if 'message' in response and 'tool_calls' in response['message']:
                    print("[DEBUG] Found tool_calls in response")
                    yield from self._process_tool_calls(response['message']['tool_calls'])
                elif 'message' in response and 'content' in response['message']:
                    content = response['message']['content']
                    self.add_message('assistant', content)
                    yield content
            
            # 处理流式响应
            elif isinstance(response, Generator):
                print("[DEBUG] Processing generator response")
                content = []
                tool_call_data = None
                
                for chunk in response:
                    print(f"[DEBUG] Processing chunk: {chunk}")
                    
                    # 如果是字典类型的响应，可能包含工具调用
                    if isinstance(chunk, dict):
                        if 'message' in chunk and 'tool_calls' in chunk['message']:
                            print("[DEBUG] Found tool calls in stream")
                            tool_call_data = chunk['message']['tool_calls']
                            continue
                    
                    # 如果是普通文本内容
                    elif chunk:
                        print(f"[DEBUG] Got text chunk: {chunk}")
                        content.append(chunk)
                        yield chunk
                
                # 处理累积的内容
                if content:
                    final_content = "".join(content)
                    print(f"[DEBUG] Adding content to history: {final_content}")
                    self.add_message('assistant', final_content)
                
                # 如果收集到了工具调用，进行处理
                if tool_call_data:
                    print(f"[DEBUG] Processing collected tool calls: {json.dumps(tool_call_data, ensure_ascii=False)}")
                    yield from self._process_tool_calls(tool_call_data)
                    
        except Exception as e:
            print(f"[DEBUG] Error in handle_response: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            yield f"处理出错: {str(e)}"

    def _process_tool_calls(self, tool_calls: List[Dict]) -> Generator[str, None, None]:
        """处理工具调用"""
        for tool_call in tool_calls:
            yield "正在处理您的请求...\n"
            try:
                tool_result = self.image_agent.execute_tool(tool_call)
                print(f"[DEBUG] Tool result: {json.dumps(tool_result, ensure_ascii=False)}")
                
                if tool_result["status"] == "success":
                    # 获取工具类型和处理结果
                    is_vision = tool_result.get("type") == "vision_analysis"
                    processing_msg = "正在分析图片...\n" if is_vision else "正在获取天气信息...\n"
                    yield processing_msg
                    
                    # 处理结果内容
                    result_content = self._format_tool_result(tool_result["data"])
                    self.add_message('assistant', result_content)
                    
                    result_type = "分析结果" if is_vision else "天气信息"
                    yield f"{result_type}：{result_content}\n"
                    
                    # 获取AI补充说明
                    yield from self._get_ai_explanation()
                else:
                    yield f"错误: {tool_result['message']}\n"
            except Exception as e:
                print(f"[DEBUG] Error executing tool: {str(e)}")
                yield f"工具调用出错: {str(e)}\n"

    def _handle_tool_call(self, tool_call: Dict) -> Generator[str, None, None]:
        """处理工具调用"""
        try:
            tool_result = self.image_agent.execute_tool(tool_call)
            
            if tool_result["status"] == "success":
                # 获取工具类型和处理结果
                is_vision = tool_result.get("type") == "vision_analysis"
                processing_msg = "正在分析图片...\n" if is_vision else "正在获取天气信息...\n"
                yield processing_msg
                
                # 处理结果内容
                result_content = self._format_tool_result(tool_result["data"])
                self.add_message('assistant', result_content)
                
                result_type = "分析结果" if is_vision else "天气信息"
                yield f"{result_type}：{result_content}\n"
                
                # 获取AI补充说明
                yield from self._get_ai_explanation()
            else:
                yield f"错误: {tool_result['message']}"
            
        except Exception as e:
            yield f"工具调用出错: {str(e)}"

    def _format_tool_result(self, data: Union[List[str], Any]) -> str:
        """格式化工具调用结果"""
        if isinstance(data, list):
            return "".join(data)
        return str(data)

    def _get_ai_explanation(self) -> Generator[str, None, None]:
        """获取AI补充说明"""
        yield "AI助手正在思考...\n"
        response = self.api_client.llm_chat(
            messages=self.conversation_history + [{
                'role': 'user',
                'content': '基于这个结果，你有什么补充说明的吗？'
            }]
        )
        
        if isinstance(response, dict) and 'message' in response:
            if 'content' in response['message']:
                explanation = response['message']['content']
                self.add_message('assistant', explanation)
                yield f"AI助手补充：{explanation}"

def run_chat_session():
    """运行对话会话"""
    session = ChatSession()
    print("欢迎使用AI助手，输入 'quit' 退出程序\n")
    
    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()
            if user_input.lower() == 'quit':
                break
            
            session.add_message('user', user_input)
            print("\nAI助手: ", end='', flush=True)
            
            for chunk in session.handle_response():
                print(chunk, end='', flush=True)
            print()
            
        except KeyboardInterrupt:
            print("\n程序已终止")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

if __name__ == "__main__":
    run_chat_session()