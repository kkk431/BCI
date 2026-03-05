from openai import OpenAI

class SimpleAIChat:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.model_name = "qwen2.5:7b"
        self.system_prompt = {
            "role": "system",
            "content": (
                "你是智融脑机平台的科研助手，专门解答脑机接口（BCI）相关问题。"
                "你拥有神经科学、信号处理、机器学习等领域的专业知识。"
                "请用专业、简洁、结构化的方式回答，避免冗长。"
                "如果问题涉及具体技术细节，请提供清晰的解释；如果不确定，请如实说明。"
                "回答中可适当使用emoji增强可读性，但保持学术严谨性。"
            )
        }
        self.history = [self.system_prompt]

    def chat(self, text):
        self.history.append({"role": "user", "content": text})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"连接失败，请确保 Ollama 已启动且安装了 {self.model_name}\n错误: {str(e)}"

    def clear(self):
        self.history = [self.system_prompt]
