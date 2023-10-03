import gradio as gr
import json
from weather.langchain_claude_agent import get_weather

def ask_weather(message, history):
    result = get_weather(message)
    json_result = json.loads(result)
    if "FinalAnswer" in json_result.keys():
        return json.dumps(json_result["FinalAnswer"])
    else:
        return json.dumps(json_result)

gr.ChatInterface(
    ask_weather,    
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Ask me for the Weather in your location of preference", container=True, scale=7),
    title="The Weather Chatbot",
    description="Ask me for a location and I will tell you the weather",
    theme="soft",
    examples=["Las Vegas", "Seattle", "New York City"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()