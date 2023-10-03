import json
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

# ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----
# os.environ["AWS_DEFAULT_REGION"] = "<REGION_NAME>"  # E.g. "us-east-1"
# os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
# os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."

def get_weather(location="Seattle,WA"):
    bedrock_runtime = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None)
    )

    if len(location) < 5:
        location = "Seattle, WA"

    prompt_data = f"""Human: Answer the following questions as best you can. You have access to the following tools:    

    Use the following instructions:
    1. Question: the input question you must answer
    2. Thought: you should always think about what to do
    3. Action: the action to take, should be use a search query
    4. Action Input: the input to the action
    5. Observation: the result of the action
    6. Thought: I now know the final answer
    7. Final Answer: the final answer to the original input question
    The steps 1 to 5 can repeat N times

    Return only your answer to the human question as a JSON object in form of key:value pairs. The final answer must be named with the key "FinalAnswer", and break it down into the different names and values that compose it, in the form of key-value pairs using the following keys: Location(string), Temperature (numeric), Units(string), Conditions(string).

    What is the weather in ${location} right now?
    Use the results you got from the search engine to answer the question, and don't invent a number.

    Assistant:
    """

    body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 500, "temperature": 0.2, "top_k": 250, "top_p": 0.1, "stop_sequences": ["\n\nHuman:"]})
    modelId = "anthropic.claude-v2"  # change this to use a different version from the model provider
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    return response_body.get("completion")