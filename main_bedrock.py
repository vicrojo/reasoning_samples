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

def get_weather(location:str) -> str:
    bedrock_runtime = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None)
    )

    prompt_data = """
    Human: Answer the following questions as best you can. You have access to the following tools:
    Search: A search engine. Useful for when you need to answer questions about current events. Input should be a search query.

    Use the following instructions:
    1. Question: the input question you must answer
    2. Thought: you should always think about what to do
    3. Action: the action to take, should be one of [Search]
    4. Action Input: the input to the action
    5. Observation: the result of the action
    6. Thought: I now know the final answer
    7. Final Answer: the final answer to the original input question
    The steps 1 to 5 can repeat N times

    Human: After you answer my following question, I want you to try and verify the answer.
    The verification process is as follows:
    Step 1: Examine the answer and identify elements that might be important to verify, such as notable facts, figures, and any other significant considerations. 
    Step 2: Come up with verification questions that are specific to those identified elements. 
    Step 3: Separately answer each of the verification questions, one at a time. 
    Step 4: Finally, after having answered the verification questions, review the initial answer that you gave to my question and adjust the initial answer based on the results of the verification questions. 
    Other aspects: Make sure to show me the verification questions that you come up with, and their answers, and whatever adjustments to the initial answer you are going to make. It is okay for you to make the adjustments and you do not need to wait for my approval to do so. Do you understand all of these instructions?

    Return only your answer to the human question as a JSON object in form of key:value pairs. The final answer must be named with the key "FinalAnswer", and break it down into the different names and values that compose it, in the form of key-value pairs too.

    Human: what is the weather where in las vegas?
    Assistant:
    """

    body = json.dumps({"prompt": prompt_data, "max_tokens_to_sample": 500, "temperature": 0.2, "top_k": 250, "top_p": 0.1, "stop_sequences": ["\n\nHuman:"]})
    modelId = "anthropic.claude-instant-v1"  # change this to use a different version from the model provider
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    print_ww(response_body.get("completion"))