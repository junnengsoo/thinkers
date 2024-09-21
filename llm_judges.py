import os
from openai import OpenAI
import weave
from dotenv import load_dotenv


load_dotenv()

weave.init('together-weave')

system_content = "You are a travel agent. Be descriptive and helpful."
user_content = "Tell me about San Francisco"


os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
chat_completion = client.chat.completions.create(
    # extra_headers={
    # "HTTP-Referer": $YOUR_SITE_URL, # Optional, for including your app on openrouter.ai rankings.
    # "X-Title": $YOUR_APP_NAME, # Optional. Shows in rankings on openrouter.ai.
    # },
    model="openai/gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ],
    temperature=0.7,
    max_tokens=1024,
)
response = chat_completion.choices[0].message.content
print("Model response:\n", response)