import os 
from anthropic import Anthropic
import pandas as pd 
import privacy_csv 
from privacy_csv import df 


os.environ["ANTHROPIC_API_KEY"] = "YOUR_KEY_HERE"
client = Anthropic() 


#import series 
text_series = df['Text']

def summarize_text(text): 
    response = client.messages.create(
        model = "claude-3-5-sonnet-latest", 
        max_tokens= 110, 
        temperature = 0.5,
        messages = [{"role": "user", "content": f"Please summarize the main points of this policy in 80 words or fewer. If the policy is too short to summarize meaningfully, please return Incomplete Policy." + text}]
    )
    return response.content[0] 

sliced_series = text_series[1:10] #can increase to full list 

print(sliced_series) 
summarized_policies_sample = [summarize_text(text) for text in sliced_series]

for i, summary in enumerate(summarized_policies_sample):
   print(f"Summary {i + 1}:\n{summary}\n")

