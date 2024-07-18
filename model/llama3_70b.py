from typing import List, Set
import json
import pandas as pd
import re
import random
import sys
from pydantic import BaseModel, conlist, conset
from groq import Groq

client = Groq(
    api_key="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",
)


def synthesize_sentiment(label, text) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a sophisticated Indonesian sentiment analyzer specializing in smart waste management."
                f" Your task is to convert general sentiments into nuanced sentiments tailored for the smart waste management domain."
                f" You will convert this general sentiment: {text}."
                f" This includes customer feedback for the website, smart garbage products, and waste management services."
                f" You need to create distinct sentiments for three categories: positive, negative, and neutral."
                f" Ensure your generated sentiments are relevant to the context of a smart waste management system.\n"
                # Pass the json schema to the model. Pretty printing improves results.
                f" The output should be only the synthesized sentiment text without any additional explanations, translations or labels."
            },
            {
                "role": "user",
                "content": f"Synthesize an Indonesian waste sentiment for {label}."
                f" Make sure the synthesized sentiments are unique and do not repeat those from the existing dataset or among the newly generated sentiments.",
            },
        ],
        model="llama3-70b-8192",
        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic
        # and repetitive.
        temperature=1,
        # Streaming is not supported in JSON mode
        stream=False,
    )
    return chat_completion.choices[0].message.content


class SynthesizeSentiment():
    # 1. def __init__()
    # 2. def setup()

    def __init__(self, label, text):
        super(SynthesizeSentiment, self).__init__()
        self.label = label
        self.text = text

    def setup(self):
        return synthesize_sentiment(self.label, self.text)


# sentiment = synthesize_sentiment("adult")
# print(sentiment)