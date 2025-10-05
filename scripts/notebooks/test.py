import os
from pathlib import Path
from cotnav.models.vlms.openaimodel import OpenAIModel
from dotenv import load_dotenv

load_dotenv()

m = OpenAIModel(model="gpt-5")
print(m.generate_one("Explain SE(3) in two sentences.", max_output_tokens=200))