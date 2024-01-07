from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# repo_id = "databricks/dolly-v2-3b"
repo_id = "google/flan-t5-xxl"
# repo_id = "Writer/camel-5b-hf"  # See https://huggingface.co/Writer for other options
# repo_id = "tiiuae/falcon-40b"
question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

#
# llm_hugging_face= HuggingFaceHub(
#     repo_id=repo_id,
#     model_kwargs={"temperature":0.5,"max_length":64})
# output = llm_hugging_face.predict("can you tell me the capital of Australia")
# print(output)
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))