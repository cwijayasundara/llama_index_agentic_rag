from helper import get_openai_api_key
from utils_l4 import get_doc_tools
from pathlib import Path
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

OPENAI_API_KEY = get_openai_api_key()

nest_asyncio.apply()

# 1. Set up an agent over 3 papers
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "docs/metagpt.pdf",
    "docs/longlora.pdf",
    "docs/selfrag.pdf",
]

paper_to_tools_dict = {}

for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

llm = OpenAI(model="gpt-3.5-turbo")
print(len(initial_tools))

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools,
    llm=llm,
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)
print(str(response))

response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))
