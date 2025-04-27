from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
)
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall
)
from datasets import Dataset
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure OPENAI_API_KEY is available for RAGAS
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")

# Declare the model for the web search agent and the orchestration agent
# When the model is running on ramalama local server, you can get the model_id with the command:
# curl http://localhost:8080/v1/models | jq .
model = LiteLLMModel(
    model_id="openai/qwen2.5:7b", # replace with your model id - prefer the use of code models for this demo
    api_base="http://localhost:8080/v1", # ramalama exposes an OpenAI API compatible endpoint
    api_key="YOUR_API_KEY", # replace with API key if necessary
    num_ctx=8096, # ollama default is 2048 which will fail horribly. 8096 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

# Initialize the search tool
web_search_tool = DuckDuckGoSearchTool()

agent = CodeAgent(
    tools=[web_search_tool],
    model=model,
    planning_interval=1,
    max_steps=3,
    verbosity_level=2
)

def evaluate_rag_output(question, answer, contexts=None):
    """
    Evaluate the RAG output using RAGAS metrics.
    
    Args:
        question (str): The input question
        answer (str): The generated answer
        contexts (list, optional): List of context strings used to generate the answer
    
    Returns:
        dict: Evaluation results
    """
    # If contexts are not provided, we can only evaluate answer_relevancy
    if contexts is None:
        # Create a dataset with just the question and answer
        data = {
            "question": [question],
            "answer": [answer],
        }
        dataset = Dataset.from_pandas(pd.DataFrame(data))
        
        # Evaluate with just answer_relevancy
        result = evaluate(dataset, [answer_relevancy])
    else:
        # Create a dataset with question, answer, and contexts
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],  # contexts should be a list of strings
        }
        dataset = Dataset.from_pandas(pd.DataFrame(data))
        
        # Evaluate with multiple metrics
        result = evaluate(
            dataset, 
            [faithfulness, answer_relevancy, context_recall]
        )
    
    return result

# Define your question
question = "Provide a historical review of NVIDIA revenue from 2020 until now."

# Run the agent
agent.visualize()
agent_output = agent.run(question)

print("Final output:")
print(agent_output)

# Extract contexts from agent's thought process if available
# This depends on how your agent stores intermediate contexts
# For example, if your agent has a method to access search results
try:
    # This is a placeholder - you'll need to adapt this to how your agent actually stores contexts
    contexts = [step.output for step in agent.steps if hasattr(step, 'output') and isinstance(step.output, str)]
except:
    contexts = None

# Evaluate the output
evaluation_results = evaluate_rag_output(question, agent_output, contexts)
print("\nEvaluation Results:")
print(evaluation_results)