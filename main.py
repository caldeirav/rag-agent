from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    ToolCallingAgent,
    LiteLLMModel,
)

# Declare the model for the web search agent
model = LiteLLMModel(
    model_id="text-completion-openai/granite-3.1-8b-instruct",
    api_base="http://127.0.0.1:1234/v1",
    api_key="YOUR_API_KEY", # replace with API key if necessary
    num_ctx=8096, # ollama default is 2048 which will fail horribly. 8096 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

search_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can perform web search.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=5,
    verbosity_level=2,
    managed_agents=[search_agent],
)
manager_agent.run("If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?")