"""Module providing async functionality."""
import asyncio
import semantic_kernel as sk
from dotenv import dotenv_values

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.planning.sequential_planner.sequential_planner_config import (
    SequentialPlannerConfig
)

import data

config = dotenv_values(".env")

PLUGINS_DIRECTORY = "./plugins-sk"
CUSTOMER_FEEDBACK = data.CUSTOMER_FEEDBACK
ASK = data.ASK

# Connect to OpenAPI or Azure OpenAPI. Then load the planner and import the skills
kernel = sk.Kernel()

if config.get("AZURE_OPENAI_API_KEY", None):
    deployment, api_key, endpoint, _ = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service(
        "azureopenai",
        AzureChatCompletion(deployment, endpoint, api_key)
    )
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service(
        "openai",
        OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id)
    )

planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))
kernel.import_semantic_skill_from_directory(PLUGINS_DIRECTORY, "DesignThinking")

print("A kernel is now ready.")

# Run Design Thinking Process around the provided customer feedback

plan = asyncio.run(planner.create_plan_async(goal=ASK))
result = asyncio.run(plan.invoke_async(input=CUSTOMER_FEEDBACK))

for index, step in enumerate(plan._steps):
    print(f"✅ Step {index+1} used function `{step._function.name}`")

print("## ✨ Generated result from the ask: " + str(ASK) +  "\n\n---\n"  + str(result))
