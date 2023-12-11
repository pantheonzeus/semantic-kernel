import semantic_kernel as sk
import asyncio

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig

import customer_feedback

pluginsDirectory = "./plugins-sk"
useAzureOpenAI = False
customer_feedback = customer_feedback.customer_feedback

# Connect to OpenAPI or Azure OpenAPI. Then load the planner and import the skills
kernel = sk.Kernel()

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))

planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))
kernel.import_semantic_skill_from_directory(pluginsDirectory, "DesignThinking")

print("A kernel is now ready.")

# Run Design Thinking Process around the provided customer feedback

ask = 'This is customer feedback. Emphasize it, Define it. Ideate.'
plan = asyncio.run(planner.create_plan_async(goal=ask))
result = asyncio.run(plan.invoke_async(input=customer_feedback))

for index, step in enumerate(plan._steps):
    print(f"✅ Step {index+1} used function `{step._function.name}`")

print("## ✨ Generated result from the ask: " + str(ask) +  "\n\n---\n"  + str(result))