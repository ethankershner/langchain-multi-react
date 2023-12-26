# flake8: noqa
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

FORMAT_INSTRUCTIONS_MULTI = """As much as possible, use tools simultaneously. 
However, don't run a tool simultaneously with another tool required to get the 
needed input for that tool. In this case, run the tool needed to get input first.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Actions: the actions to take, should be one or more of [{tool_names}] separated by a comma.
Action Inputs: the inputs to the actions separated by a comma
Observations: the results of an action
... (this Thought/Actions/Action Inputs/Observations can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""