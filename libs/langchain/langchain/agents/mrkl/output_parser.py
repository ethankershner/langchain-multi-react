import re
from typing import Union, List

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.agent import AgentOutputParser, MultiActionAgentOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)

ACTIONS_INPUT_MISMATCH_ERROR_MESSAGE = (
    "Invalid Format: Number of 'Actions' not equal to number of 'Action Inputs':"
)

FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)

class MRKLOutputParser(AgentOutputParser):
    """MRKL Output parser for the chat agent."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match and includes_answer:
            if text.find(FINAL_ANSWER_ACTION) < text.find(action_match.group(0)):
                # if final answer is before the hallucination, return final answer
                start_index = text.find(FINAL_ANSWER_ACTION) + len(FINAL_ANSWER_ACTION)
                end_index = text.find("\n\n", start_index)
                return AgentFinish(
                    {"output": text[start_index:end_index].strip()}, text[:end_index]
                )
            else:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )

        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl"
    
class MRKLMultiActionOutputParser(MultiActionAgentOutputParser):
    """MRKL mutli-action output parser."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = r"Action[s]*:\s(.*?)(?:\s+Action Inputs:|\Z)\s*(.*?)$"
        action_match = re.search(regex, text, re.MULTILINE)

        if action_match and includes_answer:
            if text.find(FINAL_ANSWER_ACTION) < text.find(action_match.group(0)):
                # if final answer is before the hallucination, return final answer
                start_index = text.find(FINAL_ANSWER_ACTION) + len(FINAL_ANSWER_ACTION)
                end_index = text.find("\n\n", start_index)
                return AgentFinish(
                    {"output": text[start_index:end_index].strip()}, text[:end_index]
                )
            else:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )

        if action_match:

            tool_names_group = action_match.group(1)
            tool_inputs_group = action_match.group(2)
            
            tool_names = [name.strip() for name in re.split(r",\s*", tool_names_group)]
            tool_inputs = [input_.strip() for input_ in re.split(r",\s*", tool_inputs_group)]

            if len(tool_names)==len(tool_inputs):
                return [AgentAction(tool=tool_names[i], tool_input=tool_inputs[i].strip(" ").strip('"'), log=text) for i in range(len(tool_names))]

            else:
                raise OutputParserException(f"{ACTIONS_INPUT_MISMATCH_ERROR_MESSAGE}: {text}")

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        
        if not re.search(r"Actions\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Inputs\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl_multi"