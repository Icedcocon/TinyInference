from typing import Dict, List, Optional, Set

from pydantic import BaseModel

BUILTIN_LLM_PROMPT_STYLE: Dict[str, "PromptStyleV1"] = {}
BUILTIN_LLM_MODEL_CHAT_FAMILIES: Set[str] = set()
BUILTIN_LLM_MODEL_GENERATE_FAMILIES: Set[str] = set()
BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES: Set[str] = set()


class PromptStyleV1(BaseModel):
    style_name: str
    system_prompt: str = ""
    roles: List[str]
    intra_message_sep: str = ""
    inter_message_sep: str = ""
    stop: Optional[List[str]]
    stop_token_ids: Optional[List[int]]