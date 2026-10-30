"""Base Agent class for typed agent execution."""

from typing import Literal

from pydantic import BaseModel

from jobagent.llm.client import LLMClient


class Agent[InputT: BaseModel, OutputT: BaseModel]:
    """Generic base class for typed agents."""

    name: str
    system_prompt: str
    tier: Literal["small", "strong"] = "small"
    input_type: type[InputT]
    output_type: type[OutputT]

    def __init__(
        self,
        client: LLMClient,
        name: str | None = None,
        system_prompt: str | None = None,
        tier: Literal["small", "strong"] | None = None,
    ) -> None:
        self.client = client
        if name is not None:
            self.name = name
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if tier is not None:
            self.tier = tier

    def build_prompt(self, input_data: InputT) -> str:
        """Construct the prompt sent to the model from input data.

        Subclasses can override this to customize prompt construction.
        """
        return input_data.model_dump_json(indent=2)

    def run(self, input_data: InputT) -> OutputT:
        """Run the agent and return the validated output model."""
        prompt = self.build_prompt(input_data)
        return self.client.complete_structured(
            system=self.system_prompt,
            prompt=prompt,
            response_model=self.output_type,
            tier=self.tier,
        )
