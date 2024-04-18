import asyncio
from typing import Any, Dict, Optional, Sequence, Callable

from ibm_watson_machine_learning.foundation_models.model import Model  # type: ignore
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes  # type: ignore
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.watsonx import WatsonX  # type: ignore


class CustomWatsonX(WatsonX):
    """
    IBM WatsonX LLM. Wrapper around the existing WatonX LLM module to provide following features:
    1. Support for dynamically updated WatsonX models. The supported models are hardcoded in original implementation and are outdated as of 2/19/24.
    2. Implements the Async methods for the WatsonX LLM. While these are not true async methods, the implementation allows it to be used in async context.
    """

    def __init__(
        self,
        credentials: Dict[str, Any],
        model_id: Optional[str] = "ibm/mpt-7b-instruct2",
        validate_model_id: bool = True,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        max_new_tokens: Optional[int] = 512,
        temperature: Optional[float] = 0.1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        if validate_model_id:
            supported_models = [model.value for model in ModelTypes]
            if model_id not in supported_models:
                raise ValueError(
                    f"Model name {model_id} not found in {supported_models}"
                )

        super().__init__(
            credentials=credentials,
            model_id="meta-llama/llama-2-70b-chat",  # The WatsonX package is outdated in terms of supported models. This is a dummy model id to bypass the check.
            project_id=project_id,
            space_id=space_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        # Set the model_id and model_info to the actual model
        self.model_id = model_id
        self._model = Model(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            space_id=space_id,
        )
        self.model_info = self._model.get_details()

    @classmethod
    def class_name(self) -> str:
        """Get Class Name."""
        return "CustomWatsonX_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.model_info["model_limits"]["max_sequence_length"],
            num_output=self.max_new_tokens,
            model_name=self.model_id,
        )

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.complete, prompt, formatted, **kwargs
        )

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat, messages, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.stream_chat, messages, **kwargs)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.stream_complete, prompt, formatted, **kwargs
        )
