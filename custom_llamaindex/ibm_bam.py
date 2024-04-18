from typing import Any, Callable, Dict, Optional, Sequence, List
import asyncio

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    LLM,
)
from llama_index.core.base.llms.generic_utils import (
    completion_to_chat_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from genai import Client, Credentials  # type: ignore
from genai.schema import TextGenerationParameters  # type: ignore


GENAI_API_ENDPOINT = "https://bam-api.res.ibm.com"


class IbmBamLLM(LLM):
    """IBM Big AI Model (BAM) Laboratory LLM."""

    model_id: str = Field(description="The Model to use.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    temperature: float = Field(description="The temperature to use for sampling.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional Kwargs for the WatsonX model"
    )
    model_info: Dict[str, Any] = Field(
        default_factory=dict, description="Details about the selected model"
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        apikey: Optional[str] = None,
        model_id: Optional[str] = "ibm/mpt-7b-instruct2",
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
        """Initialize params."""
        if apikey is None:
            credentials = Credentials.from_env()
        else:
            credentials = Credentials(api_key=apikey, api_endpoint=GENAI_API_ENDPOINT)
        client = Client(credentials=credentials)
        self._client = client
        if model_id not in [model.id for model in client.model.list().results]:
            raise ValueError(f"Model name {model_id} not supported")
        model_info = client.model.retrieve(model_id).result.model_dump(
            include=["name", "description", "id", "developer", "size", "token_limits"]
        )
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            additional_kwargs=additional_kwargs,
            model_info=model_info,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get Class Name."""
        return "IbmBamLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.model_info["token_limits"][0]["token_limit"],
            num_output=self.max_new_tokens,
            model_name=self.model_id,
        )

    @property
    def sample_model_kwargs(self) -> List[str]:
        """Get a sample of Model kwargs that a user can pass to the model."""
        params = list(vars(TextGenerationParameters))
        params.remove("return_options")
        return params

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {**self._model_kwargs, **kwargs}

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = list(
            self._client.text.generation.create(
                model_id=self.model_id, inputs=[prompt], parameters=all_kwargs
            )
        )
        generated_text = response[0].results[0].generated_text

        return CompletionResponse(text=generated_text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)

        stream_response = self._client.text.generation.create(
            model_id=self.model_id, inputs=[prompt], parameters=all_kwargs
        )

        def gen() -> CompletionResponseGen:
            content = ""
            for stream_delta in stream_response:
                content += stream_delta
                yield CompletionResponse(text=content, delta=stream_delta)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_fn = completion_to_chat_decorator(self.complete)

        return chat_fn(messages, **all_kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_stream_fn = stream_completion_to_chat_decorator(self.stream_complete)

        return chat_stream_fn(messages, **all_kwargs)

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
