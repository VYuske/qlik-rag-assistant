"""Testes do grafo de geração, usando LLM e retriever falsos."""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from qlik_rag.config.prompts import GRADE_PROMPT, REFORMULATE_PROMPT, SYSTEM_PROMPT
from qlik_rag.config.settings import Settings
from qlik_rag.generation.graph import build_rag_graph

SYSTEM_MARKER = SYSTEM_PROMPT.splitlines()[0]
GRADE_MARKER = GRADE_PROMPT.splitlines()[0]
REFORMULATE_MARKER = REFORMULATE_PROMPT.splitlines()[0]


class RoutingFakeChatModel(BaseChatModel):
    """LLM falso que decide a resposta conforme o prompt recebido."""

    grade_answer: str = "SIM"
    answer: str = "resposta final"
    reformulated: str = "pergunta reformulada"

    @property
    def _llm_type(self) -> str:
        return "routing-fake"

    def _prompt_text(self, messages: list[BaseMessage]) -> str:
        return "\n".join(str(message.content) for message in messages)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> ChatResult:
        text = self._prompt_text(messages)
        if SYSTEM_MARKER in text:
            content = self.answer
        elif GRADE_MARKER in text:
            content = self.grade_answer
        else:
            content = self.reformulated
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> Iterator[ChatGenerationChunk]:
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        for word in str(message.content).split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


class RecordingRetriever:
    """Retriever falso que registra as consultas recebidas."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.calls: list[str] = []

    def __call__(self, query: str) -> list[Document]:
        self.calls.append(query)
        return self._documents


def make_documents() -> list[Document]:
    return [
        Document(
            page_content="Bookmarks let you save selection states.",
            metadata={"source": "guide.pdf", "page": 30},
        )
    ]


def build_graph(
    settings: Settings,
    retriever: RecordingRetriever,
    llm: RoutingFakeChatModel,
):
    return build_rag_graph(settings=settings, retrieve=retriever, llm=llm)


def payload(question: str, messages: list[BaseMessage] | None = None) -> dict:
    return {
        "messages": list(messages or []),
        "question": question,
        "search_query": question,
        "documents": [],
        "attempts": 0,
        "sufficient": False,
    }


def test_happy_path_answers_after_single_retrieval(settings: Settings) -> None:
    retriever = RecordingRetriever(make_documents())
    graph = build_graph(settings, retriever, RoutingFakeChatModel(grade_answer="SIM"))

    result = graph.invoke(payload("how to create a bookmark?"))

    assert retriever.calls == ["how to create a bookmark?"]
    assert result["messages"][-1].content == "resposta final"


def test_failed_grade_reformulates_and_retries(settings: Settings) -> None:
    retriever = RecordingRetriever(make_documents())
    graph = build_graph(settings, retriever, RoutingFakeChatModel(grade_answer="NAO"))

    result = graph.invoke(payload("how to create a bookmark?"))

    assert len(retriever.calls) == 2
    assert retriever.calls[1] == "pergunta reformulada"
    assert result["messages"][-1].content == "resposta final"


def test_retrieval_loops_are_bounded_by_settings(settings: Settings) -> None:
    bounded = settings.model_copy(update={"max_retrieval_loops": 0})
    retriever = RecordingRetriever(make_documents())
    graph = build_graph(bounded, retriever, RoutingFakeChatModel(grade_answer="NAO"))

    graph.invoke(payload("anything"))

    assert len(retriever.calls) == 1


def test_grade_failure_degrades_to_generate(settings: Settings) -> None:
    class ExplodingGradeModel(RoutingFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs: object):
            if GRADE_MARKER in self._prompt_text(messages):
                raise RuntimeError("modelo indisponível")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    permissive = settings.model_copy(update={"max_retrieval_loops": 3})
    retriever = RecordingRetriever(make_documents())
    graph = build_graph(permissive, retriever, ExplodingGradeModel())

    result = graph.invoke(payload("anything"))

    assert len(retriever.calls) == 1
    assert result["messages"][-1].content == "resposta final"


def test_retrieve_emits_sources_as_custom_stream(settings: Settings) -> None:
    retriever = RecordingRetriever(make_documents())
    graph = build_graph(settings, retriever, RoutingFakeChatModel(grade_answer="SIM"))

    streamed = list(graph.stream(payload("bookmark"), stream_mode="custom"))

    assert streamed[0]["type"] == "sources"
    assert streamed[0]["documents"][0].metadata["page"] == 30


def test_history_is_forwarded_to_generate(settings: Settings) -> None:
    retriever = RecordingRetriever(make_documents())
    graph = build_graph(settings, retriever, RoutingFakeChatModel(grade_answer="SIM"))
    history = [AIMessage(content="resposta anterior")]

    result = graph.invoke(payload("e agora?", history))

    contents = [message.content for message in result["messages"]]
    assert "resposta anterior" in contents
    assert result["messages"][-1].content == "resposta final"


def test_history_reaches_the_model_prompt(settings: Settings) -> None:
    seen_prompts: list[str] = []

    class RecordingModel(RoutingFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs: object):
            seen_prompts.append(self._prompt_text(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    retriever = RecordingRetriever(make_documents())
    graph = build_graph(settings, retriever, RecordingModel(grade_answer="SIM"))

    graph.invoke(payload("e agora?", [AIMessage(content="resposta anterior")]))

    assert any("resposta anterior" in prompt for prompt in seen_prompts)


def test_original_question_is_preserved_when_reformulating(settings: Settings) -> None:
    retriever = RecordingRetriever(make_documents())
    graph = build_graph(settings, retriever, RoutingFakeChatModel(grade_answer="NAO"))

    result = graph.invoke(payload("how to create a bookmark?"))

    assert result["question"] == "how to create a bookmark?"


def test_generate_answers_the_original_question(settings: Settings) -> None:
    seen_prompts: list[str] = []

    class RecordingModel(RoutingFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs: object):
            text = self._prompt_text(messages)
            if SYSTEM_MARKER in text:
                seen_prompts.append(text)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    retriever = RecordingRetriever(make_documents())
    graph = build_graph(settings, retriever, RecordingModel(grade_answer="NAO"))

    graph.invoke(payload("how to create a bookmark?"))

    assert seen_prompts
    assert "how to create a bookmark?" in seen_prompts[0]


def test_retry_does_not_repeat_the_same_query_when_reformulation_is_empty(
    settings: Settings,
) -> None:
    class EmptyReformulationModel(RoutingFakeChatModel):
        reformulated: str = ""

    retriever = RecordingRetriever(make_documents())
    graph = build_graph(settings, retriever, EmptyReformulationModel(grade_answer="NAO"))

    graph.invoke(payload("anything"))

    assert len(retriever.calls) == 1


def test_retry_stops_when_reformulation_repeats_the_question(settings: Settings) -> None:
    class IdentityReformulationModel(RoutingFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs: object):
            text = self._prompt_text(messages)
            if REFORMULATE_MARKER in text:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="anything"))]
                )
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    permissive = settings.model_copy(update={"max_retrieval_loops": 3})
    retriever = RecordingRetriever(make_documents())
    graph = build_graph(permissive, retriever, IdentityReformulationModel(grade_answer="NAO"))

    graph.invoke(payload("anything"))

    assert len(retriever.calls) == 1
