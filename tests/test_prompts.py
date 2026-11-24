"""Testes dos prompts usados pela camada de geração."""

from __future__ import annotations

from qlik_rag.config.prompts import GRADE_PROMPT, RAG_USER_PROMPT, REFORMULATE_PROMPT, SYSTEM_PROMPT


def test_system_prompt_keeps_the_refusal_sentence() -> None:
    assert "I don't have enough information in the available documentation" in SYSTEM_PROMPT


def test_system_prompt_forbids_inventing_examples() -> None:
    assert "Do not create examples" in SYSTEM_PROMPT


def test_rag_user_prompt_exposes_context_and_question_placeholders() -> None:
    assert "{context}" in RAG_USER_PROMPT
    assert "{question}" in RAG_USER_PROMPT


def test_grade_prompt_demands_a_binary_answer() -> None:
    assert "SIM" in GRADE_PROMPT
    assert "NAO" in GRADE_PROMPT


def test_reformulate_prompt_asks_for_a_new_question() -> None:
    assert "{question}" in REFORMULATE_PROMPT
