from src.evidence_bureau.slm import retrieve_context, generate_answer


def test_retrieve_context_returns_formatted_chunks():
    result = retrieve_context("What personality types were assigned to agents?")
    context = result[0]
    assert context
    assert "[" in context and "]" in context  # chunk_id tags present


def test_ask_returns_non_empty_answer():
    answer = generate_answer("What game is the simulation inspired by?")
    assert answer
    assert isinstance(answer["answer"], str)