from src.evidence_bureau.slm import retrieve_context, ask


def test_retrieve_context_returns_formatted_chunks():
    context = retrieve_context("What personality types were assigned to agents?")
    assert context
    assert "[" in context and "]" in context  # chunk_id tags present


def test_ask_returns_non_empty_answer():
    answer = ask("What game is the simulation inspired by?", stream=False)
    assert answer
    assert isinstance(answer, str)


if __name__ == "__main__":
    test_retrieve_context_returns_formatted_chunks()
    test_ask_returns_non_empty_answer()
    print("All tests passed.")