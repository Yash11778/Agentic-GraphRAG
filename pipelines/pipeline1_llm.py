import time

from pipelines.utils import count_tokens, gemini_generate, make_result, setup_gemini

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = setup_gemini()
    return _client


def pipeline1(question: str) -> dict:
    client = _get_client()
    prompt = (
        "You are an expert assistant with deep, accurate knowledge across all domains. "
        "Answer the question below in full. "
        "Include specific names, dates, places, causes, effects, and any other relevant facts. "
        "Write in clear prose — do not truncate or say 'see also'.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    start   = time.time()
    answer  = gemini_generate(client, prompt, max_tokens=512)
    latency = round(time.time() - start, 3)

    p_tok = count_tokens(prompt)
    c_tok = count_tokens(answer)
    return make_result('llm_only', answer, p_tok, c_tok, latency)
