import logging
import sys

import openai

# These model(s) raises an error saying that the organization must be verified if streaming is attempted.
_NON_STREAMABLE_MODELS = {"o3"}


def _non_streamed_openai_response(
    client: openai.OpenAI, model: str, prompt: str
) -> str:

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        # reasoning={"effort": "high"},  # you can choose "low", "medium", "high"
        background=False,
    )
    return response.output_text


def streamed_openai_response(
    *,
    client: openai.OpenAI,
    model: str,
    max_completion_tokens: int,
    prompt: str,
) -> str:
    """Replacement for client.responses.create.

    Except, it also logs the response to stderr in real-time.
    """
    if model in _NON_STREAMABLE_MODELS:
        logging.info("Using non-streaming query for model: %s", model)
        return _non_streamed_openai_response(client=client, model=model, prompt=prompt)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_completion_tokens=max_completion_tokens,
        )
    except openai.BadRequestError as e:
        if "must be verified to stream" in e.message:
            logging.error("Note to developer: Try adding the model to _NON_STREAMABLE.")
        raise  # Re-raise.
    except openai.APIConnectionError as e:
        # TODO: This should be thrown as an exception, and handled appropriately.
        logging.warning(f"openai.APIConnectionError: {e}")
        return "(LLM server error)"
    tokens: list[str] = []
    logging.info(f"Streaming response to stderr:")
    try:
        for event in stream:
            token = event.choices[0].delta.content
            # Token will be None at the end.
            if token is not None:
                tokens.append(token)
                print(token, end="", flush=True)
    except openai.APIError as e:
        logging.warning(f"openai.APIError: {e}")
        # Rely on parser to catch the error, for example if JSON is expected.
        return "(LLM server error)"
    finally:
        print(file=sys.stderr)  # Newline.
    return "".join(tokens)
