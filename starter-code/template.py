"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    api_key = os.getenv("OPENAI_API_KEY") or "mock-key"
    start_time = time.time()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        latency = time.time() - start_time

        response_text = response.choices[0].message.content or ""

        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        return response_text, latency, usage

    except Exception as e:
        latency = time.time() - start_time

        return (
            f"[OpenAI error] {str(e)}",
            latency,
            {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )
# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    api_keys = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    start_time = time.time()
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_keys)
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        latency = time.time() - start_time

        response_text = response.text or ""

        usage = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
        }

        return response_text, latency, usage
    except Exception as e:
        latency = time.time() - start_time

        return (
            f"[Gemini error] {str(e)}",
            latency,
            {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    raise NotImplementedError("Implement call_gemini")


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"
    start_time = time.time()

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        latency = time.time() - start_time

        response_text = response.content[0].text if response.content else ""

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return response_text, latency, usage

    except Exception as e:
        latency = time.time() - start_time

        return (
            f"[Anthropic error] {str(e)}",
            latency,
            {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    gpt4o_response, gpt4o_latency, gpt4o_usage = call_openai(
        prompt=prompt,
        model=OPENAI_MODEL,
    )

    gpt4o_mini_response, gpt4o_mini_latency, gpt4o_mini_usage = call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
    )

    gemini_response, gemini_latency, gemini_usage = call_gemini(
        prompt=prompt,
        model=GEMINI_MODEL,
    )

    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * 5.00 / 1_000_000
        + gpt4o_usage["output_tokens"] * 20.00 / 1_000_000
    )

    gpt4o_mini_cost = (
        gpt4o_mini_usage["input_tokens"] * 0.15 / 1_000_000
        + gpt4o_mini_usage["output_tokens"] * 0.60 / 1_000_000
    )

    gemini_cost = (
        gemini_usage["input_tokens"] * 0.075 / 1_000_000
        + gemini_usage["output_tokens"] * 0.30 / 1_000_000
    )

    return {
        "gpt4o": {
            "response": gpt4o_response,
            "latency": gpt4o_latency,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": gpt4o_mini_response,
            "latency": gpt4o_mini_latency,
            "cost": gpt4o_mini_cost,
            "input_tokens": gpt4o_mini_usage["input_tokens"],
            "output_tokens": gpt4o_mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_response,
            "latency": gemini_latency,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set. Running in dummy mode.\033[0m")
        api_key = "mock-key"

    print("\n\033[94m================================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("================================================\033[0m\n")

    history = []

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=512,
        )

        while True:
            user_input = input("\033[92mYou: \033[0m").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("\033[94mGoodbye!\033[0m")
                break

            if not user_input:
                continue

            if api_key == "mock-key":
                print("\033[96mAssistant: \033[0mThis is a dummy response because GEMINI_API_KEY is not set.\n")
                continue

            context = ""

            for turn in history[-3:]:
                context += f"User: {turn['user']}\n"
                context += f"Assistant: {turn['assistant']}\n"

            full_prompt = context + f"User: {user_input}\nAssistant:"

            print("\033[96mAssistant: \033[0m", end="", flush=True)

            full_response = ""

            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=config,
            )

            for chunk in stream:
                chunk_text = getattr(chunk, "text", None)

                if chunk_text:
                    print(chunk_text, end="", flush=True)
                    full_response += chunk_text

            print("\n")

            history.append({
                "user": user_input,
                "assistant": full_response,
            })

            history = history[-3:]

    except Exception as e:
        print(f"\033[91mChatbot failed to start: {e}\033[0m")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return fn()

        except Exception as e:
            last_exception = e

            if attempt == max_retries:
                break

            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

    raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []

    for prompt in prompts:
        try:
            result = compare_models(prompt)
        except TypeError:
            result = compare_models()

        result["prompt"] = prompt
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    rows = []
    rows.append("| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |")
    rows.append("|---|---|---|---:|---:|---:|")

    model_display_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    for result in results:
        prompt = str(result.get("prompt", ""))

        for model_key in ["gpt4o", "gpt4o_mini", "gemini_flash"]:
            if model_key not in result:
                continue

            model_result = result[model_key]
            model_name = model_display_names.get(model_key, model_key)

            response = str(model_result.get("response", "")).replace("\n", " ")
            if len(response) > 50:
                response = response[:50] + "..."

            latency = model_result.get("latency", 0)
            input_tokens = model_result.get("input_tokens", 0)
            output_tokens = model_result.get("output_tokens", 0)
            cost = model_result.get("cost", 0)

            rows.append(
                f"| {prompt} | {model_name} | {response} | "
                f"{latency:.2f}s | {input_tokens}/{output_tokens} | ${cost:.6f} |"
            )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
