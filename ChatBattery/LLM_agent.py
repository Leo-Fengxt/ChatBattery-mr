import sys
import os
import re
import openai
import time


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def _configure_openai_client():
    """
    Configure OpenAI-compatible client settings.

    - Default: OpenAI Chat Completions endpoint.
    - Optional: OpenRouter (OpenAI-compatible) if you set either:
        - OPENAI_API_BASE / OPENAI_BASE_URL to https://openrouter.ai/api/v1, or
        - OPENROUTER_API_KEY (we will default api_base to OpenRouter).
    """
    api_base = (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_API_BASE")
        or os.getenv("OPENROUTER_BASE_URL")
    )

    if api_base:
        openai.api_base = api_base.rstrip("/")
    elif os.getenv("OPENROUTER_API_KEY"):
        # Convenience default: if user provides an OpenRouter key, use OpenRouter base.
        openai.api_base = OPENROUTER_API_BASE

    # Convenience: allow OPENROUTER_API_KEY without needing to also set OPENAI_API_KEY.
    if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENROUTER_API_KEY")


def _is_openrouter_base(api_base: str) -> bool:
    return "openrouter.ai" in (api_base or "")


def _normalize_model_for_api(model: str) -> str:
    """
    OpenRouter requires fully-qualified model ids like `openai/gpt-4o-mini`.
    For convenience, if the base url is OpenRouter and the model has no provider
    prefix, we add one.
    """
    if _is_openrouter_base(getattr(openai, "api_base", "")) and "/" not in model:
        prefix = os.getenv("CHATBATTERY_OPENROUTER_MODEL_PREFIX", "openai/").strip()
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return f"{prefix}{model}"
    return model


def _clean_formula_candidate(text: str) -> str:
    """
    Convert common LLM formatting (markdown/LaTeX) into a plain-text formula.

    Examples:
      "$Li_{1.02}Fe_{0.70}Mn_{0.25}Mg_{0.05}PO_4$" -> "Li1.02Fe0.70Mn0.25Mg0.05PO4"
      "**LiFePO_4**" -> "LiFePO4"
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Remove basic markdown emphasis/backticks.
    text = text.replace("**", "").replace("__", "").strip("`").strip()

    # If LaTeX math chunks exist, prefer the longest $...$ span.
    if "$" in text:
        chunks = re.findall(r"\$([^$]+)\$", text)
        if chunks:
            text = max(chunks, key=len)
        else:
            text = text.replace("$", "")

    # Replace common LaTeX dot.
    text = text.replace(r"\cdot", "·")

    # Remove LaTeX commands (best-effort).
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # Convert LaTeX subscripts:
    #   X_{1.02} -> X1.02
    #   O_4 -> O4
    text = re.sub(r"_\{([^}]+)\}", r"\1", text)
    text = re.sub(r"_([0-9]+(?:\.[0-9]+)?)", r"\1", text)

    # Drop braces left behind.
    text = text.replace("{", "").replace("}", "")

    # Remove whitespace.
    text = re.sub(r"\s+", "", text)

    # Keep only characters that can appear in formulas that this repo supports.
    text = re.sub(r"[^A-Za-z0-9\.\(\)\[\]/·]", "", text)

    return text


def parse(raw_text, history_battery_list):
    """
    Extract candidate formulas from an LLM response.

    The LLM may format formulas with markdown/LaTeX (e.g. subscripts). This
    parser attempts to recover plain-text formulas and avoids extracting tokens
    from "Reasoning" bullets.
    """
    record = []

    for line in (raw_text or "").strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("Assistant:"):
            line = line.replace("Assistant:", "", 1).strip()

        # Only consider bullet lines (prompt asks for asterisks).
        if not line.startswith("*"):
            continue

        # Skip reasoning bullets (these often contain fragments like PO_4, Li^+, etc.).
        if re.search(r"\breasoning\b", line, flags=re.IGNORECASE):
            continue

        body = line.lstrip("*").strip()
        if not body:
            continue

        # Prefer math chunks if present; otherwise parse the whole bullet.
        segments = re.findall(r"\$([^$]+)\$", body)
        if not segments:
            segments = [body]

        for seg in segments:
            candidate = _clean_formula_candidate(seg)
            if not candidate:
                continue

            # Avoid extracting pure anion fragments (e.g., PO4) by requiring Li/Na.
            if ("Li" not in candidate) and ("Na" not in candidate):
                continue

            # Must contain at least 2 element tokens.
            elements = re.findall(r"[A-Z][a-z]?", candidate)
            if len(elements) < 2:
                continue

            if candidate in history_battery_list:
                continue

            if candidate not in record:
                record.append(candidate)

    return record


class LLM_Agent:
    @staticmethod
    def optimize_batteries(messages, LLM_type, temperature=0, loaded_model=None, loaded_tokenizer=None):
        print("===== Checking messages in the LLM agent =====")
        for message in messages:
            content = message["content"]
            print("content: {}".format(content))
        print("===== Done checking. =====\n")

        if LLM_type == 'chatgpt_3.5':
            return LLM_Agent.optimize_batteries_chatgpt(messages, model="gpt-3.5-turbo", temperature=temperature)
        elif LLM_type == 'chatgpt_o1':
            return LLM_Agent.optimize_batteries_chatgpt(messages, model="o1-mini", temperature=temperature)
        elif LLM_type == 'chatgpt_o3':
            return LLM_Agent.optimize_batteries_chatgpt(messages, model="o3-mini", temperature=temperature)
        elif LLM_type == 'chatgpt_4o':
            return LLM_Agent.optimize_batteries_chatgpt(messages, model="gpt-4o-mini", temperature=temperature)
        elif LLM_type == "llama2":
            return LLM_Agent.optimize_batteries_open_source(messages, loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
        elif LLM_type == "llama3":
            return LLM_Agent.optimize_batteries_open_source(messages, loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
        elif LLM_type == "qwen":
            return LLM_Agent.optimize_batteries_open_source(messages, loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
        else:
            # Treat any other value as a direct OpenAI-compatible model id
            # (e.g. "google/gemini-3-flash-preview" on OpenRouter).
            return LLM_Agent.optimize_batteries_chatgpt(messages, model=LLM_type, temperature=temperature)

    @staticmethod
    def optimize_batteries_chatgpt(messages, model, temperature):
        _configure_openai_client()
        api_model = _normalize_model_for_api(model)

        received = False

        history_battery_list = []

        while not received:
            try:
                if model == "chatgpt_3.5":
                    response = openai.ChatCompletion.create(
                        model=api_model,
                        messages=messages,
                        temperature=temperature,
                        frequency_penalty=0.2,
                        n=None)
                else:
                    response = openai.ChatCompletion.create(
                        model=api_model,
                        messages=messages,
                        temperature=temperature,
                        n=None)
                raw_generated_text = response["choices"][0]["message"]['content']

                # ignore batteries that have shown up before
                generated_battery_list = parse(raw_generated_text, history_battery_list)

                print("===== Parsing messages in the LLM agent =====")
                print("raw_generated_text", raw_generated_text.replace("\n", "\t"))
                print("generated_battery_list", generated_battery_list)
                print("===== Done parsing. =====\n")

                if len(generated_battery_list) == 0:
                    raw_generated_battery_list = parse(raw_generated_text, [])
                    print("raw_generated_battery_list", raw_generated_battery_list)
                    messages[-1]["content"] += "Please do not generate batteries in this list {}.".format(raw_generated_battery_list)

                assert len(generated_battery_list) > 0, "The generated batteries have been discussed in our previous rounds of discussion. Will retry."

                received = True
                    
            except Exception as e:
                print(e)
                time.sleep(1)
        return raw_generated_text, generated_battery_list

    @staticmethod
    def optimize_batteries_open_source(messages, loaded_model, loaded_tokenizer):
        received = False
        history_battery_list = []

        input_text = ""
        for message in messages:
            content = message["content"]
            input_text = "{}\n{}".format(input_text, content)
        print("input_text", input_text)
    
        inputs = loaded_tokenizer(input_text, return_tensors="pt").to("cuda")

        while not received:
            try:
                # raw_generated_text = response["choices"][0]["message"]['content']
                outputs = loaded_model.generate(**inputs)
                raw_generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)


                # ignore batteries that have shown up before
                generated_battery_list = parse(raw_generated_text, history_battery_list)

                print("===== Parsing messages in the LLM agent =====")
                print("raw_generated_text", raw_generated_text.replace("\n", "\t"))
                print("generated_battery_list", generated_battery_list)
                print("===== Done parsing. =====\n")

                if len(generated_battery_list) == 0:
                    raw_generated_battery_list = parse(raw_generated_text, [])
                    print("raw_generated_battery_list", raw_generated_battery_list)
                    messages[-1]["content"] += "Please do not generate batteries in this list {}.".format(raw_generated_battery_list)

                assert len(generated_battery_list) > 0, "The generated batteries have been discussed in our previous rounds of discussion. Will retry."

                received = True
                    
            except Exception as e:
                print(e)
                time.sleep(1)
        return raw_generated_text, generated_battery_list


    @staticmethod
    def rank_batteries(messages, LLM_type, temperature):
        if LLM_type == 'chatgpt_3.5':
            return LLM_Agent.rank_batteries_chatgpt(messages, model="gpt-3.5-turbo", temperature=temperature)
        elif LLM_type == 'chatgpt_o1':
            return LLM_Agent.rank_batteries_chatgpt(messages, model="o1-mini", temperature=temperature)
        elif LLM_type == 'chatgpt_o3':
            return LLM_Agent.rank_batteries_chatgpt(messages, model="o3-mini", temperature=temperature)
        elif LLM_type == 'chatgpt_4o':
            return LLM_Agent.rank_batteries_chatgpt(messages, model="gpt-4o-mini", temperature=temperature)
        else:
            # Treat any other value as a direct OpenAI-compatible model id
            # (e.g. "google/gemini-3-flash-preview" on OpenRouter).
            return LLM_Agent.rank_batteries_chatgpt(messages, model=LLM_type, temperature=temperature)

    @staticmethod
    def rank_batteries_chatgpt(messages, model, temperature):
        _configure_openai_client()
        api_model = _normalize_model_for_api(model)

        received = False

        if model == "chatgpt_3.5":
            response = openai.ChatCompletion.create(
                model=api_model,
                messages=messages,
                temperature=temperature,
                frequency_penalty=0.2,
                n=None)
        else:
            response = openai.ChatCompletion.create(
                model=api_model,
                messages=messages,
                temperature=temperature,
                n=None)
        raw_generated_text = response["choices"][0]["message"]['content']

        return raw_generated_text
