from jinja2 import Environment, FileSystemLoader, StrictUndefined


def build_messages(
        row,
        system_prompt_name,
        user_prompt_name,
        prompt_folder: str,
        assistant_response_col_name: str | None = None,
):
    prompt = []
    if 'evidence' not in row:
        row['evidence'] = ''

    if system_prompt_name:
        system_prompt = _render_prompt(system_prompt_name, prompt_folder, **row)
        prompt.append(
            {"role": "system", "content": system_prompt},
        )

    user_prompt = _render_prompt(user_prompt_name, prompt_folder, **row)

    prompt.append(
        {"role": "user", "content": user_prompt},
    )

    if assistant_response_col_name:
        prompt.append(
            {"role": "assistant", "content": row[assistant_response_col_name]}
        )

    return {"prompt": prompt}


def _render_prompt(template_name: str, prompt_folder: str, **data) -> str:
    env = Environment(
        loader=FileSystemLoader(prompt_folder),
        autoescape=False,  # we’re not rendering HTML
        undefined=StrictUndefined,  # fail fast on missing vars
        trim_blocks=True,  # nicer whitespace
        lstrip_blocks=True,  # ^
        keep_trailing_newline=True,
    )

    tmpl = env.get_template(template_name)
    return tmpl.render(**data).rstrip() + "\n"


if __name__ == "__main__":
    a = build_messages(
        {
            "question": "What is the capital of France?",
            "evidence": "France is a country in Europe. Its capital is Paris.",
            "schema": "Question: <question>\nAnswer: <answer>",
        },
        "base_think_system_prompt.jinja",
        "base_think_user_prompt.jinja",
        "prompts",
    )

    print(a)
