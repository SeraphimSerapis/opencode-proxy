from __future__ import annotations

import json

from opencode_proxy.compat import (
    build_tool_call_chunks,
    convert_chat_completion_response,
    find_raw_tool_start,
    has_complete_raw_tool_block,
    has_raw_tool_prefix,
    make_tool_call,
    normalize_raw_tool_markup,
    parse_raw_tool_calls,
    strip_empty_tool_calls,
)

BAR = "\uff5c"


def test_parse_deepseek_dsml_name_parameters() -> None:
    content = f"""
    <{BAR}DSML{BAR}tool_calls>
    <name>bash</name>
    <parameters>{{&quot;cmd&quot;:&quot;ls -la&quot;}}</parameters>
    </{BAR}DSML{BAR}tool_calls>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"] == {"name": "bash", "arguments": '{"cmd":"ls -la"}'}


def test_parse_deepseek_ascii_dsml_invoke_parameters() -> None:
    content = """
    <|DSML|tool_calls>
    <|DSML|invoke name="edit">
      <|DSML|parameter name="path">README.md</|DSML|parameter>
      <|DSML|parameter name="content">hello</|DSML|parameter>
    </|DSML|invoke>
    </|DSML|tool_calls>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "edit"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "path": "README.md",
        "content": "hello",
    }


def test_parse_bare_tool_calls_invoke_parameters() -> None:
    content = """
    <tool_calls>
    <invoke name="search"><parameter name="query">OpenCode</parameter></invoke>
    </tool_calls>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "OpenCode"}


def test_parse_qwen_tool_call_name_parameters() -> None:
    content = """
    <tool_call>
      <name>read_file</name>
      <parameters>{"path": "src/main.py"}</parameters>
    </tool_call>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"] == {
        "name": "read_file",
        "arguments": '{"path":"src/main.py"}',
    }


def test_parse_qwen_function_parameter_format() -> None:
    content = """
    <tool_call>
      <function=glob><parameter=pattern>*.py</parameter></function>
    </tool_call>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "glob"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"pattern": "*.py"}


def test_convert_non_streaming_response_replaces_content_with_tool_calls() -> None:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<tool_call><name>ls</name><parameters>{}</parameters></tool_call>",
                },
                "finish_reason": "stop",
            },
        ],
    }

    converted, changed = convert_chat_completion_response(body)

    assert changed is True
    choice = converted["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {"name": "ls", "arguments": "{}"}


def test_existing_tool_calls_pass_through() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [make_tool_call("ls", {})],
                },
                "finish_reason": "tool_calls",
            },
        ],
    }

    converted, changed = convert_chat_completion_response(body)

    assert changed is False
    assert converted == body


def test_raw_tool_prefix_detects_split_marker_tail() -> None:
    assert has_raw_tool_prefix("hello <tool_")


def test_build_tool_call_chunks_streams_arguments() -> None:
    tool_call = make_tool_call("write", {"path": "README.md", "content": "abcdef"})

    chunks = build_tool_call_chunks(
        [tool_call],
        chunk_id="chatcmpl-test",
        model="model-a",
        argument_chunk_size=8,
    )

    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "write"
    streamed_args = "".join(
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
        for chunk in chunks[:-1]
    )
    assert json.loads(streamed_args) == {"path": "README.md", "content": "abcdef"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# --- normalize_raw_tool_markup direct tests ---


def test_normalize_ascii_dsml_to_fullwidth() -> None:
    text = "<|DSML|tool_calls><name>bash</name></|DSML|tool_calls>"
    result = normalize_raw_tool_markup(text)
    assert f"<{BAR}DSML{BAR}tool_calls>" in result
    assert f"</{BAR}DSML{BAR}tool_calls>" in result


def test_normalize_dsml_colon_invoke_format() -> None:
    text = '<DSML: invoke name="edit">body</DSML: invoke>'
    result = normalize_raw_tool_markup(text)
    assert f"<{BAR}DSML{BAR}invoke" in result
    assert f"</{BAR}DSML{BAR}invoke>" in result


def test_normalize_dsml_colon_parameter_format() -> None:
    text = '<DSML: parameter name="path">value</DSML: parameter>'
    result = normalize_raw_tool_markup(text)
    assert f"<{BAR}DSML{BAR}parameter" in result
    assert f"</{BAR}DSML{BAR}parameter>" in result


def test_normalize_bare_tool_calls_tags() -> None:
    text = '<tool_calls><invoke name="ls">body</invoke></tool_calls>'
    result = normalize_raw_tool_markup(text)
    assert f"<{BAR}DSML{BAR}tool_calls>" in result
    assert f"<{BAR}DSML{BAR}invoke" in result


def test_normalize_dsml_tag_with_colon_tool_calls() -> None:
    text = "<DSML>tool_calls><name>bash</name></DSML: tool_calls>"
    result = normalize_raw_tool_markup(text)
    assert f"<{BAR}DSML{BAR}tool_calls>" in result
    assert f"</{BAR}DSML{BAR}tool_calls>" in result


def test_normalize_preserves_unrelated_text() -> None:
    text = "Some normal text without any tool markup."
    result = normalize_raw_tool_markup(text)
    assert result == text


# --- Multi-tool-call tests ---


def test_parse_multiple_tool_calls_in_single_dsml_block() -> None:
    content = f"""
    <{BAR}DSML{BAR}tool_calls>
    <name>read_file</name>
    <parameters>{{"path":"a.py"}}</parameters>
    <name>write_file</name>
    <parameters>{{"path":"b.py","content":"hello"}}</parameters>
    </{BAR}DSML{BAR}tool_calls>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert tool_calls[1]["function"]["name"] == "write_file"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "a.py"}
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {
        "path": "b.py",
        "content": "hello",
    }


def test_parse_multiple_qwen_tool_calls() -> None:
    content = """
    <tool_call>
    <name>ls</name>
    <parameters>{"path":"/"}</parameters>
    </tool_call>
    <tool_call>
    <name>cat</name>
    <parameters>{"path":"/etc/hosts"}</parameters>
    </tool_call>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "ls"
    assert tool_calls[1]["function"]["name"] == "cat"


def test_parse_multiple_dsml_invoke_blocks() -> None:
    content = f"""
    <{BAR}DSML{BAR}tool_calls>
    <{BAR}DSML{BAR}invoke name="read">
      <{BAR}DSML{BAR}parameter name="path">a.py</{BAR}DSML{BAR}parameter>
    </{BAR}DSML{BAR}invoke>
    <{BAR}DSML{BAR}invoke name="write">
      <{BAR}DSML{BAR}parameter name="path">b.py</{BAR}DSML{BAR}parameter>
      <{BAR}DSML{BAR}parameter name="content">hi</{BAR}DSML{BAR}parameter>
    </{BAR}DSML{BAR}invoke>
    </{BAR}DSML{BAR}tool_calls>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "read"
    assert tool_calls[1]["function"]["name"] == "write"
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {
        "path": "b.py",
        "content": "hi",
    }


# --- Deduplication tests ---


def test_deduplicate_identical_tool_calls() -> None:
    """When both DSML and Qwen parsers extract the same call, it should be deduped."""
    content = f"""
    <{BAR}DSML{BAR}tool_calls>
    <name>bash</name>
    <parameters>{{"cmd":"ls"}}</parameters>
    </{BAR}DSML{BAR}tool_calls>
    <tool_call>
    <name>bash</name>
    <parameters>{{"cmd":"ls"}}</parameters>
    </tool_call>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "bash"


# --- find_raw_tool_start tests ---


def test_find_raw_tool_start_returns_position_of_dsml() -> None:
    text = f"Some text before <{BAR}DSML{BAR}tool_calls><name>x</name>"
    result = find_raw_tool_start(text)
    assert result == text.index(f"<{BAR}DSML{BAR}tool_calls>")


def test_find_raw_tool_start_returns_position_of_tool_call() -> None:
    text = "Some text <tool_call>body</tool_call>"
    result = find_raw_tool_start(text)
    assert result == text.index("<tool_call>")


def test_find_raw_tool_start_returns_none_for_plain_text() -> None:
    assert find_raw_tool_start("just plain text") is None


# --- has_complete_raw_tool_block tests ---


def test_has_complete_raw_tool_block_dsml() -> None:
    text = (
        f"<{BAR}DSML{BAR}tool_calls><name>x</name>"
        f"<parameters>{{}}</parameters></{BAR}DSML{BAR}tool_calls>"
    )
    assert has_complete_raw_tool_block(text) is True


def test_has_complete_raw_tool_block_qwen() -> None:
    assert has_complete_raw_tool_block("<tool_call><name>x</name></tool_call>") is True


def test_has_complete_raw_tool_block_incomplete() -> None:
    assert has_complete_raw_tool_block(f"<{BAR}DSML{BAR}tool_calls><name>x</name>") is False


def test_has_complete_raw_tool_block_plain_text() -> None:
    assert has_complete_raw_tool_block("no tool blocks here") is False


# --- has_raw_tool_prefix edge cases ---


def test_raw_tool_prefix_empty_string() -> None:
    assert has_raw_tool_prefix("") is False


def test_raw_tool_prefix_full_marker() -> None:
    assert has_raw_tool_prefix("text <tool_call>") is True


def test_raw_tool_prefix_dsml_split() -> None:
    assert has_raw_tool_prefix(f"text <{BAR}DSM") is True


# --- strip_empty_tool_calls tests ---


def test_strip_empty_tool_calls_removes_empty_list() -> None:
    delta = {"content": "hello", "tool_calls": []}
    result = strip_empty_tool_calls(delta)
    assert "tool_calls" not in result
    assert result["content"] == "hello"


def test_strip_empty_tool_calls_preserves_non_empty() -> None:
    delta = {"content": None, "tool_calls": [{"index": 0}]}
    result = strip_empty_tool_calls(delta)
    assert result is delta


def test_strip_empty_tool_calls_no_key() -> None:
    delta = {"content": "just text"}
    result = strip_empty_tool_calls(delta)
    assert result is delta


# --- convert_chat_completion_response edge cases ---


def test_convert_response_no_choices_unchanged() -> None:
    body: dict[str, object] = {"id": "test", "model": "m"}
    result, changed = convert_chat_completion_response(body)
    assert changed is False
    assert result is body


def test_convert_response_empty_choices_unchanged() -> None:
    body: dict[str, object] = {"choices": []}
    _, changed = convert_chat_completion_response(body)
    assert changed is False


def test_convert_response_no_content_unchanged() -> None:
    body: dict[str, object] = {
        "choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]
    }
    _, changed = convert_chat_completion_response(body)
    assert changed is False


def test_convert_response_plain_text_unchanged() -> None:
    body: dict[str, object] = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello, how can I help?"},
                "finish_reason": "stop",
            }
        ]
    }
    _, changed = convert_chat_completion_response(body)
    assert changed is False


# --- make_tool_call edge cases ---


def test_make_tool_call_with_none_arguments() -> None:
    tc = make_tool_call("test", None)
    assert tc["function"]["arguments"] == "{}"


def test_make_tool_call_with_empty_string_arguments() -> None:
    tc = make_tool_call("test", "")
    assert tc["function"]["arguments"] == "{}"


def test_make_tool_call_with_html_escaped_name() -> None:
    tc = make_tool_call("read&amp;write", {})
    assert tc["function"]["name"] == "read&write"


def test_make_tool_call_custom_id() -> None:
    tc = make_tool_call("test", {}, call_id="custom-123")
    assert tc["id"] == "custom-123"
