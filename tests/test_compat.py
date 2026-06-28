from __future__ import annotations

import json

from opencode_proxy.compat import (
    build_tool_call_chunks,
    convert_chat_completion_response,
    has_raw_tool_prefix,
    make_tool_call,
    parse_raw_tool_calls,
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
