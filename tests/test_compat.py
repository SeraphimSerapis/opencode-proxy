from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencode_proxy.compat import (
    RAW_TOOL_BLOCK_PATTERNS,
    RAW_TOOL_START_MARKERS,
    build_tool_call_chunks,
    complete_truncated_json,
    convert_chat_completion_response,
    extract_raw_tool_call_segments,
    find_raw_tool_start,
    has_complete_raw_tool_block,
    has_raw_tool_prefix,
    make_tool_call,
    normalize_raw_tool_markup,
    parse_raw_tool_calls,
    strip_empty_tool_calls,
    tool_calls_within_limits,
)

BAR = "\uff5c"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tool_calls"


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


def test_parse_degraded_dsml_close_tag_and_spaced_equals() -> None:
    """Degraded DSML still parses when the close tag drops the backslashes and
    spaces appear around ``=`` (both realistic tokenizer artefacts)."""
    content = f"""
    <DSML>tool_calls>
    <{BAR}DSML{BAR}invoke name = "bash">
      <{BAR}DSML{BAR}parameter name = "cmd" string = "true">pwd</{BAR}DSML{BAR}parameter>
      <{BAR}DSML{BAR}parameter name = "count" string = "false">5</{BAR}DSML{BAR}parameter>
    </{BAR}DSML{BAR}invoke>
    </DSML>tool_calls>
    """

    tool_calls = parse_raw_tool_calls(content)
    extracted, remainder, changed = extract_raw_tool_call_segments(content)

    assert has_complete_raw_tool_block(content)
    assert changed
    assert not remainder.strip()
    assert len(extracted) == 1
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "bash"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "cmd": "pwd",
        "count": 5,
    }


DEGRADED_OPENERS = (
    "<DSML>tool_calls>",
    "<DSML: tool_calls>",
    "<DSML:tool_calls>",
    "<DSML tool_calls>",
)


@pytest.mark.parametrize("opener", DEGRADED_OPENERS)
def test_degraded_opener_is_held_back_when_split_across_tokens(opener: str) -> None:
    """The streaming guard must buffer a degraded opener that arrives in pieces.

    Without this the first half is flushed as content and the block can never be
    reassembled, so the tool call reaches the client as raw markup.
    """
    for split in range(4, len(opener)):
        assert has_raw_tool_prefix("some text " + opener[:split]), opener[:split]
    assert has_raw_tool_prefix("some text " + opener)


def test_block_grammar_and_streaming_guard_agree() -> None:
    """Keep the block grammar and the streaming guard from drifting apart.

    ``RAW_TOOL_BLOCK_PATTERNS`` decides what parses; ``RAW_TOOL_START_MARKERS``
    decides what streaming holds back while a marker is still arriving. An
    opener in the first but not the second parses fine in a buffered response
    and corrupts in a streamed one, which is the harder bug to spot.
    """
    for opener in DEGRADED_OPENERS:
        assert any(p.fullmatch(opener) for p, _ in RAW_TOOL_BLOCK_PATTERNS), (
            f"{opener!r} is not accepted by the block grammar"
        )
        assert opener in RAW_TOOL_START_MARKERS, (
            f"{opener!r} parses but streaming will not buffer it"
        )

    for marker in RAW_TOOL_START_MARKERS:
        assert any(p.fullmatch(marker) for p, _ in RAW_TOOL_BLOCK_PATTERNS), (
            f"{marker!r} is buffered by streaming but never parses"
        )


@pytest.mark.parametrize(
    ("truncated", "expected"),
    [
        # The exact shapes observed from vLLM closing a turn mid-arguments.
        ('{"pattern": "def main|uvicorn|FastAPI', {"pattern": "def main|uvicorn|FastAPI"}),
        ('{"path": "src/opencode_proxy/proxy.py', {"path": "src/opencode_proxy/proxy.py"}),
        ('{"a": [1, 2', {"a": [1, 2]}),
        ('{"a": {"b": "x', {"a": {"b": "x"}}),
        ('{"a":', {"a": None}),
        ('{"a": "x\\\\', {"a": "x\\"}),
        ('{"a": "he said \\"hi', {"a": 'he said "hi'}),
    ],
)
def test_complete_truncated_json_repairs_by_appending(truncated: str, expected: object) -> None:
    suffix = complete_truncated_json(truncated)
    assert suffix is not None
    assert json.loads(truncated + suffix) == expected


def test_complete_truncated_json_returns_empty_suffix_when_already_valid() -> None:
    assert complete_truncated_json('{"a": 1}') == ""


@pytest.mark.parametrize(
    "unrepairable",
    [
        '{"a": 1, ',  # would need the trailing comma removed, not appended to
        '{"a": tru',  # guessing the literal could invert the caller's intent
        "",
        "   ",
    ],
)
def test_complete_truncated_json_refuses_what_it_cannot_append_to(unrepairable: str) -> None:
    assert complete_truncated_json(unrepairable) is None


def test_malformed_joined_dsml_marker_is_not_normalized() -> None:
    content = f"""
    <DSMLtool_calls>
    <{BAR}DSML{BAR}invoke name="bash"></{BAR}DSML{BAR}invoke>
    </DSMLtool_calls>
    """

    assert find_raw_tool_start(content) is None
    assert not has_complete_raw_tool_block(content)
    assert not parse_raw_tool_calls(content)


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


def test_parse_qwen3_multi_line_function_parameter_format() -> None:
    content = """
    <tool_call>
    <function=get_weather>
    <parameter=location>San Francisco, CA</parameter>
    <parameter=unit>celsius</parameter>
    </function>
    </tool_call>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "location": "San Francisco, CA",
        "unit": "celsius",
    }


def test_parse_qwen3_multiple_functions_in_one_tool_call() -> None:
    content = """
    <tool_call>
    <function=get_weather>
    <parameter=location>San Francisco</parameter>
    <parameter=unit>celsius</parameter>
    </function>
    <function=get_weather>
    <parameter=location>Seattle</parameter>
    <parameter=unit>fahrenheit</parameter>
    </function>
    </tool_call>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "location": "San Francisco",
        "unit": "celsius",
    }
    assert tool_calls[1]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {
        "location": "Seattle",
        "unit": "fahrenheit",
    }


def test_parse_laguna_arg_key_value_single_arg() -> None:
    content = "<tool_call>terminal<arg_key>cmd</arg_key><arg_value>uname -a</arg_value></tool_call>"

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "terminal"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"cmd": "uname -a"}


def test_parse_laguna_arg_key_value_multiple_args() -> None:
    content = (
        "<tool_call>edit_file"
        "<arg_key>path</arg_key><arg_value>src/main.py</arg_value>"
        "<arg_key>content</arg_key><arg_value>print('hello')</arg_value>"
        "</tool_call>"
    )

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "edit_file"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "path": "src/main.py",
        "content": "print('hello')",
    }


def test_parse_laguna_json_arg_values() -> None:
    content = (
        "<tool_call>configure"
        "<arg_key>debug</arg_key><arg_value>true</arg_value>"
        "<arg_key>retries</arg_key><arg_value>5</arg_value>"
        '<arg_key>tags</arg_key><arg_value>["prod", "v2"]</arg_value>'
        "</tool_call>"
    )

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "configure"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "debug": True,
        "retries": 5,
        "tags": ["prod", "v2"],
    }


def test_parse_laguna_multiple_tool_calls() -> None:
    content = (
        "<tool_call>terminal<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>\n"
        "<tool_call>terminal<arg_key>cmd</arg_key><arg_value>pwd</arg_value></tool_call>"
    )

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "terminal"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"cmd": "ls -la"}
    assert tool_calls[1]["function"]["name"] == "terminal"
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"cmd": "pwd"}


def test_extract_raw_tool_call_segments_laguna() -> None:
    content = (
        "Let me run a command.\n"
        "<tool_call>terminal<arg_key>cmd</arg_key><arg_value>git status</arg_value></tool_call>\n"
        "Finished checking status."
    )

    tool_calls, remaining_text, changed = extract_raw_tool_call_segments(content)

    assert changed is True
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "terminal"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"cmd": "git status"}
    assert remaining_text == "Let me run a command.\n\nFinished checking status."


def test_parse_laguna_negative_number_arg_value() -> None:
    content = (
        "<tool_call>adjust"
        "<arg_key>offset</arg_key><arg_value>-42</arg_value>"
        "<arg_key>ratio</arg_key><arg_value>-3.14</arg_value>"
        "</tool_call>"
    )

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "adjust"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"offset": -42, "ratio": -3.14}


def test_parse_laguna_multiline_arg_value() -> None:
    code = "def hello():\n    print('world')\n"
    content = (
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>main.py</arg_value>"
        f"<arg_key>content</arg_key><arg_value>{code}</arg_value>"
        "</tool_call>"
    )

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "write_file"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["path"] == "main.py"
    assert args["content"] == code.strip()


def test_parse_laguna_rejects_incomplete_argument_pair() -> None:
    content = "<tool_call>terminal<arg_key>cmd</arg_key>missing-value</tool_call>"

    assert parse_raw_tool_calls(content) == []


def test_parse_laguna_rejects_empty_argument_key() -> None:
    content = "<tool_call>terminal<arg_key> </arg_key><arg_value>pwd</arg_value></tool_call>"

    assert parse_raw_tool_calls(content) == []


def test_parse_identical_tool_calls_preserves_both_invocations() -> None:
    call = "<tool_call><name>ping</name><parameters>{}</parameters></tool_call>"

    tool_calls = parse_raw_tool_calls(call + call)

    assert len(tool_calls) == 2
    assert tool_calls[0]["id"] != tool_calls[1]["id"]


def test_parse_qwen_json_tool_call_format() -> None:
    content = """
    <tool_call>
    {"name":"search","arguments":{"query":"OpenCode proxy"}}
    </tool_call>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "search"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "query": "OpenCode proxy",
    }


def test_parse_qwen_xml_json_tool_call_format_with_newlines() -> None:
    content = """<tool_call>
{"name": "get_weather", "arguments": {"location":"Tokyo"}}
</tool_call>"""

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Tokyo"}


def test_parse_qwen_xml_json_multiple_tool_calls() -> None:
    content = """<tool_call>
{"name": "get_weather", "arguments": {"location":"Shanghai"}}
</tool_call><tool_call>
{"name": "add", "arguments": {"x":1,"y":2}}
</tool_call>"""

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Shanghai"}
    assert tool_calls[1]["function"]["name"] == "add"
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"x": 1, "y": 2}


def test_parse_qwen_xml_json_escaped_function_name() -> None:
    content = """<tool_call>
{"name":"say_\\"hi","arguments":{}}
</tool_call>"""

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == 'say_"hi'
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {}


def test_fixture_corpus_parses_supported_tool_call_formats() -> None:
    expected_names = {
        "deepseek_dsml.txt": "bash",
        "qwen_xml.txt": "read_file",
        "qwen_json.txt": "search",
    }

    for fixture_name, expected_name in expected_names.items():
        tool_calls = parse_raw_tool_calls((FIXTURE_DIR / fixture_name).read_text())
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == expected_name


def test_extract_raw_tool_call_segments_preserves_surrounding_text() -> None:
    content = (
        'Before <tool_call><name>ls</name><parameters>{"path":"."}</parameters></tool_call> after.'
    )

    tool_calls, remaining, changed = extract_raw_tool_call_segments(content)

    assert changed is True
    assert len(tool_calls) == 1
    assert remaining == "Before  after."


def test_extract_raw_tool_call_segments_preserves_oversized_block() -> None:
    content = '<tool_call><name>ls</name><parameters>{"path":"."}</parameters></tool_call>'

    tool_calls, remaining, changed = extract_raw_tool_call_segments(
        content,
        max_raw_tool_block_chars=10,
    )

    assert tool_calls == []
    assert remaining == content
    assert changed is False


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


def test_convert_non_streaming_response_repairs_all_choices() -> None:
    body = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": (
                        "<tool_call><name>read</name>"
                        '<parameters>{"path":"a"}</parameters></tool_call>'
                    ),
                },
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {
                    "content": (
                        "<tool_call><name>write</name>"
                        '<parameters>{"path":"b"}</parameters></tool_call>'
                    ),
                },
                "finish_reason": "stop",
            },
        ],
    }

    converted, changed = convert_chat_completion_response(body)

    assert changed is True
    assert converted["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert converted["choices"][1]["message"]["tool_calls"][0]["function"]["name"] == "write"
    assert converted["choices"][0]["finish_reason"] == "tool_calls"
    assert converted["choices"][1]["finish_reason"] == "tool_calls"


def test_convert_non_streaming_response_scans_reasoning_content() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": (
                        "<tool_call><name>read</name>"
                        '<parameters>{"path":"README.md"}</parameters></tool_call>'
                    ),
                },
                "finish_reason": "stop",
            }
        ]
    }

    converted, changed = convert_chat_completion_response(body)

    assert changed is True
    message = converted["choices"][0]["message"]
    assert message["reasoning_content"] is None
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "read"


def test_convert_non_streaming_response_preserves_over_limit_tool_call() -> None:
    content = (
        '<tool_call><name>write</name><parameters>{"content":"abcdef"}</parameters></tool_call>'
    )
    body = {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ]
    }

    converted, changed = convert_chat_completion_response(
        body,
        max_tool_argument_chars=5,
    )

    assert changed is False
    assert converted["choices"][0]["message"]["content"] == content


def test_tool_calls_within_limits_rejects_too_many_calls() -> None:
    tool_calls = [make_tool_call("a", {}), make_tool_call("b", {})]

    assert tool_calls_within_limits(tool_calls, max_tool_calls=1) is False


def test_tool_calls_within_limits_rejects_large_arguments() -> None:
    tool_calls = [make_tool_call("write", {"content": "abcdef"})]

    assert tool_calls_within_limits(tool_calls, max_tool_argument_chars=5) is False


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


def test_parse_dsml_invoke_preserves_string_false_json_types() -> None:
    content = f"""
    <{BAR}DSML{BAR}tool_calls>
    <{BAR}DSML{BAR}invoke name="edit">
      <{BAR}DSML{BAR}parameter name="path" string="true">README.md</{BAR}DSML{BAR}parameter>
      <{BAR}DSML{BAR}parameter name="count" string="false">5</{BAR}DSML{BAR}parameter>
      <{BAR}DSML{BAR}parameter name="active" string="false">true</{BAR}DSML{BAR}parameter>
      <{BAR}DSML{BAR}parameter name="tags" string="false">["a","b"]</{BAR}DSML{BAR}parameter>
      <{BAR}DSML{BAR}parameter name="meta" string="false">{{"k":"v"}}</{BAR}DSML{BAR}parameter>
    </{BAR}DSML{BAR}invoke>
    </{BAR}DSML{BAR}tool_calls>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert arguments == {
        "path": "README.md",
        "count": 5,
        "active": True,
        "tags": ["a", "b"],
        "meta": {"k": "v"},
    }


def test_parse_ascii_dsml_invoke_preserves_string_false_json_types() -> None:
    content = """
    <|DSML|tool_calls>
    <|DSML|invoke name="edit">
      <|DSML|parameter name="count" string="false">42</|DSML|parameter>
      <|DSML|parameter name="label" string="true">hello</|DSML|parameter>
    </|DSML|invoke>
    </|DSML|tool_calls>
    """

    tool_calls = parse_raw_tool_calls(content)

    assert len(tool_calls) == 1
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert arguments == {"count": 42, "label": "hello"}


# --- Deduplication tests ---


def test_preserve_identical_tool_calls_across_formats() -> None:
    """Distinct raw blocks remain distinct even when their calls have equal arguments."""
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

    assert len(tool_calls) == 2
    assert tool_calls[0]["id"] != tool_calls[1]["id"]
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
