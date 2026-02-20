"""
Property-based tests for NVIDIA RAG pipeline and worker.
Ensures invariants hold across a wide range of inputs (production-critical for regulated systems).
"""

import string
from pathlib import Path

import pytest

# Hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    given = None
    st = None

# Import pipeline helpers (pure functions we can test)
import importlib.util

PIPELINE_PATH = Path(__file__).resolve().parents[1] / "docker" / "pipelines" / "nvidia_ingest_bridge_pipe.py"
_spec = importlib.util.spec_from_file_location("nvidia_ingest_bridge_pipe", str(PIPELINE_PATH))
_pipe_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_pipe_mod)  # type: ignore[attr-defined]


if HAS_HYPOTHESIS:
    class TestCollectionNameProperties:
        """Property: sanitized collection names always satisfy Milvus constraints."""

        @given(st.text(alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")), min_size=0, max_size=300))
        def test_sanitize_collection_name_non_empty_input_produces_safe_chars(self, name: str):
            """Sanitized names contain only letters, digits, - _ . :"""
            p = _pipe_mod.Pipeline()
            result = p._sanitize_collection_name(name)
            if result:
                allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:")
                assert all(c in allowed for c in result), f"Invalid char in {result!r} from {name!r}"

        @given(st.text(alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")), min_size=0, max_size=300))
        def test_to_milvus_safe_collection_name_valid_milvus(self, name: str):
            """Milvus-safe names: letters, digits, underscore only; cannot start with digit."""
            p = _pipe_mod.Pipeline()
            result = p._to_milvus_safe_collection_name(name)
            if result:
                assert result[0].isalpha() or result[0] == "_", f"Must start with letter/underscore: {result!r}"
                allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                assert all(c in allowed for c in result), f"Invalid char in {result!r}"
                assert len(result) <= 255, "Milvus name length limit"

    class TestLastUserMessageProperties:
        """Property: _last_user_message_text extraction invariants."""

        @given(
            messages=st.lists(
                st.fixed_dictionaries({
                    "role": st.sampled_from(["user", "assistant", "system"]),
                    "content": st.one_of(st.none(), st.text(alphabet=string.printable, max_size=500))
                }),
                max_size=20
            )
        )
        def test_last_user_message_returns_string(self, messages):
            """Always returns a string (never None)."""
            p = _pipe_mod.Pipeline()
            result = p._last_user_message_text(messages)
            assert isinstance(result, str)

        @given(content=st.text(alphabet=string.printable, min_size=1, max_size=200))
        def test_last_user_message_preserves_content_when_last_is_user(self, content):
            """When last message is user with string content, we get that content back (stripped)."""
            p = _pipe_mod.Pipeline()
            messages = [{"role": "user", "content": content}]
            result = p._last_user_message_text(messages)
            assert result == content.strip()


# Property tests that run without hypothesis
class TestPipelineHelperProperties:
    """Property tests that run without hypothesis."""

    def test_parse_collection_name_library_format(self):
        """owui-u-{safe}-library parses as library type."""
        p = _pipe_mod.Pipeline()
        ctype, payload = p._parse_collection_name("owui-u-testuser-library")
        assert ctype == "library"
        assert payload == "testuser"

    def test_parse_collection_name_kb_format(self):
        """owui-kb-{id} parses as kb type."""
        p = _pipe_mod.Pipeline()
        ctype, payload = p._parse_collection_name("owui-kb-abc123")
        assert ctype == "kb"
        assert payload == "abc123"

    def test_parse_collection_name_unknown_prefix_returns_unknown(self):
        """Non-owui prefix returns unknown."""
        p = _pipe_mod.Pipeline()
        ctype, _ = p._parse_collection_name("other-prefix-name")
        assert ctype == "unknown"

    def test_looks_like_collection_token_with_prefix(self):
        """Tokens with COLLECTION_PREFIX- are collection tokens."""
        p = _pipe_mod.Pipeline()
        assert p._looks_like_collection_token("owui-custom-collection") is True

    def test_looks_like_collection_token_plain_word_not_token(self):
        """Plain words without punctuation are not collection tokens."""
        p = _pipe_mod.Pipeline()
        assert p._looks_like_collection_token("hello") is False
