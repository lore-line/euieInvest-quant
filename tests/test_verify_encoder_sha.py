"""Tests for ``quant.tracks.verify_encoder_sha``.

Defense against silent encoder swaps in the downstream A/B sequence.
Approved by server team on PR #1 issuecomment-4462335903 thread.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quant.tracks import verify_encoder_sha


def _make_encoder(tmp_path: Path, content: bytes = b"fake-encoder-bytes") -> Path:
    enc = tmp_path / "encoder.pt"
    enc.write_bytes(content)
    return enc


def test_returns_actual_sha_when_no_expected(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path, b"abc123")
    expected = hashlib.sha256(b"abc123").hexdigest()
    sha = verify_encoder_sha(enc, expected_sha=None)
    assert sha == expected
    assert len(sha) == 64
    assert sha.islower()


def test_passes_when_expected_matches_bare_hex(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path, b"payload-v2")
    expected = hashlib.sha256(b"payload-v2").hexdigest()
    # No exception
    sha = verify_encoder_sha(enc, expected_sha=expected)
    assert sha == expected


def test_passes_when_expected_has_sha256_prefix(tmp_path: Path) -> None:
    """foundation_pretrain.py writes manifests as 'sha256:abc...' — the
    helper must tolerate that form when callers pass the manifest value
    straight through."""
    enc = _make_encoder(tmp_path, b"payload-v2")
    actual = hashlib.sha256(b"payload-v2").hexdigest()
    sha = verify_encoder_sha(enc, expected_sha=f"sha256:{actual}")
    assert sha == actual


def test_passes_when_expected_has_uppercase_hex(tmp_path: Path) -> None:
    """Hex SHA strings sometimes get capitalized in transit (cut/paste from
    GitHub UI). The comparison should be case-insensitive."""
    enc = _make_encoder(tmp_path, b"payload-v2")
    actual = hashlib.sha256(b"payload-v2").hexdigest()
    sha = verify_encoder_sha(enc, expected_sha=actual.upper())
    assert sha == actual


def test_passes_with_whitespace_in_expected(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path, b"x")
    actual = hashlib.sha256(b"x").hexdigest()
    sha = verify_encoder_sha(enc, expected_sha=f"  sha256:{actual}  ")
    assert sha == actual


def test_raises_on_mismatch(tmp_path: Path) -> None:
    """The core defense — bare-hex form."""
    enc = _make_encoder(tmp_path, b"v1-content")
    wrong = hashlib.sha256(b"v2-different-content").hexdigest()
    with pytest.raises(ValueError, match="Encoder SHA mismatch"):
        verify_encoder_sha(enc, expected_sha=wrong)


def test_raises_on_mismatch_with_sha256_prefix(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path, b"v1-content")
    wrong = hashlib.sha256(b"v2-different-content").hexdigest()
    with pytest.raises(ValueError, match="Encoder SHA mismatch"):
        verify_encoder_sha(enc, expected_sha=f"sha256:{wrong}")


def test_error_message_includes_both_values(tmp_path: Path) -> None:
    """When the gate fires, the operator needs to see both SHAs to triage.
    Don't bury this in a generic 'mismatch' message."""
    enc = _make_encoder(tmp_path, b"actual-content")
    actual = hashlib.sha256(b"actual-content").hexdigest()
    wrong = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(ValueError) as excinfo:
        verify_encoder_sha(enc, expected_sha=wrong)
    msg = str(excinfo.value)
    assert actual in msg, "actual SHA missing from error message"
    assert wrong in msg, "expected SHA missing from error message"
    assert "swapped" in msg.lower() or "corrupted" in msg.lower(), \
        "operator-facing message should mention the threat model"


def test_consistent_across_calls(tmp_path: Path) -> None:
    """Same encoder.pt → same SHA. Sanity check that we're using a
    deterministic hash."""
    enc = _make_encoder(tmp_path, b"some-bytes")
    a = verify_encoder_sha(enc)
    b = verify_encoder_sha(enc)
    assert a == b


def test_different_content_different_sha(tmp_path: Path) -> None:
    enc1 = _make_encoder(tmp_path, b"v1")
    tmp2 = tmp_path / "subdir"
    tmp2.mkdir()
    enc2 = tmp2 / "encoder.pt"
    enc2.write_bytes(b"v2")
    assert verify_encoder_sha(enc1) != verify_encoder_sha(enc2)


def test_missing_file_raises(tmp_path: Path) -> None:
    """Bad path should raise a clear OS error, not an opaque hashlib error."""
    nonexistent = tmp_path / "no-such-encoder.pt"
    with pytest.raises(FileNotFoundError):
        verify_encoder_sha(nonexistent)
