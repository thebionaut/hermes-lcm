"""Pre-compaction extraction — extract decisions and commitments before summarization.

Best-effort: failures never block compaction. Extracted content is written to
daily note files so key decisions survive even if the DAG summary loses nuance.
"""

import errno
import json
import logging
import os
import re
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .model_routing import apply_lcm_model_route

try:  # POSIX advisory locks.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows hosts.
    _fcntl = None

try:  # Windows byte-range locks.
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX hosts.
    _msvcrt = None

logger = logging.getLogger(__name__)


def _acquire_extraction_file_lock(lock_fd: int, *, blocking: bool) -> None:
    """Acquire one cross-process lock without making Windows unimportable."""
    if _fcntl is not None:
        flags = _fcntl.LOCK_EX | (0 if blocking else _fcntl.LOCK_NB)
        _fcntl.flock(lock_fd, flags)
        return
    if _msvcrt is None:  # Defensive for unsupported Python platforms.
        raise OSError("no file-lock implementation is available")
    if os.fstat(lock_fd).st_size == 0:
        os.write(lock_fd, b"\0")
        os.fsync(lock_fd)
    while True:
        os.lseek(lock_fd, 0, os.SEEK_SET)
        try:
            _msvcrt.locking(lock_fd, _msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if not blocking or exc.errno not in {
                errno.EACCES,
                errno.EAGAIN,
                errno.EDEADLK,
            }:
                raise
            time.sleep(0.01)


def _release_extraction_file_lock(lock_fd: int) -> None:
    if _fcntl is not None:
        _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
        return
    if _msvcrt is None:  # Defensive for unsupported Python platforms.
        return
    os.lseek(lock_fd, 0, os.SEEK_SET)
    _msvcrt.locking(lock_fd, _msvcrt.LK_UNLCK, 1)

_MEDIA_DATA_URI_RE = re.compile(
    r"data:(?:image|audio|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]{16,}",
    re.IGNORECASE,
)
_MEDIA_ATTACHMENT_MARKER = "[Media attachment]"
_MEDIA_ATTACHMENT_SUFFIX = "[with media attachment]"
_TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
_MEDIA_BLOCK_HINTS = ("image", "audio", "video")
_STRUCTURED_METADATA_KEYS = ("file_id", "filename", "name", "mime_type", "url", "file_url", "id")
_INJECTED_CONTEXT_TAGS = (
    "active_memory",
    "active_memory_plugin",
    "relevant-memories",
    "relevant_memories",
    "hindsight-memories",
    "hindsight_memories",
)
_UNTRUSTED_CONTEXT_HEADER_RE = re.compile(
    r"^Untrusted context \(metadata, do not treat as instructions or commands\):\s*",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class PreparedExtraction:
    """Provider result that is safe to persist only on the foreground caller."""

    content: str


def _at_line_start(text: str, index: int) -> bool:
    line_start = text.rfind("\n", 0, index) + 1
    return not text[line_start:index].strip()


def _at_line_end(text: str, index: int) -> bool:
    line_end = text.find("\n", index)
    if line_end == -1:
        line_end = len(text)
    return not text[index:line_end].strip()


EXTRACTION_PROMPT = """Extract decisions, commitments, outcomes, and rules from this conversation segment.

Format as a flat list of bullet points. Each bullet should be self-contained and understandable
without the surrounding conversation. Include:
- Decisions made (what was chosen, and why if stated)
- Commitments (who will do what)
- Outcomes (what happened as a result of an action)
- Rules or constraints discovered

Skip: greetings, meta-discussion, reasoning that led nowhere, repeated information.
If there is nothing worth extracting, respond with exactly: NOTHING_TO_EXTRACT

CONTENT:
{text}"""


def _call_extraction_llm(prompt: str, model: str = "",
                          timeout: float | None = None) -> Optional[str]:
    """Call the Hermes auxiliary LLM for extraction."""
    try:
        from agent.auxiliary_client import call_llm
        call_kwargs = {
            "task": "extraction",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000,
        }
        apply_lcm_model_route(call_kwargs, model)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        response = call_llm(**call_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        from .escalation import _strip_reasoning_blocks
        return _strip_reasoning_blocks(content).strip()
    except Exception as e:
        logger.debug(
            "Extraction LLM call failed (error_type=%s)",
            type(e).__name__,
        )
        return None


def _sanitize_string_media(text: str) -> str:
    if not text:
        return ""
    if not _MEDIA_DATA_URI_RE.search(text):
        return text

    without_media = _MEDIA_DATA_URI_RE.sub("", text)
    without_media = without_media.strip()
    without_media = re.sub(r"\n{3,}", "\n\n", without_media)

    if not without_media:
        return _MEDIA_ATTACHMENT_MARKER
    if _MEDIA_ATTACHMENT_SUFFIX in without_media:
        return without_media
    return f"{without_media}\n{_MEDIA_ATTACHMENT_SUFFIX}"


def _looks_like_media_block(block_type: str, block: Dict[str, Any]) -> bool:
    if any(hint in block_type for hint in _MEDIA_BLOCK_HINTS):
        return True
    return any(key in block for key in ("image_url", "input_image", "output_image", "audio_url", "video_url"))


def _extract_structured_metadata(block: Dict[str, Any]) -> str:
    parts: List[str] = []
    block_type = str(block.get("type", "")).strip()
    if block_type:
        parts.append(f"type={block_type}")

    for key in _STRUCTURED_METADATA_KEYS:
        value = block.get(key)
        if isinstance(value, dict):
            for nested_key in _STRUCTURED_METADATA_KEYS:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, (str, int, float)) and nested_value:
                    parts.append(f"{nested_key}={nested_value}")
                    break
            continue
        if isinstance(value, (str, int, float)) and value:
            parts.append(f"{key}={value}")

    if not parts:
        return "[Structured content]"
    return "[Structured content: " + ", ".join(dict.fromkeys(parts)) + "]"


def _sanitize_content_block(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _sanitize_string_media(content)
    if isinstance(content, list):
        parts: List[str] = []
        media_seen = False
        for block in content:
            block_text = _sanitize_content_block(block)
            if not block_text:
                continue
            if block_text == _MEDIA_ATTACHMENT_MARKER:
                media_seen = True
                continue
            if block_text.endswith(_MEDIA_ATTACHMENT_SUFFIX):
                media_seen = True
                block_text = block_text[: -len(_MEDIA_ATTACHMENT_SUFFIX)].rstrip()
                if not block_text:
                    continue
            parts.append(block_text)
        combined = "\n".join(part for part in parts if part).strip()
        combined = re.sub(r"\n{3,}", "\n\n", combined)
        if media_seen and combined:
            return f"{combined}\n{_MEDIA_ATTACHMENT_SUFFIX}"
        if media_seen:
            return _MEDIA_ATTACHMENT_MARKER
        return combined
    if isinstance(content, dict):
        block_type = str(content.get("type", "")).lower()
        if block_type in _TEXT_BLOCK_TYPES:
            text_value = content.get("text")
            if isinstance(text_value, dict):
                text_value = text_value.get("value", "")
            if not text_value:
                text_value = content.get("content", "")
            return _sanitize_content_block(text_value)
        if _looks_like_media_block(block_type, content):
            return _MEDIA_ATTACHMENT_MARKER
        for key in ("text", "content"):
            if key in content:
                return _sanitize_content_block(content.get(key))
        return _extract_structured_metadata(content)
    return str(content)


def _select_injected_context_closer(
    text: str,
    opener: re.Match[str],
    close_re: re.Pattern[str],
) -> re.Match[str] | None:
    closers = list(close_re.finditer(text, opener.end()))
    if not closers:
        return None

    # Prompt-injected context normally uses a block shape:
    #   <tag>\n...\n</tag>
    # In that shape, a line-delimited close/open pair inside recalled text is
    # indistinguishable from two adjacent injected blocks with user text between
    # them. Choose safety over preservation: block-shaped repeated same-tag
    # sections are stripped as one untrusted region up to the last line-isolated
    # close, even if that drops a real inter-block gap.
    if _at_line_end(text, opener.end()):
        line_closers = [
            closer
            for closer in closers
            if _at_line_start(text, closer.start()) and _at_line_end(text, closer.end())
        ]
        if not line_closers:
            if len(closers) == 1 and _at_line_end(text, closers[0].end()):
                return closers[0]
            for index, closer in enumerate(closers):
                if index + 1 < len(closers) or not _at_line_start(text, closer.start()):
                    continue
                line_end = text.find("\n", closer.end())
                if line_end == -1:
                    line_end = len(text)
                suffix = text[closer.end() : line_end]
                if "<" not in suffix and suffix.strip():
                    return closer
            return None
        return line_closers[-1]

    # Inline wrappers are stripped one complete block at a time. A later same-tag
    # inline opener may be another injected block with real user/tool text
    # between the two blocks; consuming through the later close would silently
    # delete that interstitial text. Keep the safety-first multi-close behavior
    # for block-shaped wrappers above, where line-delimited recalled text is more
    # likely to be spoofing the context envelope.
    return closers[0]


def strip_injected_context_blocks(text: str) -> str:
    """Remove transient memory/context blocks before compaction summarization."""
    if not text:
        return ""

    cleaned = text
    changed = False
    if "<" not in text:
        cleaned = _UNTRUSTED_CONTEXT_HEADER_RE.sub("", text)
        changed = cleaned != text
        return cleaned.strip() if changed else cleaned

    for tag in _INJECTED_CONTEXT_TAGS:
        escaped = re.escape(tag)
        self_close_re = re.compile(rf"<{escaped}(?:\s[^>]*)?\s*/\s*>", re.IGNORECASE)
        open_re = re.compile(rf"<{escaped}(?:\s[^>]*)?>", re.IGNORECASE)
        close_re = re.compile(rf"</{escaped}\s*>", re.IGNORECASE)
        before_self_close = cleaned
        cleaned = self_close_re.sub("", cleaned)
        changed = changed or cleaned != before_self_close

        while True:
            opener = open_re.search(cleaned)
            if not opener:
                break

            closer = _select_injected_context_closer(cleaned, opener, close_re)
            if closer is None:
                if _at_line_end(cleaned, opener.end()):
                    cleaned = cleaned[: opener.start()]
                else:
                    cleaned = cleaned[: opener.start()] + cleaned[opener.end() :]
                changed = True
                continue
            cleaned = cleaned[: opener.start()] + cleaned[closer.end() :]
            changed = True

    before_header = cleaned
    cleaned = _UNTRUSTED_CONTEXT_HEADER_RE.sub("", cleaned)
    changed = changed or cleaned != before_header
    return cleaned.strip() if changed else cleaned


def _sanitize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            (
                strip_injected_context_blocks(_sanitize_string_media(key))
                if isinstance(key, str)
                else key
            ): _sanitize_json_like(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json_like(item) for item in value]
    if isinstance(value, str):
        return strip_injected_context_blocks(_sanitize_string_media(value))
    return value


def sanitize_pre_compaction_content(text: Any) -> str:
    """Replace inline media/base64 payloads and transient injected context before compaction."""
    return strip_injected_context_blocks(_sanitize_content_block(text))


def sanitize_pre_compaction_tool_arguments(arguments: Any) -> str:
    """Clean tool-call argument payloads while preserving JSON-like structure when possible."""
    if arguments is None:
        return ""
    if isinstance(arguments, (dict, list)):
        return json.dumps(_sanitize_json_like(arguments), ensure_ascii=False)
    if not isinstance(arguments, str):
        return sanitize_pre_compaction_content(arguments)
    try:
        parsed = json.loads(arguments)
    except Exception:
        return sanitize_pre_compaction_content(arguments)
    return json.dumps(_sanitize_json_like(parsed), ensure_ascii=False)


def extract_before_compaction(
    serialized_messages: str,
    output_path: str,
    session_id: str = "",
    model: str = "",
    timeout: float | None = None,
    cancel_event: threading.Event | None = None,
    defer_persistence: bool = False,
) -> bool | PreparedExtraction:
    """Extract decisions from messages about to be compacted and write to a daily file.

    Returns True if extraction succeeded, False otherwise.
    Never raises — failures are logged and swallowed.
    """
    try:
        prompt = EXTRACTION_PROMPT.format(text=serialized_messages)
        result = _call_extraction_llm(prompt, model=model, timeout=timeout)

        if cancel_event is not None and cancel_event.is_set():
            logger.debug("Pre-compaction extraction cancelled before persistence")
            return False

        if not result or result.strip() == "NOTHING_TO_EXTRACT":
            logger.debug("Pre-compaction extraction: nothing to extract")
            prepared = PreparedExtraction(content="")
        else:
            prepared = PreparedExtraction(content=result)

        if defer_persistence:
            return prepared
        return persist_pre_compaction_extraction(
            prepared,
            output_path=output_path,
            session_id=session_id,
            cancel_event=cancel_event,
        )

    except Exception as e:
        logger.warning(
            "Pre-compaction extraction failed (non-blocking) (error_type=%s)",
            type(e).__name__,
        )
        return False


def persist_pre_compaction_extraction(
    prepared: PreparedExtraction,
    *,
    output_path: str,
    session_id: str = "",
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
    now: Callable[[], float] | None = None,
) -> bool:
    """Atomically persist a timely provider result.

    The daily note is replaced only after a same-directory temporary file has
    been flushed and fsynced.  An advisory lock serializes writers across both
    threads and processes, so a failed/partial write cannot expose a prefix and
    concurrent appends cannot lose each other.  Deadline-owned automatic work
    is checked after every preparatory wait/write and immediately before the
    atomic replace; manual callers retain the historical ``deadline=None``
    behavior. ``False`` means no new record remains visible. If the filesystem
    accepts the commit but repeatedly refuses every compensating mutation, the
    complete candidate remains visible, a durable recovery image is retained,
    and this returns ``True`` rather than falsely reporting a rollback.
    """
    try:
        clock = now or time.monotonic
        publication_uncertain = False

        def cancelled_or_late() -> bool:
            return bool(
                (cancel_event is not None and cancel_event.is_set())
                or (deadline is not None and float(clock()) >= float(deadline))
            )

        if cancelled_or_late():
            logger.debug("Pre-compaction extraction cancelled before persistence")
            return False
        result = str(prepared.content or "")
        if not result:
            return True

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if cancelled_or_late():
            logger.debug("Pre-compaction extraction cancelled before append")
            return False

        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = output_dir / f"{date_str}.md"

        header = f"\n\n## Extraction — {datetime.now().strftime('%H:%M')}"
        if session_id:
            header += f" ({session_id})"
        header += "\n\n"
        record = f"{header}{result}\n"

        lock_path = output_dir / f".{file_path.name}.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        locked = False
        candidate_path: Path | None = None
        rollback_path: Path | None = None
        replaced = False
        existed = False

        def restore_prior_target() -> bool:
            """Retry compensation while the interprocess lock remains held."""
            nonlocal rollback_path
            for attempt in range(3):
                try:
                    if rollback_path is not None:
                        os.replace(rollback_path, file_path)
                        rollback_path = None
                    elif not existed:
                        file_path.unlink(missing_ok=True)
                    directory_fd = os.open(output_dir, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    return True
                except Exception:
                    if attempt == 2:
                        logger.critical(
                            "Pre-compaction extraction rollback failed after retries",
                            exc_info=True,
                        )
                    else:
                        time.sleep(0.01)
            return False

        try:
            if deadline is None and cancel_event is None:
                _acquire_extraction_file_lock(lock_fd, blocking=True)
                locked = True
            else:
                while not locked:
                    if cancelled_or_late():
                        logger.debug(
                            "Pre-compaction extraction cancelled before file lock"
                        )
                        return False
                    try:
                        _acquire_extraction_file_lock(lock_fd, blocking=False)
                        locked = True
                    except OSError as exc:
                        if exc.errno not in {
                            errno.EACCES,
                            errno.EAGAIN,
                            errno.EDEADLK,
                        }:
                            raise
                        remaining = (
                            None
                            if deadline is None
                            else max(0.0, float(deadline) - float(clock()))
                        )
                        time.sleep(
                            0.01 if remaining is None else min(0.01, remaining)
                        )

            if cancelled_or_late():
                logger.debug("Pre-compaction extraction cancelled after file lock")
                return False

            existed = file_path.exists()
            # Preserve the pre-existing note byte-for-byte. Reading as text
            # would silently normalize CRLF files and reject unrelated legacy
            # bytes that append mode historically left untouched.
            previous = file_path.read_bytes() if existed else b""
            previous_mode = (
                stat.S_IMODE(file_path.stat().st_mode) if existed else 0o600
            )
            if cancelled_or_late():
                logger.debug("Pre-compaction extraction cancelled before temp write")
                return False

            candidate_fd, candidate_name = tempfile.mkstemp(
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                dir=output_dir,
                text=True,
            )
            os.close(candidate_fd)
            candidate_path = Path(candidate_name)
            candidate_content = previous + record.encode("utf-8")
            os.chmod(candidate_path, previous_mode)
            with open(candidate_path, "wb") as candidate:
                written = candidate.write(candidate_content)
                if written != len(candidate_content):
                    raise OSError("short extraction temporary-file write")
                candidate.flush()
                os.fsync(candidate.fileno())

            if cancelled_or_late():
                logger.debug("Pre-compaction extraction cancelled after temp write")
                return False

            # Prepare an already-durable rollback image before the atomic
            # commit.  This keeps an fsync/late-observation failure from
            # returning False while leaving a newly visible record behind.
            if existed:
                rollback_fd, rollback_name = tempfile.mkstemp(
                    prefix=f".{file_path.name}.rollback.",
                    suffix=".tmp",
                    dir=output_dir,
                    text=True,
                )
                os.close(rollback_fd)
                rollback_path = Path(rollback_name)
                os.chmod(rollback_path, previous_mode)
                with open(rollback_path, "wb") as rollback:
                    written = rollback.write(previous)
                    if written != len(previous):
                        raise OSError("short extraction rollback-file write")
                    rollback.flush()
                    os.fsync(rollback.fileno())

            if cancelled_or_late():
                logger.debug("Pre-compaction extraction cancelled before replace")
                return False

            os.replace(candidate_path, file_path)
            candidate_path = None
            replaced = True
            directory_fd = os.open(output_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

            if cancelled_or_late():
                logger.debug("Pre-compaction extraction expired during commit; reverting")
                if restore_prior_target():
                    replaced = False
                    return False
                publication_uncertain = True
                if rollback_path is not None:
                    logger.critical(
                        "Retaining extraction recovery image at %s",
                        rollback_path,
                    )
                    rollback_path = None
                return True
        except BaseException:
            if replaced and not restore_prior_target():
                publication_uncertain = True
                if rollback_path is not None:
                    logger.critical(
                        "Retaining extraction recovery image at %s",
                        rollback_path,
                    )
                    rollback_path = None
            raise
        finally:
            for temporary_path in (candidate_path, rollback_path):
                if temporary_path is None:
                    continue
                try:
                    temporary_path.unlink(missing_ok=True)
                except Exception:
                    # A cleanup failure must not turn a completed durable
                    # replace into a False result. The unique hidden file is
                    # harmless and a later writer does not consult it.
                    logger.warning(
                        "Pre-compaction extraction temporary cleanup failed",
                        exc_info=True,
                    )
            try:
                if locked:
                    _release_extraction_file_lock(lock_fd)
            except Exception:
                logger.warning(
                    "Pre-compaction extraction file unlock failed",
                    exc_info=True,
                )
            finally:
                try:
                    os.close(lock_fd)
                except Exception:
                    logger.warning(
                        "Pre-compaction extraction lock close failed",
                        exc_info=True,
                    )

        logger.info("Pre-compaction extraction written to %s", file_path)
        return True
    except Exception as e:
        if publication_uncertain:
            logger.critical(
                "Pre-compaction extraction remains durably published after "
                "filesystem rollback failure (error_type=%s)",
                type(e).__name__,
            )
            return True
        logger.warning(
            "Pre-compaction extraction persistence failed (non-blocking) "
            "(error_type=%s)",
            type(e).__name__,
        )
        return False
