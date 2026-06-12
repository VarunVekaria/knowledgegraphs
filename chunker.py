"""Split a document into contiguous, section-aligned chunks.

Guarantees (enforced here, not left to the LLM):
  - every line of the source belongs to exactly one chunk (full coverage)
  - chunks never overlap: chunk[n+1].start_line == chunk[n].end_line + 1
  - the first chunk starts at line 1, the last ends at the final line

Chunks are built by grouping whole paragraphs (never splitting one) until a
target size is reached, preferring to start a fresh chunk at a section heading.
"""

import re

TARGET_CHARS = 1800  # soft cap; a chunk closes once it exceeds this


def _normalize(text: str) -> str:
    r"""Turn literal '\n' escape sequences into real newlines, unify line ends."""
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _looks_like_heading(line: str) -> bool:
    """Heuristic: short, title-ish line that introduces a new section."""
    s = line.strip()
    if not s or len(s) > 90:
        return False
    # e.g. "Strategic Risks", "Item 1A. - Risk Factors", or a "Title — sentence"
    if re.match(r"^Item\s+\d+", s):
        return True
    if " — " in s or " - " in s:  # risk-factor style "Heading — description"
        return True
    # Title Case-ish line with no terminal punctuation.
    return s[-1] not in ".!?:;" and s[:1].isupper()


def chunk_document(text: str, source: str, target_chars: int = TARGET_CHARS) -> list[dict]:
    """Return a list of chunk dicts covering the whole document contiguously."""
    lines = _normalize(text).split("\n")  # 1-indexed conceptually

    # Group consecutive non-blank lines into paragraphs, tracking line spans.
    paragraphs = []  # (start_line, end_line, text)
    buf, start = [], None
    for i, line in enumerate(lines, start=1):
        if line.strip():
            if start is None:
                start = i
            buf.append(line)
        elif buf:
            paragraphs.append((start, i - 1, " ".join(buf)))
            buf, start = [], None
    if buf:
        paragraphs.append((start, len(lines), " ".join(buf)))

    if not paragraphs:
        return []

    # Pack paragraphs into chunks, closing on size or at the next heading.
    chunks = []
    cur_start = paragraphs[0][0]
    cur_end = paragraphs[0][1]
    cur_text = [paragraphs[0][2]]

    for start_l, end_l, para in paragraphs[1:]:
        size = sum(len(p) for p in cur_text)
        boundary = size >= target_chars or (_looks_like_heading(para) and size > 0)
        if boundary:
            chunks.append((cur_start, cur_end, "\n\n".join(cur_text)))
            cur_start, cur_end, cur_text = start_l, end_l, [para]
        else:
            cur_end = end_l
            cur_text.append(para)
    chunks.append((cur_start, cur_end, "\n\n".join(cur_text)))

    # Stitch line spans so coverage is contiguous (no gaps from blank lines).
    result = []
    for index, (s, e, body) in enumerate(chunks, start=1):
        start_line = result[-1]["end_line"] + 1 if result else 1
        result.append(
            {
                "chunk_id": f"{source}#{index}",
                "index": index,
                "source": source,
                "start_line": start_line,
                "end_line": e,
                "text": body,
            }
        )
    # Make the final chunk reach the true last line.
    if result:
        result[-1]["end_line"] = max(result[-1]["end_line"], len(lines))
    return result
