"""
Helpers for indexing public web pages (e.g. product / service pages) into the
vector store.

URLs are listed one per line in `product_pages.txt` (blank lines and lines
starting with '#' are ignored). Only http/https URLs are accepted. The list is
maintained by us, never taken from user input.
"""
import logging
import os
from urllib.parse import urlparse

import requests
from lxml import html as lxml_html

logger = logging.getLogger("ilios-api.web")

REQUEST_TIMEOUT = 20          # seconds
MAX_PAGE_BYTES = 5 * 1024 * 1024
USER_AGENT = "IliosBot-Indexer/1.0"

# Page chrome that repeats on every page and would pollute search results.
_DROP_TAGS = (
    "script", "style", "noscript", "nav", "header", "footer",
    "form", "svg", "iframe", "button",
)
# Elements that should start/end a new line in the extracted text.
_BLOCK_TAGS = (
    "p", "div", "section", "article", "li", "ul", "ol", "br", "tr",
    "h1", "h2", "h3", "h4", "h5", "h6",
)


def load_urls(path: str) -> list[str]:
    """Read the list of page URLs to index. Missing file means no pages."""
    if not os.path.exists(path):
        return []

    urls: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if urlparse(line).scheme not in ("http", "https"):
                logger.warning("Ignoring non-http(s) entry in %s: %s", path, line)
                continue
            if line not in urls:
                urls.append(line)
    return urls


def read_web_page(url: str) -> str:
    """Download a page and return its readable text (empty string if none)."""
    resp = requests.get(
        url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    if len(resp.content) > MAX_PAGE_BYTES:
        raise ValueError(f"Page is larger than {MAX_PAGE_BYTES} bytes: {url}")

    tree = lxml_html.fromstring(resp.content)

    for el in tree.xpath(" | ".join(f"//{t}" for t in _DROP_TAGS)):
        el.drop_tree()  # drop_tree keeps the text that follows the element

    for el in tree.iter(*_BLOCK_TAGS):
        el.text = "\n" + (el.text or "")
        el.tail = "\n" + (el.tail or "")

    lines = (" ".join(ln.split()) for ln in tree.text_content().splitlines())
    return "\n".join(ln for ln in lines if ln)
