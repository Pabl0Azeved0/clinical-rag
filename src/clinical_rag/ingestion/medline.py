from __future__ import annotations

import html
import io
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import requests

from clinical_rag.domain.models import Document

_MEDLINE_XML_INDEX = "https://medlineplus.gov/xml.html"
_ZIP_PATTERN = re.compile(r'href="(https://medlineplus\.gov/xml/mplus_topics_compressed_[\d-]+\.zip)"')


def discover_latest_url() -> str:
    """Fetch the MedlinePlus XML index page and return the URL of the newest zip."""
    resp = requests.get(_MEDLINE_XML_INDEX, timeout=30)
    resp.raise_for_status()
    matches = _ZIP_PATTERN.findall(resp.text)
    if not matches:
        raise RuntimeError("No mplus_topics_compressed_*.zip link found on xml.html")
    # Links are listed newest-first; take the first one
    return matches[0]


def download_topics(raw_dir: Path) -> Path:
    """Download the latest MedlinePlus zip, unzip it, and return the .xml path.

    Skips the download if the xml file already exists in raw_dir.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = discover_latest_url()
    zip_name = url.split("/")[-1]  # e.g. mplus_topics_compressed_2025-06-25.zip
    date_part = zip_name.replace("mplus_topics_compressed_", "").replace(".zip", "")
    xml_name = f"mplus_topics_{date_part}.xml"
    xml_path = raw_dir / xml_name

    if xml_path.exists():
        print(f"Already downloaded: {xml_path}", file=sys.stderr)
        return xml_path

    print(f"Downloading {url} ...", file=sys.stderr)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    zip_bytes = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_bytes) as zf:
        zf.extract(xml_name, path=raw_dir)

    print(f"Extracted to {xml_path}", file=sys.stderr)
    return xml_path


class _HTMLStripper(HTMLParser):
    """Minimal HTML tag stripper using stdlib html.parser."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(s: str) -> str:
    """Unescape HTML entities, strip tags, and collapse whitespace."""
    unescaped = html.unescape(s)
    stripper = _HTMLStripper()
    stripper.feed(unescaped)
    text = stripper.get_text()
    return re.sub(r"\s+", " ", text).strip()


def parse_topics(xml_path: Path) -> list[Document]:
    """Parse MedlinePlus XML and return English-only Documents with non-empty summaries."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    docs: list[Document] = []
    for topic in root.findall("health-topic"):
        if topic.get("language") != "English":
            continue

        summary_el = topic.find("full-summary")
        if summary_el is None or not (summary_el.text or "").strip():
            continue

        text = _strip_html(summary_el.text)
        if not text:
            continue

        also_called = [el.text for el in topic.findall("also-called") if el.text]

        docs.append(
            Document(
                id=topic.get("id", ""),
                title=topic.get("title", ""),
                url=topic.get("url", ""),
                text=text,
                also_called=also_called,
            )
        )

    return docs
