"""Download real RBI Master Circulars / Master Directions / notifications.

RBI publishes every circular under a small number of stable ASPX endpoints:

    Listing  https://www.rbi.org.in/Scripts/BS_ViewMasCirculardetails.aspx   (Master Circulars)
             https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx        (Master Directions)
             https://www.rbi.org.in/Scripts/NotificationUser.aspx            (Notifications)
    Document https://www.rbi.org.in/Scripts/BS_ViewMasCirculardetails.aspx?id=<id>
             https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=<id>&Mode=0
    PDFs     https://rbidocs.rbi.org.in/rdocs/...PDFs/....PDF

This module scrapes the listing page for document links, downloads each document
(HTML page or its attached PDF), and writes it to the output directory together
with a ``manifest.json`` recording URL, title and fetch time — so the downloaded
corpus is reproducible and auditable.

Usage
-----
    python -m ingestion.fetch_rbi --category master-circulars --limit 40
    python -m ingestion.fetch_rbi --category master-directions --out corpus/raw
    python -m ingestion.fetch_rbi --url "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13699&Mode=0"

Then index what was downloaded:

    python -m ingestion.reindex --path corpus/raw

Note: rbi.org.in must be reachable from the machine running this. In a sandboxed
CI environment without egress the command fails fast with a clear message and
you can fall back to ``corpus/seed`` (see ``corpus/seed/README.md``).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx

from core.logging_conf import get_logger

log = get_logger(__name__)

BASE = "https://www.rbi.org.in"
LISTINGS = {
    "master-circulars": f"{BASE}/Scripts/BS_ViewMasCirculardetails.aspx",
    "master-directions": f"{BASE}/Scripts/BS_ViewMasDirections.aspx",
    "notifications": f"{BASE}/Scripts/NotificationUser.aspx",
}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 RegulationTrackingRAG/1.0 (research use)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}
DOC_LINK_RE = re.compile(
    r"(BS_ViewMasCirculardetails\.aspx\?id=\d+|BS_ViewMasDirections\.aspx\?id=\d+|NotificationUser\.aspx\?Id=\d+)",
    re.I,
)


@dataclass
class FetchedDoc:
    url: str
    title: str
    filename: str
    content_type: str
    bytes: int
    fetched_at: str


def _slug(text: str, limit: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (text[:limit] or "rbi-document").strip("-")


def _client(timeout: float = 45.0) -> httpx.Client:
    return httpx.Client(
        headers=HEADERS, timeout=timeout, follow_redirects=True, verify=True, http2=False
    )


def discover_links(client: httpx.Client, listing_url: str, limit: int) -> list[tuple[str, str]]:
    """Return [(absolute_url, title)] from an RBI listing page."""
    from bs4 import BeautifulSoup

    resp = client.get(listing_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not DOC_LINK_RE.search(href):
            continue
        url = urljoin(listing_url, href)
        if url in seen:
            continue
        seen.add(url)
        title = a.get_text(" ", strip=True) or "RBI document"
        if len(title) < 8:
            continue
        out.append((url, title))
        if len(out) >= limit:
            break
    return out


def _pdf_link(html: str, page_url: str) -> str | None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            return urljoin(page_url, href)
    return None


def fetch_document(
    client: httpx.Client, url: str, title: str, out_dir: Path, prefer_pdf: bool = True
) -> FetchedDoc | None:
    resp = client.get(url)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "").split(";")[0].strip()

    data = resp.content
    suffix = ".html"
    if "pdf" in content_type:
        suffix = ".pdf"
    elif prefer_pdf:
        pdf_url = _pdf_link(resp.text, url)
        if pdf_url:
            try:
                pdf_resp = client.get(pdf_url)
                pdf_resp.raise_for_status()
                if pdf_resp.content[:4] == b"%PDF":
                    data, suffix, content_type, url = (
                        pdf_resp.content,
                        ".pdf",
                        "application/pdf",
                        pdf_url,
                    )
            except httpx.HTTPError as exc:
                log.warning("PDF fetch failed for %s (%s); keeping HTML.", pdf_url, exc)

    if len(data) < 500:
        log.warning("Skipping %s: response too small (%d bytes)", url, len(data))
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_slug(title)}{suffix}"
    counter = 1
    while path.exists():
        path = out_dir / f"{_slug(title)}-{counter}{suffix}"
        counter += 1
    path.write_bytes(data)
    log.info("Saved %s (%.1f KB) <- %s", path.name, len(data) / 1024, url)

    return FetchedDoc(
        url=url,
        title=title,
        filename=path.name,
        content_type=content_type,
        bytes=len(data),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def write_manifest(out_dir: Path, docs: list[FetchedDoc]) -> Path:
    manifest_path = out_dir / "manifest.json"
    existing: list[dict] = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:  # pragma: no cover
            existing = []
    known = {d.get("url") for d in existing}
    existing.extend(asdict(d) for d in docs if d.url not in known)
    manifest_path.write_text(json.dumps(existing, indent=2))
    return manifest_path


def run(
    category: str = "master-circulars",
    limit: int = 25,
    out: str | Path = "corpus/raw",
    urls: list[str] | None = None,
    delay: float = 1.5,
) -> list[FetchedDoc]:
    out_dir = Path(out)
    fetched: list[FetchedDoc] = []

    try:
        with _client() as client:
            targets: list[tuple[str, str]]
            if urls:
                targets = [(u, f"rbi-{i}") for i, u in enumerate(urls, start=1)]
            else:
                listing = LISTINGS.get(category)
                if not listing:
                    raise SystemExit(
                        f"Unknown category '{category}'. Options: {sorted(LISTINGS)}"
                    )
                log.info("Discovering documents from %s", listing)
                targets = discover_links(client, listing, limit)
                log.info("Found %d document links", len(targets))

            for url, title in targets:
                try:
                    doc = fetch_document(client, url, title, out_dir)
                    if doc:
                        fetched.append(doc)
                except httpx.HTTPError as exc:
                    log.warning("Failed %s: %s", url, exc)
                time.sleep(delay)  # be polite to rbi.org.in
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"Could not reach rbi.org.in ({exc}). Check network egress/proxy. "
            "Offline? Use the fixtures in corpus/seed (see corpus/seed/README.md)."
        ) from exc

    if fetched:
        manifest = write_manifest(out_dir, fetched)
        log.info("Wrote %d documents; manifest at %s", len(fetched), manifest)
    return fetched


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RBI circulars into a local corpus.")
    parser.add_argument("--category", default="master-circulars", choices=sorted(LISTINGS))
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out", default="corpus/raw")
    parser.add_argument("--url", action="append", dest="urls", help="Fetch a specific URL (repeatable)")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    args = parser.parse_args()

    docs = run(args.category, args.limit, args.out, args.urls, args.delay)
    print(f"Downloaded {len(docs)} documents to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
