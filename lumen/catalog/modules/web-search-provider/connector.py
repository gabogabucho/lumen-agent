"""Web search module — real handlers for web__search and web__extract.

The built-in `web` connector ships without a handler, so installing the
catalog entry used to leave the tools inert. This module registers both
actions. Search goes through a provider chosen by configuration. Extract
fetches one URL. Neither follows a redirect until the next hop has been
checked: localhost, private ranges, and link-local addresses are refused
before the request and again after every redirect.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, urljoin, urlsplit

logger = logging.getLogger("lumen.module.web_search")

MODULE_NAME = "x-lumen-web-search"
DEFAULT_PROVIDER = "duckduckgo"
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_MAX_RESULTS = 5
MAX_REDIRECTS = 5
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
TAVILY_URL = "https://api.tavily.com/search"

_BLOCKED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",
}
_SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-subscription-token",
    "api-key",
    "x-api-key",
}
_REDIRECTS = {301, 302, 303, 307, 308}


class HttpResponse:
    """Minimal response the search runtime knows how to read.

    Tests pass fakes of this shape. The real client also returns it and
    never follows redirects on its own.
    """

    def __init__(self, status: int, headers: dict | None = None, body: bytes | str = b"", url: str = ""):
        self.status = int(status)
        self.headers = {str(key).lower(): str(value) for key, value in (headers or {}).items()}
        if isinstance(body, bytes):
            self.body = body
        else:
            self.body = str(body).encode("utf-8")
        self.url = url


class HttpxHttpClient:
    """httpx wrapper. Redirects and env proxies stay off so SSRF checks win."""

    def __init__(self):
        import httpx

        self._httpx = httpx
        self._client = httpx.Client(follow_redirects=False, trust_env=False)

    def request(self, method: str, url: str, *, headers=None, body=None, timeout=DEFAULT_TIMEOUT, max_bytes=DEFAULT_MAX_BYTES):
        limit = max(int(max_bytes), 0) + 1
        try:
            with self._client.stream(
                method,
                url,
                headers=headers,
                content=body,
                timeout=self._httpx.Timeout(timeout),
            ) as response:
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    if total >= limit:
                        break
                    piece = chunk[: limit - total]
                    chunks.append(piece)
                    total += len(piece)
                return HttpResponse(
                    response.status_code,
                    dict(response.headers),
                    b"".join(chunks),
                    str(response.url),
                )
        except self._httpx.TimeoutException as exc:
            raise TimeoutError("request timed out") from exc
        except self._httpx.HTTPError as exc:
            raise ConnectionError("request failed") from exc

    def close(self):
        self._client.close()


def _err(code: str, message: str, **extra) -> dict:
    payload = {"error": code, "message": message}
    payload.update(extra)
    return payload


def _clamp_int(value, default: int, low: int, high: int) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return default
    if number < low or number > high:
        return default
    return number


def _name_blocked(host: str) -> bool:
    name = host.strip().rstrip(".").lower()
    if name in _BLOCKED_HOSTS:
        return True
    return name.endswith(".localhost") or name.endswith(".local")


def _parse_ip(host: str):
    """Return an IP, None when `host` is a name, or 'blocked' when it is an IP trick."""
    text = host.strip().strip("[]")
    if "%" in text:
        return "blocked"
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        pass
    if text.isdigit():
        number = int(text)
        if 0 <= number <= 0xFFFFFFFF:
            return ipaddress.IPv4Address(number)
        return "blocked"
    # 127.1, 127.0.1, and zero-padded octets are interpreted as addresses by
    # some stacks and as hostnames by others. Refuse the ambiguity.
    if re.fullmatch(r"\d+(?:\.\d+){1,3}", text):
        parts = text.split(".")
        if len(parts) != 4:
            return "blocked"
        if any((len(part) > 1 and part.startswith("0")) or not part.isdigit() or int(part) > 255 for part in parts):
            return "blocked"
        try:
            return ipaddress.ip_address(".".join(str(int(part)) for part in parts))
        except ValueError:
            return "blocked"
    return None


def _ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip.is_multicast or ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_unspecified or ip.is_reserved:
        return True
    return not bool(getattr(ip, "is_global", False))


def default_resolver(host: str) -> list:
    """Resolve a hostname to every address getaddrinfo returns."""
    import socket

    found = []
    seen: set[str] = set()
    for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
        addr = info[4][0]
        if "%" in addr:
            addr = addr.split("%", 1)[0]
        ip = ipaddress.ip_address(addr)
        key = str(ip)
        if key not in seen:
            seen.add(key)
            found.append(ip)
    return found


def _host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _strip_sensitive(headers: dict) -> dict:
    return {key: value for key, value in headers.items() if key.lower() not in _SENSITIVE_HEADERS}


class _DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._capture: str | None = None
        self._buf: list[str] = []
        self._pending_url = ""

    def _classes(self, attrs) -> set[str]:
        for key, value in attrs:
            if key == "class" and value:
                return set(str(value).split())
        return set()

    def _href(self, attrs) -> str:
        for key, value in attrs:
            if key == "href" and value:
                return str(value)
        return ""

    def handle_starttag(self, tag, attrs):
        classes = self._classes(attrs)
        if tag == "a" and ({"result__a", "result-link"} & classes):
            self._pending_url = _result_target(self._href(attrs))
            self._capture = "title"
            self._buf = []
            return
        if {"result__snippet", "result-snippet"} & classes:
            self._capture = "snippet"
            self._buf = []

    def handle_data(self, data):
        if self._capture:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if self._capture == "title" and tag == "a":
            title = _clean_text("".join(self._buf))
            if self._pending_url:
                self.results.append({"title": title, "url": self._pending_url, "snippet": ""})
            self._capture = None
            self._buf = []
            return
        if self._capture == "snippet" and tag in {"a", "td", "div", "span"}:
            snippet = _clean_text("".join(self._buf))
            if self.results and not self.results[-1]["snippet"]:
                self.results[-1]["snippet"] = snippet
            self._capture = None
            self._buf = []


class _HtmlTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.parts.append(data)


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _collapse_text(value: str) -> str:
    lines = [_clean_text(line) for line in str(value or "").splitlines()]
    kept: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if kept and not blank:
                kept.append("")
            blank = True
            continue
        blank = False
        kept.append(line)
    return "\n".join(kept).strip()


def _result_target(href: str) -> str:
    raw = str(href or "").strip()
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlsplit(raw)
    target = parse_qs(parsed.query).get("uddg", [])
    if target and target[0]:
        return target[0]
    return raw


def _media_type(header: str) -> str:
    return str(header or "").split(";", 1)[0].strip().lower()


def _decode(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    lowered = content_type.lower()
    if "charset=" in lowered:
        charset = lowered.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _parse_duckduckgo(body: bytes, content_type: str) -> list[dict]:
    parser = _DuckDuckGoParser()
    parser.feed(_decode(body, content_type))
    parser.close()
    return parser.results


def _parse_json_results(body: bytes, *, bucket: str | None, snippet_key: str) -> list[dict]:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("response was not JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("response was not a JSON object")
    rows = data.get(bucket) if bucket else data
    if bucket == "web":
        rows = (data.get("web") or {}).get("results") if isinstance(data.get("web"), dict) else None
    if not isinstance(rows, list):
        rows = []
    results = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "snippet": str(item.get(snippet_key) or item.get("snippet") or ""),
            }
        )
    return results


class WebSearchRuntime:
    def __init__(self, context, *, client=None, resolver=None, timeout: float | None = None, max_bytes: int | None = None):
        self.context = context
        self.client = client
        self.resolver = resolver
        self._timeout_override = timeout
        self._max_bytes_override = max_bytes
        self._owned_client: HttpxHttpClient | None = None
        self._connector = None
        self._installed: list[tuple] = []
        self._schema_names: list[str] = []

    def close(self):
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    def _setting(self, key: str, *env_names: str, default: str | None = None) -> str | None:
        context = self.context
        if context is not None and hasattr(context, "resolve_setting"):
            for env_name in env_names or (None,):
                value = context.resolve_setting(key, env_name)
                if value:
                    return str(value)
        for env_name in env_names:
            if not env_name:
                continue
            raw = os.environ.get(env_name)
            if raw:
                return raw
        return default

    def provider_name(self) -> str:
        raw = self._setting("provider", "LUMEN_WEB_PROVIDER", default=DEFAULT_PROVIDER)
        return str(raw or DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER

    def timeout(self) -> float:
        if self._timeout_override is not None:
            return float(self._timeout_override)
        raw = self._setting("timeout", "LUMEN_WEB_TIMEOUT")
        return float(_clamp_int(raw, int(DEFAULT_TIMEOUT), 1, 60))

    def max_bytes(self) -> int:
        if self._max_bytes_override is not None:
            return int(self._max_bytes_override)
        raw = self._setting("max_bytes", "LUMEN_WEB_MAX_BYTES")
        return _clamp_int(raw, DEFAULT_MAX_BYTES, 1, 5_000_000)

    def max_results(self, requested=None) -> int:
        if requested not in (None, ""):
            return _clamp_int(requested, DEFAULT_MAX_RESULTS, 1, 10)
        raw = self._setting("max_results", "LUMEN_WEB_MAX_RESULTS")
        return _clamp_int(raw, DEFAULT_MAX_RESULTS, 1, 10)

    def _api_key(self, provider: str) -> str | None:
        if provider == "brave":
            return self._setting("brave_api_key", "BRAVE_API_KEY", "LUMEN_BRAVE_API_KEY")
        if provider == "tavily":
            return self._setting("tavily_api_key", "TAVILY_API_KEY", "LUMEN_TAVILY_API_KEY")
        return None

    def _client(self):
        if self.client is not None:
            return self.client
        if self._owned_client is None:
            self._owned_client = HttpxHttpClient()
        return self._owned_client

    def classify_url(self, url: str) -> dict | None:
        """Return an error dict when `url` must not be fetched."""
        raw = str(url or "").strip()
        if not raw or len(raw) > 2048 or "\\" in raw or "\x00" in raw:
            return _err("blocked_url", "Refusing to fetch this URL.")
        parts = urlsplit(raw)
        if parts.username or parts.password:
            return _err("blocked_url", "Refusing to fetch a URL that embeds credentials.")
        if parts.scheme not in {"http", "https"}:
            return _err("blocked_url", "Only public http and https URLs can be fetched.")
        host = parts.hostname
        if not host:
            return _err("blocked_url", "The URL has no host.")
        host_name = host.lower().rstrip(".")
        if _name_blocked(host_name):
            return _err(
                "blocked_url",
                "Refusing to fetch localhost, a private address, or a link-local address.",
            )
        parsed = _parse_ip(host_name)
        if parsed == "blocked":
            return _err(
                "blocked_url",
                "Refusing to fetch localhost, a private address, or a link-local address.",
            )
        if isinstance(parsed, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            if _ip_blocked(parsed):
                return _err(
                    "blocked_url",
                    "Refusing to fetch localhost, a private address, or a link-local address.",
                )
            return None
        resolver = self.resolver or default_resolver
        try:
            addresses = list(resolver(host_name))
        except Exception:
            logger.info("web fetch host unverified: %s", host_name)
            return _err("unverified_host", "Could not verify that the host is a public address.")
        checked = []
        for item in addresses:
            if isinstance(item, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
                checked.append(item)
                continue
            try:
                checked.append(ipaddress.ip_address(str(item)))
            except ValueError:
                return _err("unverified_host", "Could not verify that the host is a public address.")
        if not checked:
            return _err("unverified_host", "Could not verify that the host is a public address.")
        if any(_ip_blocked(ip) for ip in checked):
            logger.info("web fetch blocked host: %s", host_name)
            return _err(
                "blocked_url",
                "Refusing to fetch localhost, a private address, or a link-local address.",
            )
        return None

    def _fetch(self, method: str, url: str, *, headers: dict | None = None, body: bytes | None = None) -> dict | HttpResponse:
        current = url
        current_method = method
        current_body = body
        current_headers = dict(headers or {})
        origin = _host_of(url)
        client = None
        timeout = self.timeout()
        max_bytes = self.max_bytes()
        for _hop in range(MAX_REDIRECTS + 1):
            blocked = self.classify_url(current)
            if blocked:
                return blocked
            if client is None:
                client = self._client()
            try:
                response = client.request(
                    current_method,
                    current,
                    headers=current_headers,
                    body=current_body,
                    timeout=timeout,
                    max_bytes=max_bytes,
                )
            except TimeoutError:
                return _err("timeout", "The request timed out.")
            except (ConnectionError, OSError):
                return _err("request_failed", "The request failed.")
            status = int(getattr(response, "status", 0) or 0)
            resp_headers = {str(k).lower(): str(v) for k, v in dict(getattr(response, "headers", {}) or {}).items()}
            if status not in _REDIRECTS:
                response.headers = resp_headers
                return response
            location = resp_headers.get("location", "").strip()
            if not location:
                return _err("http_error", f"HTTP {status} redirect missing Location.", status=status)
            nxt = urljoin(current, location)
            next_host = _host_of(nxt)
            # 301/302/303 become GET. 307/308 keep the method and body.
            # Sensitive headers are dropped only when the host changes.
            if status in {301, 302, 303}:
                current_method = "GET"
                current_body = None
            if next_host != origin:
                current_headers = _strip_sensitive(current_headers)
            current = nxt
        return _err("too_many_redirects", "The URL redirected too many times.")

    def _truncate(self, body: bytes) -> tuple[bytes, bool]:
        limit = self.max_bytes()
        if len(body) > limit:
            return body[:limit], True
        return body, False

    def _keep_result(self, url: str) -> bool:
        if not url:
            return False
        return self.classify_url(url) is None

    def _search_sync(self, query: str, max_results) -> dict:
        text = str(query or "").strip()
        if not text:
            return _err("query_required", "A search query is required.")
        if len(text) > 2000:
            text = text[:2000]
        provider = self.provider_name()
        limit = self.max_results(max_results)
        prepared = self._prepare_search(provider, text, limit)
        if "error" in prepared:
            return prepared
        fetched = self._fetch(prepared["method"], prepared["url"], headers=prepared["headers"], body=prepared["body"])
        if isinstance(fetched, dict):
            return fetched
        if not 200 <= int(fetched.status) < 300:
            return _err("http_error", f"HTTP {fetched.status}", status=int(fetched.status))
        body, _truncated = self._truncate(fetched.body)
        content_type = fetched.headers.get("content-type", "")
        try:
            rows = prepared["parse"](body, content_type)
        except ValueError:
            return _err("bad_response", f"{provider} returned a response that could not be read.")
        results = []
        for row in rows:
            target = str(row.get("url") or "").strip()
            if not self._keep_result(target):
                continue
            results.append(
                {
                    "title": _clean_text(row.get("title") or ""),
                    "url": target,
                    "snippet": _clean_text(row.get("snippet") or ""),
                }
            )
            if len(results) >= limit:
                break
        return {"status": "ok", "provider": provider, "query": text, "results": results}

    def _prepare_search(self, provider: str, query: str, limit: int) -> dict:
        if provider == "duckduckgo":
            return {
                "method": "POST",
                "url": DUCKDUCKGO_URL,
                "headers": {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "text/html",
                    "User-Agent": USER_AGENT,
                },
                "body": f"q={quote_plus(query)}".encode("utf-8"),
                "parse": lambda body, content_type: _parse_duckduckgo(body, content_type),
            }
        if provider == "brave":
            key = self._api_key("brave")
            if not key:
                return _err(
                    "missing_configuration",
                    "Brave Search needs an API key. Set BRAVE_API_KEY or LUMEN_BRAVE_API_KEY.",
                    provider="brave",
                )
            return {
                "method": "GET",
                "url": f"{BRAVE_URL}?q={quote_plus(query)}&count={limit}",
                "headers": {
                    "X-Subscription-Token": key,
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                "body": None,
                "parse": lambda body, _content_type: _parse_json_results(body, bucket="web", snippet_key="description"),
            }
        if provider == "tavily":
            key = self._api_key("tavily")
            if not key:
                return _err(
                    "missing_configuration",
                    "Tavily needs an API key. Set TAVILY_API_KEY or LUMEN_TAVILY_API_KEY.",
                    provider="tavily",
                )
            return {
                "method": "POST",
                "url": TAVILY_URL,
                "headers": {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                "body": json.dumps(
                    {"api_key": key, "query": query, "max_results": limit, "search_depth": "basic"}
                ).encode("utf-8"),
                "parse": lambda body, _content_type: _parse_json_results(body, bucket="results", snippet_key="content"),
            }
        if provider == "searxng":
            base = str(self._setting("searxng_url", "LUMEN_SEARXNG_URL") or "").strip().rstrip("/")
            if not base:
                return _err(
                    "missing_configuration",
                    "SearXNG needs a base URL. Set LUMEN_SEARXNG_URL.",
                    provider="searxng",
                )
            if "://" not in base:
                base = f"https://{base}"
            blocked = self.classify_url(base)
            if blocked:
                blocked["provider"] = "searxng"
                return blocked
            return {
                "method": "GET",
                "url": f"{base}/search?q={quote_plus(query)}&format=json",
                "headers": {"Accept": "application/json", "User-Agent": USER_AGENT},
                "body": None,
                "parse": lambda body, _content_type: _parse_json_results(body, bucket="results", snippet_key="content"),
            }
        return _err(
            "unknown_provider",
            "Unknown web provider. Use duckduckgo, brave, tavily, or searxng.",
            provider=provider,
        )

    def _extract_sync(self, url: str) -> dict:
        target = str(url or "").strip()
        if not target:
            return _err("url_required", "A URL is required.")
        fetched = self._fetch(
            "GET",
            target,
            headers={"Accept": "text/html, text/plain, application/json", "User-Agent": USER_AGENT},
        )
        if isinstance(fetched, dict):
            return fetched
        if not 200 <= int(fetched.status) < 300:
            return _err("http_error", f"HTTP {fetched.status}", status=int(fetched.status))
        body, truncated = self._truncate(fetched.body)
        content_type = fetched.headers.get("content-type", "")
        media = _media_type(content_type)
        final_url = str(getattr(fetched, "url", "") or target)
        if media in {"text/html", "application/xhtml+xml"} or media.endswith("+html") or (not media and body.lstrip().startswith(b"<")):
            parser = _HtmlTextParser()
            parser.feed(_decode(body, content_type))
            parser.close()
            text = _collapse_text("".join(parser.parts))
            title = _clean_text("".join(parser.title_parts))
        elif (
            not media
            or media.startswith("text/")
            or media in {"application/json", "application/xml", "application/javascript"}
            or media.endswith("+json")
            or media.endswith("+xml")
        ):
            text = _decode(body, content_type)
            title = ""
            if len(text.encode("utf-8")) > self.max_bytes():
                text = text.encode("utf-8")[: self.max_bytes()].decode("utf-8", errors="ignore")
                truncated = True
        else:
            return _err("unsupported_content_type", f"Cannot extract {media or 'this content type'}.", content_type=media)
        if len(text.encode("utf-8")) > self.max_bytes():
            text = text.encode("utf-8")[: self.max_bytes()].decode("utf-8", errors="ignore")
            truncated = True
        return {
            "status": "ok",
            "url": final_url,
            "title": title,
            "text": text,
            "truncated": truncated,
            "content_type": media,
        }

    async def search(self, query: str = "", max_results: int | None = None, **_ignored):
        return await asyncio.to_thread(self._search_sync, query, max_results)

    async def extract(self, url: str = "", **_ignored):
        return await asyncio.to_thread(self._extract_sync, url)

    def bind(self, connector, schemas: dict):
        self._connector = connector
        if connector is None:
            return
        for action, handler in (("search", self.search), ("extract", self.extract)):
            connector.register_handler(action, handler)
            self._installed.append((action, handler))
        setter = getattr(getattr(self.context, "connectors", None), "set_tool_schemas", None)
        if callable(setter) and schemas:
            setter(schemas)
            self._schema_names = list(schemas)

    def unbind(self):
        connector = self._connector
        if connector is not None:
            for action, handler in self._installed:
                if connector._handlers.get(action) is handler:
                    connector._handlers.pop(action, None)
        schemas = getattr(getattr(self.context, "connectors", None), "_tool_schemas", None)
        if isinstance(schemas, dict):
            for name in self._schema_names:
                schemas.pop(name, None)
        self._installed = []
        self._schema_names = []
        self._connector = None


def _tool_metadata(effect: str, risk: str, confirm_required: bool) -> dict:
    return {
        "kind": "module",
        "module": MODULE_NAME,
        "effect": effect,
        "risk": risk,
        "confirm_required": confirm_required,
    }


def _declare_policy(context, action: str, *, risk: str, confirm_required: bool, description: str) -> None:
    declare = getattr(context, "register_tool_policy", None)
    if not callable(declare):
        return
    try:
        declare(
            "web",
            action,
            risk=risk,
            confirm_required=confirm_required,
            description=description,
        )
    except RuntimeError:
        return


SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to search for."},
        "max_results": {"type": "integer", "description": "How many results to return (1-10)."},
    },
    "required": ["query"],
}
EXTRACT_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Public http(s) URL to read."},
    },
    "required": ["url"],
}


async def activate(context, *, client=None, resolver=None):
    runtime = WebSearchRuntime(context, client=client, resolver=resolver)
    search_description = (
        "Search the public web and return titles, URLs, and snippets. "
        "Use this for news, facts, and lookups."
    )
    extract_description = (
        "Fetch a public http(s) URL and return its title and readable text. "
        "Local, private, and link-local addresses are refused."
    )
    context.register_tool(
        "web__search",
        search_description,
        SEARCH_PARAMETERS,
        runtime.search,
        metadata=_tool_metadata("read", "read_only", False),
    )
    context.register_tool(
        "web__extract",
        extract_description,
        EXTRACT_PARAMETERS,
        runtime.extract,
        metadata=_tool_metadata("read", "read_only", False),
    )
    _declare_policy(context, "search", risk="read_only", confirm_required=False, description=search_description)
    _declare_policy(context, "extract", risk="read_only", confirm_required=False, description=extract_description)
    connector = None
    getter = getattr(getattr(context, "connectors", None), "get", None)
    if callable(getter):
        connector = getter("web")
    runtime.bind(
        connector,
        {
            "web__search": {"description": search_description, "parameters": SEARCH_PARAMETERS},
            "web__extract": {"description": extract_description, "parameters": EXTRACT_PARAMETERS},
        },
    )
    return runtime


async def deactivate(context, runtime):
    if runtime is not None:
        runtime.unbind()
        runtime.close()
    unregister = getattr(context, "unregister_registered_tools", None)
    if callable(unregister):
        unregister()
