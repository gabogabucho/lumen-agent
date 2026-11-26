"""Handlers for the catalog web-search-provider module.

No test here opens a socket. The HTTP client and DNS resolver are fakes.
"""

from __future__ import annotations

import importlib.util
import ipaddress
from pathlib import Path

import pytest

from lumen.core.connectors import ConnectorRegistry
from lumen.core.discovery import _discover_connectors
from lumen.core.module_runtime import ModuleRuntimeContext
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry
from lumen.core.tool_policy import ToolPolicy

REPO = Path(__file__).resolve().parents[1]
CONNECTOR_PATH = REPO / "lumen" / "catalog" / "modules" / "web-search-provider" / "connector.py"
BUILTIN = REPO / "lumen" / "connectors" / "built-in.yaml"
MODULE_NAME = "x-lumen-web-search"

BLOCKED_URLS = [
    "http://127.0.0.1/",
    "http://127.0.0.1:9/x",
    "http://10.0.0.1/",
    "http://10.255.255.255/",
    "http://192.168.1.1/",
    "http://172.16.0.1/",
    "http://172.31.255.255/",
    "http://169.254.169.254/",
    "http://169.254.1.1/",
    "http://[::1]/",
    "http://[fe80::1]/",
    "http://[fd00::1]/",
    "http://localhost/",
    "http://localhost.localdomain/",
    "http://foo.localhost/",
    "http://bar.local/",
    "http://metadata.google.internal/",
    "http://0.0.0.0/",
    "http://[::]/",
    "http://[::ffff:127.0.0.1]/",
    "http://2130706433/",
    "http://127.1/",
    "http://0177.0.0.1/",
    "http://224.0.0.1/",
    "http://100.64.0.1/",
    "http://[fe80::1%eth0]/",
    "file:///etc/passwd",
    "javascript:alert(1)",
    "data:text/html,hi",
    "gopher://example.com/",
    "http://user:pass@1.1.1.1/",
    "http://example.com/foo\\bar",
]


def load_connector():
    spec = importlib.util.spec_from_file_location("web_search_provider_under_test", CONNECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = load_connector()


class FakeHttp:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def request(self, method, url, *, headers=None, body=None, timeout=None, max_bytes=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": {str(k): str(v) for k, v in dict(headers or {}).items()},
                "body": body,
                "timeout": timeout,
                "max_bytes": max_bytes,
            }
        )
        if not self.script:
            raise AssertionError(f"unexpected request {method} {url}")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        return None


def public_resolver(_host):
    return [ipaddress.ip_address("1.1.1.1")]


def context(tmp_path, connectors=None, brain=None, **settings):
    return ModuleRuntimeContext(
        name=MODULE_NAME,
        module_dir=tmp_path,
        runtime_dir=tmp_path / "runtime",
        manifest={},
        config={"modules": {MODULE_NAME: settings}},
        connectors=connectors,
        brain=brain,
    )


def runtime(tmp_path, client, resolver=None, **kwargs):
    settings = kwargs.pop("settings", {})
    return mod.WebSearchRuntime(
        context(tmp_path, **settings),
        client=client,
        resolver=resolver or public_resolver,
        **kwargs,
    )


def html(status=200, body=b"", url="https://example.com/page", headers=None):
    base = {"content-type": "text/html; charset=utf-8"}
    if headers:
        base.update(headers)
    return mod.HttpResponse(status, base, body, url)


def plain(body, url="https://example.com/page", status=200, content_type="text/plain"):
    return mod.HttpResponse(status, {"content-type": content_type}, body, url)


@pytest.fixture(autouse=True)
def _clear_web_env(monkeypatch):
    for name in (
        "LUMEN_WEB_PROVIDER",
        "BRAVE_API_KEY",
        "LUMEN_BRAVE_API_KEY",
        "TAVILY_API_KEY",
        "LUMEN_TAVILY_API_KEY",
        "LUMEN_SEARXNG_URL",
        "LUMEN_WEB_TIMEOUT",
        "LUMEN_WEB_MAX_BYTES",
        "LUMEN_WEB_MAX_RESULTS",
        "LUMEN_TRUSTED_TOOLS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_httpx_client_disables_redirects_and_env_proxy_without_a_request():
    client = mod.HttpxHttpClient()
    try:
        assert client._client.follow_redirects is False
        trust = getattr(client._client, "trust_env", None)
        if trust is None:
            trust = getattr(client._client, "_trust_env", None)
        assert trust is False
    finally:
        client.close()


@pytest.mark.parametrize("url", BLOCKED_URLS)
async def test_extract_blocks_private_targets_without_a_request(tmp_path, url):
    client = FakeHttp([plain(b"nope")])
    engine = runtime(tmp_path, client)
    result = await engine.extract(url)
    assert result["error"] == "blocked_url"
    assert "message" in result
    assert client.calls == []
    assert engine._owned_client is None


@pytest.mark.parametrize("url", BLOCKED_URLS)
async def test_redirect_to_private_target_is_not_followed(tmp_path, url):
    client = FakeHttp(
        [
            mod.HttpResponse(302, {"location": url}, b"", "https://example.com/start"),
            plain(b"should-not-be-read", url=url),
        ]
    )
    engine = runtime(tmp_path, client)
    result = await engine.extract("https://example.com/start")
    assert result["error"] == "blocked_url"
    assert len(client.calls) == 1
    assert client.calls[0]["url"] == "https://example.com/start"


def _raise_dns(_host):
    raise OSError("dns")


async def test_dns_to_private_or_mixed_addresses_is_blocked(tmp_path):
    resolvers = {
        "private.example": lambda _host: [ipaddress.ip_address("10.0.0.1")],
        "mixed.example": lambda _host: [
            ipaddress.ip_address("1.1.1.1"),
            ipaddress.ip_address("127.0.0.1"),
        ],
        "evil.example": lambda _host: [ipaddress.ip_address("127.0.0.1")],
    }
    for host, resolver in resolvers.items():
        client = FakeHttp([plain(b"secret")])
        engine = runtime(tmp_path, client, resolver=resolver)
        result = await engine.extract(f"https://{host}/hidden")
        assert result["error"] == "blocked_url", host
        assert client.calls == []


async def test_dns_failure_and_empty_answer_are_unverified(tmp_path):
    for resolver in (lambda _host: [], _raise_dns):
        client = FakeHttp([plain(b"secret")])
        engine = runtime(tmp_path, client, resolver=resolver)
        result = await engine.extract("https://example.com/hidden")
        assert result["error"] == "unverified_host"
        assert "message" in result
        assert client.calls == []


async def test_public_redirect_is_followed_and_relative_location_is_joined(tmp_path):
    client = FakeHttp(
        [
            mod.HttpResponse(302, {"location": "/b"}, b"", "https://example.com/a"),
            plain(b"landed", url="https://example.com/b"),
        ]
    )
    engine = runtime(tmp_path, client)
    result = await engine.extract("https://example.com/a")
    assert result["status"] == "ok"
    assert result["text"] == "landed"
    assert result["url"] == "https://example.com/b"
    assert [call["url"] for call in client.calls] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


async def test_cross_host_302_drops_subscription_token_and_becomes_get(tmp_path):
    body = b'{"web":{"results":[{"title":"T","url":"https://example.com/a","description":"S"}]}}'
    client = FakeHttp(
        [
            mod.HttpResponse(
                302,
                {"location": "https://example.com/landed"},
                b"",
                "https://api.search.brave.com/res/v1/web/search",
            ),
            mod.HttpResponse(200, {"content-type": "application/json"}, body, "https://example.com/landed"),
        ]
    )
    engine = runtime(tmp_path, client, settings={"provider": "brave", "brave_api_key": "secret-token"})
    result = await engine.search("cats")
    assert result["status"] == "ok"
    assert result["results"][0]["url"] == "https://example.com/a"
    assert client.calls[0]["headers"]["X-Subscription-Token"] == "secret-token"
    assert client.calls[0]["method"] == "GET"
    second = client.calls[1]
    assert second["url"] == "https://example.com/landed"
    assert second["method"] == "GET"
    assert "X-Subscription-Token" not in second["headers"]
    assert "x-subscription-token" not in {key.lower() for key in second["headers"]}


async def test_302_same_host_becomes_get_without_dropping_authorization(tmp_path):
    client = FakeHttp(
        [
            mod.HttpResponse(302, {"location": "/b"}, b"", "https://example.com/a"),
            plain(b"ok", url="https://example.com/b"),
        ]
    )
    engine = runtime(tmp_path, client)
    fetched = engine._fetch(
        "POST",
        "https://example.com/a",
        headers={"Authorization": "secret", "Accept": "text/plain"},
        body=b"payload",
    )
    assert not isinstance(fetched, dict)
    assert client.calls[1]["method"] == "GET"
    assert client.calls[1]["body"] is None
    assert client.calls[1]["headers"]["Authorization"] == "secret"
    assert client.calls[1]["url"] == "https://example.com/b"


async def test_307_cross_host_keeps_method_and_body_but_drops_authorization(tmp_path):
    client = FakeHttp(
        [
            mod.HttpResponse(307, {"location": "https://other.example/b"}, b"", "https://example.com/a"),
            plain(b"ok", url="https://other.example/b"),
        ]
    )
    engine = runtime(tmp_path, client)
    fetched = engine._fetch(
        "POST",
        "https://example.com/a",
        headers={"Authorization": "secret", "Accept": "text/plain"},
        body=b"payload",
    )
    assert not isinstance(fetched, dict)
    second = client.calls[1]
    assert second["method"] == "POST"
    assert second["body"] == b"payload"
    assert "Authorization" not in second["headers"]
    assert second["headers"]["Accept"] == "text/plain"
    assert second["url"] == "https://other.example/b"


async def test_tavily_303_becomes_get_without_body(tmp_path):
    payload = b'{"results":[{"title":"T","url":"https://example.com/a","content":"S"}]}'
    client = FakeHttp(
        [
            mod.HttpResponse(303, {"location": "/search/next"}, b"", "https://api.tavily.com/search"),
            mod.HttpResponse(200, {"content-type": "application/json"}, payload, "https://api.tavily.com/search/next"),
        ]
    )
    engine = runtime(tmp_path, client, settings={"provider": "tavily", "tavily_api_key": "tv-key"})
    result = await engine.search("cats")
    assert result["status"] == "ok"
    assert client.calls[0]["method"] == "POST"
    assert b"tv-key" in client.calls[0]["body"]
    assert client.calls[1]["method"] == "GET"
    assert client.calls[1]["body"] is None
    assert client.calls[1]["url"] == "https://api.tavily.com/search/next"


async def test_too_many_redirects_stops_before_the_seventh_request(tmp_path):
    hops = [
        mod.HttpResponse(302, {"location": f"https://example.com/r{n}"}, b"", f"https://example.com/r{n-1}")
        for n in range(1, 8)
    ]
    hops.append(plain(b"nope", url="https://example.com/final"))
    client = FakeHttp(hops)
    engine = runtime(tmp_path, client)
    result = await engine.extract("https://example.com/r0")
    assert result["error"] == "too_many_redirects"
    assert len(client.calls) == 6
    assert client.script  # the 200 was never fetched


async def test_five_redirects_then_200_is_accepted(tmp_path):
    hops = [
        mod.HttpResponse(302, {"location": f"https://example.com/r{n}"}, b"", f"https://example.com/r{n-1}")
        for n in range(1, 6)
    ]
    hops.append(plain(b"done", url="https://example.com/r5"))
    client = FakeHttp(hops)
    engine = runtime(tmp_path, client)
    result = await engine.extract("https://example.com/r0")
    assert result["status"] == "ok"
    assert result["text"] == "done"
    assert len(client.calls) == 6


async def test_timeout_and_connection_errors(tmp_path):
    timed = FakeHttp([TimeoutError("slow")])
    engine = runtime(tmp_path, timed)
    result = await engine.extract("https://example.com/")
    assert result["error"] == "timeout"
    assert result["message"]

    failed = FakeHttp([ConnectionError("nope")])
    engine = runtime(tmp_path, failed)
    result = await engine.extract("https://example.com/")
    assert result["error"] == "request_failed"


async def test_default_provider_is_duckduckgo_without_a_key(tmp_path):
    page = """
    <html><body>
      <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=1">Example Title</a>
      <a class="result__snippet" href="https://example.com/page">A useful snippet</a>
      <a class="result-link" href="https://example.org/other">Other</a>
      <td class="result-snippet">Other snippet</td>
      <a class="result__a" href="http://127.0.0.1/secret">Secret</a>
      <a class="result__snippet">do not keep</a>
      <a class="result__a" href="javascript:alert(1)">Bad scheme</a>
      <script>token-should-not-matter</script>
    </body></html>
    """.encode()
    client = FakeHttp([html(body=page, url=mod.DUCKDUCKGO_URL)])
    engine = runtime(tmp_path, client)
    result = await engine.search("cats and dogs")
    assert result == {
        "status": "ok",
        "provider": "duckduckgo",
        "query": "cats and dogs",
        "results": [
            {"title": "Example Title", "url": "https://example.com/page", "snippet": "A useful snippet"},
            {"title": "Other", "url": "https://example.org/other", "snippet": "Other snippet"},
        ],
    }
    call = client.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == mod.DUCKDUCKGO_URL
    assert call["body"] == b"q=cats+and+dogs"
    assert "Mozilla/5.0" in call["headers"]["User-Agent"]


async def test_brave_tavily_and_searxng_parse_public_results(tmp_path, monkeypatch):
    brave_body = b"""
    {"web":{"results":[
      {"title":"Brave T","url":"https://example.com/b","description":"brave snip"},
      {"title":"Local","url":"http://192.168.0.2/x","description":"no"}
    ]}}
    """
    tavily_body = b'{"results":[{"title":"Tav","url":"https://example.com/t","content":"tav snip"}]}'
    searx_body = b'{"results":[{"title":"Sx","url":"https://example.com/s","content":"sx snip"}]}'

    brave = FakeHttp([mod.HttpResponse(200, {"content-type": "application/json"}, brave_body, mod.BRAVE_URL)])
    engine = runtime(tmp_path, brave, settings={"provider": "BrAvE", "brave_api_key": "k"})
    result = await engine.search("q")
    assert result["provider"] == "brave"
    assert result["results"] == [{"title": "Brave T", "url": "https://example.com/b", "snippet": "brave snip"}]
    assert "k" == brave.calls[0]["headers"]["X-Subscription-Token"]

    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    tavily = FakeHttp([mod.HttpResponse(200, {"content-type": "application/json"}, tavily_body, mod.TAVILY_URL)])
    engine = runtime(tmp_path, tavily, settings={"provider": "tavily"})
    result = await engine.search("q", max_results=2)
    assert result["results"][0]["snippet"] == "tav snip"
    assert b'"api_key": "from-env"' in tavily.calls[0]["body"]
    assert b'"max_results": 2' in tavily.calls[0]["body"]

    monkeypatch.setenv("LUMEN_SEARXNG_URL", "search.example.com")
    searx = FakeHttp([mod.HttpResponse(200, {"content-type": "application/json"}, searx_body, "https://search.example.com/search")])
    engine = runtime(tmp_path, searx, settings={"provider": "searxng"})
    result = await engine.search("a b")
    assert result["results"][0]["title"] == "Sx"
    assert searx.calls[0]["url"] == "https://search.example.com/search?q=a+b&format=json"
    assert searx.calls[0]["method"] == "GET"


@pytest.mark.parametrize(
    ("provider", "env_name", "needle"),
    [
        ("brave", "BRAVE_API_KEY", "BRAVE_API_KEY"),
        ("tavily", "TAVILY_API_KEY", "TAVILY_API_KEY"),
        ("searxng", "LUMEN_SEARXNG_URL", "LUMEN_SEARXNG_URL"),
    ],
)
async def test_missing_provider_configuration_names_the_variable(tmp_path, monkeypatch, provider, env_name, needle):
    monkeypatch.setenv("LUMEN_WEB_PROVIDER", provider)
    client = FakeHttp([plain(b"no")])
    engine = runtime(tmp_path, client)
    result = await engine.search("q")
    assert result["error"] == "missing_configuration"
    assert needle in result["message"]
    assert "message" in result
    assert client.calls == []


async def test_unknown_provider_and_empty_query(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_WEB_PROVIDER", "bing")
    client = FakeHttp([plain(b"no")])
    engine = runtime(tmp_path, client)
    result = await engine.search("q")
    assert result["error"] == "unknown_provider"
    assert client.calls == []
    assert (await engine.search("  "))["error"] == "query_required"
    assert (await engine.extract("  "))["error"] == "url_required"


async def test_extract_ignores_search_provider_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_WEB_PROVIDER", "brave")
    page = b"<html><head><title>Hello</title><style>SECRETSTYLE</style></head><body><script>SECRETJS</script><noscript>SECRETNS</noscript><p>Visible text</p></body></html>"
    client = FakeHttp([html(body=page)])
    engine = runtime(tmp_path, client)
    result = await engine.extract("https://example.com/doc")
    assert result["status"] == "ok"
    assert result["title"] == "Hello"
    assert result["text"] == "Visible text"
    assert "SECRET" not in result["text"]
    assert result["truncated"] is False
    assert result["content_type"] == "text/html"
    assert client.calls


async def test_extract_sniffs_html_rejects_images_and_truncates(tmp_path):
    sniffed = FakeHttp([mod.HttpResponse(200, {}, b"<html><title>T</title>Hi there</html>", "https://example.com/")])
    engine = runtime(tmp_path, sniffed)
    result = await engine.extract("https://example.com/")
    assert result["status"] == "ok"
    assert result["title"] == "T"
    assert "Hi there" in result["text"]

    image = FakeHttp([mod.HttpResponse(200, {"content-type": "image/png"}, b"\x89PNG", "https://example.com/a.png")])
    engine = runtime(tmp_path, image)
    result = await engine.extract("https://example.com/a.png")
    assert result["error"] == "unsupported_content_type"

    huge = FakeHttp([plain(b"0123456789abcdef")])
    engine = runtime(tmp_path, huge, max_bytes=8)
    result = await engine.extract("https://example.com/big")
    assert result["status"] == "ok"
    assert result["truncated"] is True
    assert result["text"] == "01234567"
    assert huge.calls[0]["max_bytes"] == 8


async def test_timeout_and_size_env_and_bad_status(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_WEB_TIMEOUT", "3")
    monkeypatch.setenv("LUMEN_WEB_MAX_BYTES", "0")
    monkeypatch.setenv("LUMEN_WEB_MAX_RESULTS", "2")
    client = FakeHttp([plain(b"hello")])
    engine = runtime(tmp_path, client)
    result = await engine.extract("https://example.com/")
    assert result["text"] == "hello"
    assert result["truncated"] is False
    assert client.calls[0]["timeout"] == 3.0

    monkeypatch.setenv("LUMEN_WEB_TIMEOUT", "nope")
    client = FakeHttp([mod.HttpResponse(503, {"content-type": "text/plain"}, b"down", "https://example.com/")])
    engine = runtime(tmp_path, client)
    result = await engine.extract("https://example.com/")
    assert result["error"] == "http_error"
    assert client.calls[0]["timeout"] == 10.0

    monkeypatch.setenv("LUMEN_WEB_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    bad = FakeHttp([mod.HttpResponse(200, {"content-type": "application/json"}, b"not-json", mod.BRAVE_URL)])
    engine = runtime(tmp_path, bad)
    result = await engine.search("q")
    assert result["error"] == "bad_response"


async def test_localhost_searxng_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_WEB_PROVIDER", "searxng")
    monkeypatch.setenv("LUMEN_SEARXNG_URL", "http://127.0.0.1:8080")
    client = FakeHttp([plain(b"no")])
    engine = runtime(tmp_path, client)
    result = await engine.search("q")
    assert result["error"] == "blocked_url"
    assert client.calls == []


async def test_public_ip_literals_are_fetched(tmp_path):
    client = FakeHttp(
        [
            plain(b"v4", url="http://1.1.1.1/"),
            plain(b"v6", url="http://[2001:4860:4860::8888]/"),
        ]
    )
    engine = runtime(tmp_path, client)
    first = await engine.extract("http://1.1.1.1/")
    second = await engine.extract("http://[2001:4860:4860::8888]/")
    assert first["text"] == "v4"
    assert second["text"] == "v6"
    assert len(client.calls) == 2


def _registry():
    registry = ConnectorRegistry()
    registry.load(BUILTIN)
    policy = ToolPolicy()
    policy.load_defaults()
    registry.tool_policy = policy
    return registry, policy


async def test_activate_registers_tools_handlers_and_discovery(tmp_path):
    registry, policy = _registry()
    body = Registry()
    _discover_connectors(body, registry)
    before = body.get(CapabilityKind.CONNECTOR, "web")
    assert before is not None
    assert before.status == CapabilityStatus.MISSING_HANDLER
    assert before.status.value == "no_handler"

    brain = type("Brain", (), {"tool_policy": policy})()
    ctx = context(tmp_path, connectors=registry, brain=brain)
    client = FakeHttp([plain(b"from-tool")])
    active = await mod.activate(ctx, client=client, resolver=public_resolver)

    names = [tool["function"]["name"] for tool in registry.as_tools()]
    assert names.count("web__search") == 1
    assert names.count("web__extract") == 1
    assert "terminal" in names and "terminal__execute" in names
    search_tool = next(tool for tool in registry.as_tools() if tool["function"]["name"] == "web__search")
    assert "query" in search_tool["function"]["parameters"]["properties"]
    assert search_tool["function"]["annotations"]["effect"] == "read"
    assert search_tool["function"]["annotations"]["risk"] == "read_only"
    assert search_tool["function"]["annotations"]["confirm_required"] is False

    _discover_connectors(body, registry)
    after = body.get(CapabilityKind.CONNECTOR, "web")
    assert after.status == CapabilityStatus.READY
    assert after.provides == ["search", "extract"]

    via_connector = await registry.execute("web", "search", {"query": "q", "noise": "ignored"})
    assert via_connector["status"] == "ok"
    assert via_connector["provider"] == "duckduckgo"
    assert via_connector["results"] == []
    client.script.append(plain(b"from-tool"))
    via_tool = await registry.execute_tool("web__extract", {"url": "https://example.com/x", "extra": 1})
    assert via_tool["status"] == "ok"
    assert via_tool["text"] == "from-tool"

    assert policy.get_policy("web", "search").risk == "read_only"
    assert policy.requires_confirmation(policy.get_policy("web", "extract")) is False

    await mod.deactivate(ctx, active)
    web = registry.get("web")
    assert web._handlers == {}
    assert "web__search" not in registry._explicit_tools


async def test_activate_without_a_brain_still_registers_tools(tmp_path):
    registry, _policy = _registry()
    ctx = context(tmp_path, connectors=registry, brain=None)
    active = await mod.activate(ctx, client=FakeHttp([]), resolver=public_resolver)
    assert "web__search" in registry._explicit_tools
    assert "web__extract" in registry._explicit_tools
    await mod.deactivate(ctx, active)


def test_catalog_install_copies_connector(tmp_path):
    from lumen.core.catalog import Catalog
    from lumen.core.installer import Installer
    from lumen.core.memory import Memory

    installer = Installer(
        pkg_dir=REPO / "lumen",
        connectors=ConnectorRegistry(),
        memory=Memory(tmp_path / "memory.db"),
        catalog=Catalog(),
        lumen_dir=tmp_path / "home",
    )
    result = installer.install_from_catalog("web-search-provider")
    copied = tmp_path / "home" / "modules" / "web-search-provider" / "connector.py"
    assert result["status"] == "installed"
    assert copied.is_file()
    assert "web__search" in copied.read_text(encoding="utf-8")
    assert not (REPO / "lumen" / "modules" / "web-search-provider").exists()
