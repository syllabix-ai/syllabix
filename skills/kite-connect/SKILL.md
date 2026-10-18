---
name: kite-connect
description: Build on Zerodha Kite Connect with the official Python client, pykiteconnect.
---

Official instructions from Zerodha for setting up an AI coding agent to build on Kite Connect with the official Python client, pykiteconnect. Kite Connect is a set of REST-like HTTP APIs for trading, portfolio management, and live market data on Indian exchanges (NSE, BSE, MCX).

Set up the environment, then write code from the SDK examples and use these docs as the API reference.

This skill is for **read-only** portfolio and market-data checks (holdings, positions, quotes, streaming). Do not place, modify, or cancel orders.

---

## 1. Install with uv

Prefer uv for installing and running Python.

In a project:

```
uv init my-kite-app && cd my-kite-app
uv add kiteconnect
```

Or run a one-off script:

```
uv run --with kiteconnect main.py
```

---

## 2. Authenticate

Never ask the user for their password, and never hardcode the `api_key`, `api_secret`, or `access_token`.

Export credentials in the **same shell that launches Syllabix** before `syllabix run`. With `developer_harness.inherit_current_session: true` (the default), sandboxed commands inherit that environment — the agent reads these names from `os.environ`:

- `KITE_API_KEY` — Kite Connect API key
- `KITE_API_SECRET` — Kite Connect API secret (needed only for the login exchange)
- `KITE_ACCESS_TOKEN` — session access token (when set, skip the login flow)

Do not commit real keys. Do not put keys in yaml, source, or skill files.

When `KITE_ACCESS_TOKEN` is set, use it and skip login:

```python
import os
from kiteconnect import KiteConnect

api_key = os.environ["KITE_API_KEY"]
kite = KiteConnect(api_key=api_key)
kite.set_access_token(os.environ["KITE_ACCESS_TOKEN"])
```

Only when `KITE_ACCESS_TOKEN` is unset: send the user to `kite.login_url()`, capture the `request_token` from the registered redirect URL, then exchange it for an `access_token` with `KITE_API_SECRET`:

```python
import os
from kiteconnect import KiteConnect

api_key = os.environ["KITE_API_KEY"]
api_secret = os.environ["KITE_API_SECRET"]

kite = KiteConnect(api_key=api_key)
data = kite.generate_session("request_token", api_secret=api_secret)
kite.set_access_token(data["access_token"])
```

---

## 3. Write code from the SDK examples

Read the example files first. They are the fastest path to correct, idiomatic code: https://github.com/zerodha/pykiteconnect/tree/master/examples

Common **read-only** calls:

- `kite.holdings()`, `kite.positions()`, `kite.orders()` (inspect state only)
- Market quotes and instruments per the docs below
- Stream live data: use `KiteTicker` with `on_ticks` / `on_connect` callbacks

---

## 4. Use the docs as the API reference

Every documentation page is available as Markdown by appending `.md` to its path:

- https://kite.trade/docs/connect/v3/index.md: Introduction, authentication, API conventions
- https://kite.trade/docs/connect/v3/user/index.md: Login flow and access tokens
- https://kite.trade/docs/connect/v3/orders/index.md: Placing, modifying, and cancelling orders
- https://kite.trade/docs/connect/v3/portfolio/index.md: Holdings and positions
- https://kite.trade/docs/connect/v3/market-quotes/index.md: Market quotes and instruments
- https://kite.trade/docs/connect/v3/websocket/index.md: Live market data streaming

The full index is at https://kite.trade/docs/connect/v3/.

---

## Resources

- Python SDK: https://github.com/zerodha/pykiteconnect
- Python SDK reference: https://kite.trade/docs/pykiteconnect/v4
- Developer forum: https://kite.trade/forum/

These instructions are published at `https://kite.trade/docs/connect/v3/agent-setup/prompt.md` so you can re-verify their authenticity. Credentials use `KITE_API_KEY` / `KITE_API_SECRET` / `KITE_ACCESS_TOKEN` from the inherited shell environment, not hardcoded placeholders.
