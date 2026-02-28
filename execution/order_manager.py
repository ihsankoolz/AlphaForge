"""
execution/order_manager.py
──────────────────────────
Translates target portfolio weights from the risk layer into actual
Alpaca paper trading orders.

Core concept — TARGET vs CURRENT:
  The optimizer outputs target weights (what we WANT to hold).
  The order manager reads current positions (what we ACTUALLY hold).
  It only trades the DELTA between the two.

  Example:
    Target:  AAPL 12%, MSFT 8%, NVDA 0%
    Current: AAPL 10%, MSFT 8%, NVDA 5%
    Action:  Buy 2% AAPL, Hold MSFT, Sell all NVDA

  This minimises transaction costs and avoids unnecessary turnover.

Safeguards (hard limits that cannot be overridden):
  1. Max position size: 15% of portfolio (inherits from risk layer)
  2. Daily loss limit: halt all trading if portfolio down >3% today
  3. Market hours: only place orders 9:30–15:55 ET
  4. Min order value: skip orders below $50 (Alpaca minimum)
  5. Dry run mode: log orders without actually placing them
"""

import os
import sys
import logging
import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from dotenv import load_dotenv

load_dotenv()

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()   # also print to console
    ]
)
log = logging.getLogger(__name__)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

MAX_POSITION_PCT  = 0.15   # hard cap: no single stock above 15% of portfolio
DAILY_LOSS_LIMIT  = 0.03   # halt trading if portfolio down >3% today
MIN_ORDER_VALUE   = 50     # skip orders below $50 (too small to be worth cost)
MARKET_OPEN       = time(9, 30)    # ET
MARKET_CLOSE_SOFT = time(15, 55)   # stop placing NEW orders 5 min before close
ET                = ZoneInfo("America/New_York")


# ── 1. CLIENT ─────────────────────────────────────────────────────────────────

def get_client(paper: bool = True) -> TradingClient:
    """
    Connect to Alpaca. Paper=True uses paper trading account (fake money).
    Always use paper=True until you're very confident in the system.

    Reads API keys from .env file:
        ALPACA_API_KEY=...
        ALPACA_SECRET_KEY=...
    """
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file"
        )

    client = TradingClient(api_key, secret_key, paper=paper)
    log.info(f"Connected to Alpaca ({'PAPER' if paper else 'LIVE'}) account")
    return client


# ── 2. ACCOUNT & POSITION SNAPSHOT ───────────────────────────────────────────

def get_account_snapshot(client: TradingClient) -> dict:
    """
    Pull current account state from Alpaca:
      - portfolio_value: total account value (cash + positions)
      - cash: uninvested cash
      - positions: dict of symbol → current market value

    This is the "current" side of the target vs current comparison.
    """
    account   = client.get_account()
    portfolio_value = float(account.portfolio_value)
    cash            = float(account.cash)

    # Get all open positions
    positions_raw = client.get_all_positions()
    positions = {}
    for pos in positions_raw:
        positions[pos.symbol] = {
            "qty":          float(pos.qty),
            "market_value": float(pos.market_value),
            "weight":       float(pos.market_value) / portfolio_value
                            if portfolio_value > 0 else 0.0,
            "avg_cost":     float(pos.avg_entry_price),
            "unrealized_pl": float(pos.unrealized_pl)
        }

    log.info(f"Account: ${portfolio_value:,.2f} total | "
             f"${cash:,.2f} cash | {len(positions)} positions")

    return {
        "portfolio_value": portfolio_value,
        "cash":            cash,
        "positions":       positions
    }


# ── 3. DAILY LOSS CHECK ───────────────────────────────────────────────────────

def check_daily_loss_limit(client: TradingClient,
                           portfolio_value: float) -> bool:
    """
    Compare today's portfolio value against yesterday's close.
    If we're down more than DAILY_LOSS_LIMIT (3%), halt all trading.

    This is a critical safeguard — without it, a bug in the signal
    generation could cause the system to keep trading into a loss spiral.

    Returns True if safe to trade, False if limit breached.
    """
    account = client.get_account()
    last_equity = float(account.last_equity)   # yesterday's closing value

    if last_equity <= 0:
        log.warning("Could not retrieve last equity — skipping loss check")
        return True

    daily_return = (portfolio_value - last_equity) / last_equity

    if daily_return < -DAILY_LOSS_LIMIT:
        log.error(
            f"DAILY LOSS LIMIT BREACHED: {daily_return:.2%} loss today "
            f"(limit: -{DAILY_LOSS_LIMIT:.0%}). Halting all trading."
        )
        return False

    log.info(f"Daily P&L check: {daily_return:+.2%} — within limits")
    return True


# ── 4. MARKET HOURS CHECK ────────────────────────────────────────────────────

def is_market_open(client: TradingClient) -> bool:
    """
    Check if the US market is currently open via Alpaca's clock API.
    Also checks our soft close time (15:55) to avoid end-of-day chaos.
    """
    clock = client.get_clock()

    if not clock.is_open:
        log.info("Market is closed — no orders will be placed")
        return False

    now_et = datetime.now(ET).time()
    if now_et >= MARKET_CLOSE_SOFT:
        log.info(f"Within 5 minutes of market close — skipping new orders")
        return False

    log.info(f"Market is open — current ET time: {now_et.strftime('%H:%M')}")
    return True


# ── 5. COMPUTE ORDER DELTAS ───────────────────────────────────────────────────

def compute_order_deltas(target_weights: pd.Series,
                         current_snapshot: dict,
                         min_weight_change: float = 0.01) -> list:
    """
    Compare target weights vs current holdings to determine what to buy/sell.

    Logic:
      For each symbol in target OR current:
        delta_weight = target_weight - current_weight
        if delta > min_weight_change  → BUY
        if delta < -min_weight_change → SELL
        otherwise                     → HOLD (avoid unnecessary trading)

    The min_weight_change threshold (default 1%) prevents over-trading.
    If the optimizer nudges a position from 10% to 10.4%, don't bother —
    the transaction cost would exceed the benefit.

    Returns list of order dicts with: symbol, side, dollar_amount
    """
    portfolio_value = current_snapshot["portfolio_value"]
    current_weights = {
        sym: info["weight"]
        for sym, info in current_snapshot["positions"].items()
    }

    orders = []
    all_symbols = set(target_weights.index) | set(current_weights.keys())

    for symbol in all_symbols:
        target_w  = target_weights.get(symbol, 0.0)
        current_w = current_weights.get(symbol, 0.0)
        delta_w   = target_w - current_w

        # Apply hard position cap to target
        if target_w > MAX_POSITION_PCT:
            log.warning(f"{symbol}: target weight {target_w:.1%} exceeds "
                        f"cap {MAX_POSITION_PCT:.0%} — clamping")
            target_w = MAX_POSITION_PCT
            delta_w  = target_w - current_w

        # Skip tiny changes — not worth the transaction cost
        if abs(delta_w) < min_weight_change:
            continue

        dollar_amount = abs(delta_w) * portfolio_value
        if dollar_amount < MIN_ORDER_VALUE:
            log.info(f"{symbol}: order ${dollar_amount:.0f} below minimum "
                     f"${MIN_ORDER_VALUE} — skipping")
            continue

        side = OrderSide.BUY if delta_w > 0 else OrderSide.SELL
        orders.append({
            "symbol":        symbol,
            "side":          side,
            "dollar_amount": dollar_amount,
            "delta_weight":  delta_w,
            "target_weight": target_w,
            "current_weight": current_w
        })

    # IMPORTANT: always process SELLS before BUYS
    # Selling first frees up cash so buys don't fail due to insufficient funds
    orders.sort(key=lambda x: 0 if x["side"] == OrderSide.SELL else 1)

    log.info(f"Order deltas computed: "
             f"{sum(1 for o in orders if o['side'] == OrderSide.SELL)} sells, "
             f"{sum(1 for o in orders if o['side'] == OrderSide.BUY)} buys")

    return orders


# ── 6. PLACE ORDERS ──────────────────────────────────────────────────────────

def place_orders(client: TradingClient,
                 orders: list,
                 dry_run: bool = True) -> list:
    """
    Submit orders to Alpaca. Uses notional (dollar) orders rather than
    share-quantity orders — easier to work with percentages of portfolio.

    dry_run=True: logs what WOULD happen without placing real orders.
                  Always start with dry_run=True to verify the logic.
    dry_run=False: actually places orders. Only enable when confident.

    We use Market orders with TimeInForce=DAY:
      - Market: executes immediately at current price (vs Limit which waits)
      - DAY: order expires at end of day if not filled (clean slate tomorrow)

    Returns list of successfully placed order IDs.
    """
    placed = []

    for order in orders:
        symbol        = order["symbol"]
        side          = order["side"]
        dollar_amount = round(order["dollar_amount"], 2)

        log.info(
            f"{'[DRY RUN] ' if dry_run else ''}ORDER: {side.value.upper()} "
            f"${dollar_amount:,.2f} of {symbol} | "
            f"weight {order['current_weight']:.1%} → {order['target_weight']:.1%} "
            f"(Δ {order['delta_weight']:+.1%})"
        )

        if dry_run:
            placed.append(f"DRY_RUN_{symbol}")
            continue

        try:
            req = MarketOrderRequest(
                symbol       = symbol,
                notional     = dollar_amount,   # dollar amount, not shares
                side         = side,
                time_in_force= TimeInForce.DAY
            )
            result = client.submit_order(req)
            placed.append(result.id)
            log.info(f"  → Order submitted: {result.id}")

        except Exception as e:
            log.error(f"  → Order FAILED for {symbol}: {e}")

    return placed


# ── 7. EXECUTION SUMMARY ─────────────────────────────────────────────────────

def log_execution_summary(target_weights: pd.Series,
                          current_snapshot: dict,
                          orders: list,
                          placed_ids: list):
    """
    Print a clean summary of what the system did today.
    This goes to both the console and the daily log file.
    """
    portfolio_value = current_snapshot["portfolio_value"]

    log.info("\n" + "=" * 60)
    log.info("  DAILY EXECUTION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Portfolio value: ${portfolio_value:,.2f}")
    log.info(f"  Target positions: {len(target_weights)}")
    log.info(f"  Orders placed:    {len(placed_ids)}")
    log.info("")
    log.info(f"  {'Symbol':<8} {'Current':>9} {'Target':>9} {'Action':>12}")
    log.info("  " + "-" * 42)

    all_symbols = sorted(set(target_weights.index) |
                         set(current_snapshot["positions"].keys()))
    for sym in all_symbols:
        current_w = current_snapshot["positions"].get(sym, {}).get("weight", 0.0)
        target_w  = target_weights.get(sym, 0.0)
        delta     = target_w - current_w

        if abs(delta) < 0.001:
            action = "HOLD"
        elif delta > 0:
            action = f"BUY  {delta:+.1%}"
        else:
            action = f"SELL {delta:+.1%}"

        log.info(f"  {sym:<8} {current_w:>8.1%} {target_w:>8.1%} {action:>12}")

    log.info("=" * 60 + "\n")


# ── 8. MAIN ENTRY POINT ───────────────────────────────────────────────────────

def execute_portfolio(target_weights: pd.Series,
                      paper: bool = True,
                      dry_run: bool = True) -> bool:
    """
    Full execution pipeline for one trading day.

    Called by run_daily.py after the risk layer produces target_weights.

    Args:
        target_weights: Series of symbol → weight from portfolio_optimiser
        paper:          True = paper account, False = live (never use False yet)
        dry_run:        True = log only, False = actually place orders

    Returns:
        True if execution completed successfully, False if halted by safeguard
    """
    log.info(f"\n{'='*60}")
    log.info(f"  AlphaForge Execution — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"  Mode: {'PAPER' if paper else 'LIVE'} | "
             f"{'DRY RUN' if dry_run else 'LIVE ORDERS'}")
    log.info(f"{'='*60}")

    try:
        # Step 1: Connect
        client = get_client(paper=paper)

        # Step 2: Check market hours
        if not is_market_open(client):
            return False

        # Step 3: Get current state
        snapshot = get_account_snapshot(client)

        # Step 4: Daily loss limit check
        if not check_daily_loss_limit(client, snapshot["portfolio_value"]):
            return False

        # Step 5: Compute what needs to change
        orders = compute_order_deltas(target_weights, snapshot)

        if not orders:
            log.info("No rebalancing needed — portfolio already at target weights")
            return True

        # Step 6: Place orders
        placed = place_orders(client, orders, dry_run=dry_run)

        # Step 7: Log summary
        log_execution_summary(target_weights, snapshot, orders, placed)

        return True

    except Exception as e:
        log.error(f"Execution pipeline failed: {e}", exc_info=True)
        return False


# ── 9. MAIN — TEST IN DRY RUN MODE ───────────────────────────────────────────

if __name__ == "__main__":
    log.info("Testing order manager in DRY RUN mode...")
    log.info("This will connect to your Alpaca paper account and show")
    log.info("what orders WOULD be placed — nothing will actually execute.\n")

    # Simulate target weights from the risk layer
    # In production these come from portfolio_optimiser.py
    test_weights = pd.Series({
        "AAPL": 0.12,
        "MSFT": 0.10,
        "GOOGL": 0.09,
        "NVDA": 0.11,
        "JPM":  0.08,
    })

    log.info(f"Simulated target weights:")
    for sym, w in test_weights.items():
        log.info(f"  {sym}: {w:.1%}")
    log.info("")

    success = execute_portfolio(
        target_weights = test_weights,
        paper          = True,    # always paper
        dry_run        = True     # always dry run when testing
    )

    if success:
        log.info("Dry run completed successfully.")
        log.info("When ready to paper trade for real, set dry_run=False in run_daily.py")
    else:
        log.info("Dry run halted — check logs above for reason")