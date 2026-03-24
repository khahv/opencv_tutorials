"""
Zalo Web Clicker — mo Zalo ban web (chat.zalo.me) bang Playwright (Real Edge).
Chay: python zalo_web_clicker.py  -> tu dong mo Edge (neu chua co) roi mo Zalo.
Dang nhap luu trong playwright_zalo_data/. Dung Edge that nen khung chat hien day du.
"""
import os
import sys
import time
import argparse
import logging
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zalo_web")

ZALO_CHAT_URL = "https://chat.zalo.me/"
USER_DATA_DIR = os.path.join(SCRIPT_DIR, "playwright_zalo_data")
CDP_PORT = 9222
# Sau khi mo trang, tu dong click vao conversation/ nhom nay (ten hien thi trong danh sach chat).
# Duoc override boi tham so receiver_name khi goi send_zalo_message(...).
DEFAULT_CLICK_AFTER_OPEN = "Nhóm HLSE"
# Neu set: click theo selector nay (tranh loi ten co &nbsp; trong HTML). None = click theo ten (conversation list).
CONV_ITEM_SELECTOR = None
# conversationList: #conversationList .msg-item chua [class*="conv-item-title"] > .truncate (ten nhom/nguoi nhan)
CONV_LIST_ID = "conversationList"
CONV_ITEM_TITLE_CLASS = "conv-item-title"
# Sau khi click nhom, dien text vao o nhap tin nhan (contenteditable #richInput / #input_line_0)
DEFAULT_CHAT_MESSAGE = "Hello"
CHAT_INPUT_SELECTOR = "#richInput"  # contenteditable
# Nut gui tin nhan (title="Gửi", class send-msg-btn)
SEND_BUTTON_SELECTOR = ".send-msg-btn"
# Popover mention @All: nut "Bao cho ca nhom" (title hoac .mention-popover__item)
MENTION_ALL_TITLE = "Báo cho cả nhóm"
# Sleep moi buoc de debug (dat 0 de tat)
DEBUG_STEP_SLEEP = 1.0


def _edge_exe():
    """Duong dan msedge.exe tren Windows."""
    for base in [os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                 os.environ.get("ProgramFiles", "C:\\Program Files")]:
        p = os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")
        if os.path.isfile(p):
            return p
    return None


def _ensure_edge_with_cdp():
    """Thu ket noi CDP. Neu loi thi tu mo Edge voi --remote-debugging-port roi ket noi lai."""
    import urllib.request
    import urllib.error
    url = "http://127.0.0.1:%s/json/version" % CDP_PORT
    try:
        req = urllib.request.urlopen(url, timeout=2)
        req.read()
        return True  # da co Edge dang listen
    except Exception:
        pass
    # Mo Edge bang tay
    exe = _edge_exe()
    if not exe:
        log.error("Khong tim thay Microsoft Edge (msedge.exe).")
        return False
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    log.info("Dang mo Edge (port %s)...", CDP_PORT)
    subprocess.Popen(
        [exe, "--remote-debugging-port=%s" % CDP_PORT, "--user-data-dir=%s" % USER_DATA_DIR],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(25):
        time.sleep(0.4)
        try:
            req = urllib.request.urlopen(url, timeout=1)
            req.read()
            return True
        except Exception:
            continue
    log.error("Edge khong kip bat port %s.", CDP_PORT)
    return False


def connect_real_edge(playwright):
    """Ket noi toi Edge (tu dong mo Edge neu chua co). Tra ve (browser, context, page)."""
    if not _ensure_edge_with_cdp():
        return None, None, None
    cdp_url = "http://127.0.0.1:%s" % CDP_PORT
    browser = playwright.chromium.connect_over_cdp(cdp_url)
    if not browser.contexts:
        browser.close()
        return None, None, None
    context = browser.contexts[0]
    pages = context.pages
    page = None
    for p in pages:
        try:
            if "chat.zalo.me" in (p.url or ""):
                page = p
                break
        except Exception:
            pass
    if page is None and pages:
        page = pages[0]
    if page is None:
        page = context.new_page()
    return browser, context, page


def click_selector(page, selector: str, timeout_ms: int = 10000) -> bool:
    """Click phan tu theo CSS selector. Tra ve True neu thanh cong."""
    try:
        page.locator(selector).first.click(timeout=timeout_ms)
        log.info("Da click: %s", selector)
        return True
    except Exception as e:
        log.warning("Khong click duoc %s: %s", selector, e)
        return False


def click_text(page, text: str, timeout_ms: int = 10000) -> bool:
    """Click phan tu co noi dung text (get_by_text). Tra ve True neu thanh cong."""
    try:
        page.get_by_text(text, exact=False).first.click(timeout=timeout_ms)
        log.info("Da click text: %s", text)
        return True
    except Exception as e:
        log.warning("Khong click duoc text %r: %s", text, e)
        return False


def click_role(page, name: str, role: str = "button", timeout_ms: int = 10000) -> bool:
    """Click phan tu theo role (button, link, ...) va accessible name."""
    try:
        page.get_by_role(role, name=name).first.click(timeout=timeout_ms)
        log.info("Da click role %s: %s", role, name)
        return True
    except Exception as e:
        log.warning("Khong click duoc role %s %r: %s", role, name, e)
        return False


def click_conversation_by_name(page, receiver_name: str, timeout_ms: int = 10000) -> bool:
    """
    Click vao conversation trong danh sach chat (conversationList) theo ten hien thi.
    Cau truc: #conversationList .msg-item chua [class*='conv-item-title'] voi .truncate = ten (co the co &nbsp;).
    Tra ve True neu click thanh cong.
    """
    try:
        # Tim .msg-item co title chua receiver_name (exact=False de khong phu thuoc khoang trang/&nbsp;)
        item = page.locator("#%s .msg-item" % CONV_LIST_ID).filter(
            has=page.locator("[class*='%s']" % CONV_ITEM_TITLE_CLASS).get_by_text(receiver_name, exact=False)
        ).first
        item.click(timeout=timeout_ms)
        log.info("Da click conversation: %s", receiver_name)
        return True
    except Exception as e:
        log.warning("Khong click duoc conversation %r: %s", receiver_name, e)
        return False


CONV_LIST_SELECTOR = ".conv-item-title__name, [class*='conv-item-title']"
CONV_LIST_TIMEOUT_MS = 25000
SEND_MAX_RETRIES = 3
SEND_RETRY_DELAY_SEC = 5


def _send_once(page, message: str, target: str, _log) -> bool:
    """Single attempt to send a Zalo message. Returns True on success."""
    if "chat.zalo.me" not in (page.url or ""):
        page.goto(ZALO_CHAT_URL, wait_until="domcontentloaded", timeout=20000)

    # Wait for conversation list; try networkidle first, fallback to selector.
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    page.wait_for_selector(CONV_LIST_SELECTOR, timeout=CONV_LIST_TIMEOUT_MS)

    time.sleep(0.5)
    if CONV_ITEM_SELECTOR:
        page.locator(CONV_ITEM_SELECTOR).first.click(timeout=5000)
    else:
        click_conversation_by_name(page, target, timeout_ms=5000)
    time.sleep(0.4)
    page.wait_for_selector(CHAT_INPUT_SELECTOR, timeout=8000)
    page.locator(CHAT_INPUT_SELECTOR).click()
    time.sleep(0.15)

    msg = message.strip()
    if msg.lower().startswith("@all "):
        text_after = msg[5:].strip()
        # Type @All slowly so Zalo's mention popover has time to appear
        page.keyboard.type("@All", delay=80)
        time.sleep(0.6)
        mention_ok = False
        try:
            page.get_by_title(MENTION_ALL_TITLE).first.click(timeout=5000)
            mention_ok = True
        except Exception:
            try:
                page.locator(".mention-popover__item").first.click(timeout=3000)
                mention_ok = True
            except Exception:
                pass
        if not mention_ok:
            _log.info("[ZaloWeb] Mention @All popover not found, typing rest as text")
        # Use keyboard.type instead of execCommand — execCommand is deprecated and
        # does not trigger React state updates in modern Chromium, leaving the send
        # button disabled.
        if text_after:
            time.sleep(0.2)
            page.keyboard.type(" " + text_after, delay=30)
    else:
        page.keyboard.type(msg, delay=30)

    time.sleep(0.3)
    # Press Enter as primary send method; fall back to button click if needed.
    # The send button may remain data-disabled when React state is not updated,
    # so keyboard Enter is more reliable.
    try:
        page.keyboard.press("Enter")
    except Exception:
        page.locator(SEND_BUTTON_SELECTOR).first.click(timeout=5000)
    return True


def send_zalo_message(message: str, receiver_name: str = None, logger=None):
    """
    Send Zalo message to the conversation (receiver_name hoac DEFAULT_CLICK_AFTER_OPEN).
    receiver_name: ten hien thi trong danh sach chat (vd. "Nhóm HLSE", "My Documents", "Safira-Chủ Căn hộ-BQT").
    Connect Edge -> open Zalo -> click conversation by name -> type message -> click Send -> disconnect.
    Retries up to SEND_MAX_RETRIES times on failure.
    Tra ve True neu thanh cong, False neu loi. Chay trong thread de tranh block.
    """
    _log = logger or log
    target = (receiver_name or "").strip() or DEFAULT_CLICK_AFTER_OPEN
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _log.warning("[ZaloWeb] Chua cai Playwright, bo qua gui tin.")
        return False

    for attempt in range(1, SEND_MAX_RETRIES + 1):
        try:
            with sync_playwright() as p:
                browser, context, page = connect_real_edge(p)
                if not page:
                    _log.warning("[ZaloWeb] Khong ket noi duoc Edge, bo qua gui tin.")
                    return False
                try:
                    _send_once(page, message, target, _log)
                    _log.info("[ZaloWeb] Sent message to %s: %s", target, message[:50])
                    return True
                finally:
                    try:
                        browser.close()
                    except Exception:
                        pass
        except Exception as e:
            _log.warning("[ZaloWeb] Loi gui tin (attempt %d/%d): %s", attempt, SEND_MAX_RETRIES, e)
            if attempt < SEND_MAX_RETRIES:
                _log.info("[ZaloWeb] Retrying in %ds...", SEND_RETRY_DELAY_SEC)
                time.sleep(SEND_RETRY_DELAY_SEC)

    _log.error("[ZaloWeb] Failed to send message after %d attempts to %s", SEND_MAX_RETRIES, target)
    return False


def main():
    parser = argparse.ArgumentParser(description="Zalo Web — mo chat.zalo.me bang Real Edge")
    parser.add_argument("--click", "-c", metavar="SELECTOR", help="Click 1 lan theo CSS selector roi thoat")
    parser.add_argument("--text", "-t", metavar="TEXT", help="Click 1 lan theo noi dung text roi thoat")
    parser.add_argument("--no-open", action="store_true", help="Khong tu mo trang (chi dung voi --click/--text khi da mo san)")
    parser.add_argument("--dump-html", metavar="FILE", help="Luu toan bo HTML trang vao file de debug (sau khi mo/goto)")
    parser.add_argument("--dump-conv-items", action="store_true", help="In ra cac conv-item de debug selector")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Chua cai Playwright. Chay: pip install playwright")
        sys.exit(1)

    with sync_playwright() as p:
        browser, context, page = connect_real_edge(p)
        if not context or not page:
            log.error("Khong ket noi duoc Edge. Thu dong het cua so Edge roi chay lai.")
            sys.exit(1)
        log.info("Da ket noi Edge (Real).")

        if not args.no_open:
            page.goto(ZALO_CHAT_URL, wait_until="domcontentloaded")
            log.info("Da mo %s", ZALO_CHAT_URL)
            # Cho danh sach hoi thoai load roi click vao "Nhóm HLSE" (conv-item-title)
            try:
                page.wait_for_selector(".conv-item-title__name, [class*='conv-item-title']", timeout=15000)
                time.sleep(0.8)
                if CONV_ITEM_SELECTOR:
                    page.locator(CONV_ITEM_SELECTOR).first.click(timeout=5000)
                    log.info("Da click conversation (theo selector)")
                else:
                    click_conversation_by_name(page, DEFAULT_CLICK_AFTER_OPEN, timeout_ms=5000)
                time.sleep(0.5)
                # Dien text vao o nhap tin nhan (contenteditable)
                page.wait_for_selector(CHAT_INPUT_SELECTOR, timeout=8000)
                page.locator(CHAT_INPUT_SELECTOR).click()
                time.sleep(0.2)
                page.keyboard.type(DEFAULT_CHAT_MESSAGE, delay=50)
                log.info("Da dien tin nhan: %s", DEFAULT_CHAT_MESSAGE)
                time.sleep(0.2)
                page.locator(SEND_BUTTON_SELECTOR).first.click(timeout=3000)
                log.info("Da nhan nut Gui.")
            except Exception as e:
                log.warning("Khong tu dong click/dien/gui duoc: %s", e)

        # Debug: luu HTML hoac in cac element
        if args.dump_html:
            try:
                html = page.content()
                with open(args.dump_html, "w", encoding="utf-8") as f:
                    f.write(html)
                log.info("Da luu HTML vao %s", args.dump_html)
            except Exception as e:
                log.warning("Khong luu duoc HTML: %s", e)
        if args.dump_conv_items:
            try:
                # Lay tat ca element co class chua conv / item / truncate
                sel = "[class*='conv'], [class*='conv-item'], .truncate"
                locs = page.locator(sel).all()
                log.info("Tim thay %d element khop %s", len(locs), sel)
                for i, loc in enumerate(locs[:50]):  # toi da 50
                    try:
                        txt = loc.inner_text(timeout=500)[:80]
                        cls = loc.get_attribute("class") or ""
                        log.info("  [%d] class=%s | text=%r", i, cls[:60], txt)
                    except Exception:
                        pass
            except Exception as e:
                log.warning("Khong dump conv items: %s", e)

        if args.click:
            ok = click_selector(page, args.click)
            try:
                browser.close()
            except Exception:
                pass
            sys.exit(0 if ok else 1)
        if args.text:
            ok = click_text(page, args.text)
            try:
                browser.close()
            except Exception:
                pass
            sys.exit(0 if ok else 1)

        log.info("Edge dang mo %s. Nhan Enter de ngat ket noi (Edge van chay).", ZALO_CHAT_URL)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        try:
            browser.close()
        except Exception:
            pass
        log.info("Da ngat ket noi.")


if __name__ == "__main__":
    main()
