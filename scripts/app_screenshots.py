"""Capture report screenshots of the four running workspaces (requires `docker compose up`).

    uv run --with playwright python scripts/app_screenshots.py [--url http://localhost:3000]

Uses the locally installed Microsoft Edge through Playwright (no browser download).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"


def run_restoration(page, url, tab, corruption, severity, out, sample_index=0):
    page.goto(f"{url}/#{tab}")
    page.wait_for_selector("main")
    page.locator("aside button:has(img)").nth(sample_index).click()
    page.get_by_role("button", name=corruption, exact=True).click()
    page.get_by_role("button", name=severity, exact=True).click()
    page.get_by_role("button", name="Run restoration").click()
    page.get_by_text("Inference time").wait_for(timeout=30000)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(800)  # let the bar animations finish
    page.screenshot(path=str(out), full_page=True)


def main(url: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 950}, device_scale_factor=1)
        run_restoration(page, url, "universal", "Salt & pepper", "High", OUT / "app_universal.png", 0)
        run_restoration(page, url, "hard", "Occlusion", "Medium", OUT / "app_hard.png", 3)
        run_restoration(page, url, "soft", "Gaussian blur", "Medium", OUT / "app_soft.png", 5)

        page.goto(f"{url}/#sketch")
        page.wait_for_selector("main")
        page.locator("aside button:has(img)").nth(4).click()
        page.get_by_role("button", name="Style 2", exact=True).click()
        page.get_by_role("button", name="Generate sketch").click()
        page.get_by_text("Download sketch").wait_for(timeout=30000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "app_sketch.png"), full_page=True)

        page.goto(f"{url}/#overview")
        page.get_by_text("System information").wait_for(timeout=15000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "app_overview.png"), full_page=True)
        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3000")
    main(ap.parse_args().url)
