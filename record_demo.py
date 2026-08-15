# Offline developer utility to record demo video of the dashboard (requires playwright; not a runtime production dependency).
"""
record_demo.py
Automated high-quality video recording of the FinPulse dashboard using Playwright.
Produces a 20-30 second demo showing:
- Initial portfolio and heatmap load
- Stress testing slider interaction (15%, 30% expense shock)
- Forecast segment switching
- Early warning alerts with GenAI plan and recommended actions
- Model performance tabs (Classification, Forecasting, Clustering, Statistics)
"""

import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("demo")
OUTPUT_DIR.mkdir(exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def record():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )
        
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(OUTPUT_DIR),
            record_video_size={"width": 1440, "height": 900}
        )
        
        page = context.new_page()
        
        print("Navigating to dashboard...")
        page.goto("http://localhost:5000", wait_until="networkidle")
        
        # Wait for loading overlay to hide
        page.wait_for_selector("#loadingOverlay.hidden", timeout=30000)
        time.sleep(2)
        
        print("1. Showing Portfolio Overview and Heatmap...")
        page.mouse.move(400, 200)
        time.sleep(2)
        
        print("2. Interacting with Stress Slider (Expense Shock)...")
        slider = page.locator("#shockSlider")
        slider.evaluate("el => { el.value = 15; el.dispatchEvent(new Event('input', { bubbles: true })); }")
        time.sleep(2.5)
        
        slider.evaluate("el => { el.value = 30; el.dispatchEvent(new Event('input', { bubbles: true })); }")
        time.sleep(2.5)
        
        slider.evaluate("el => { el.value = 0; el.dispatchEvent(new Event('input', { bubbles: true })); }")
        time.sleep(2)
        
        print("3. Switching forecast segments...")
        page.evaluate("() => document.getElementById('forecastSection').scrollIntoView({ behavior: 'smooth' })")
        time.sleep(1.5)
        
        page.select_option("#segmentSelect", "Gig/Freelance")
        time.sleep(2.5)
        
        page.select_option("#segmentSelect", "Daily Wage")
        time.sleep(2)
        
        print("4. Viewing Alerts & Recommended Actions...")
        page.evaluate("() => document.getElementById('alertsSection').scrollIntoView({ behavior: 'smooth' })")
        time.sleep(2.5)
        
        print("5. Viewing Model Performance Tabs...")
        page.evaluate("() => document.getElementById('modelSection').scrollIntoView({ behavior: 'smooth' })")
        time.sleep(1.5)
        
        # Click Forecasting tab
        page.click(".model-tab[data-tab='forecasting']")
        time.sleep(2.5)
        
        # Click Clustering tab
        page.click(".model-tab[data-tab='clustering']")
        time.sleep(2)
        
        # Click Statistics tab
        page.click(".model-tab[data-tab='statistics']")
        time.sleep(2.5)
        
        # Click Classification tab
        page.click(".model-tab[data-tab='classification']")
        time.sleep(2)
        
        print("Finishing recording...")
        time.sleep(1)
        context.close()
        browser.close()
        print("Recording saved successfully!")

if __name__ == "__main__":
    record()
