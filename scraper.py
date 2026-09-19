import asyncio
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List

try:
    from playwright.async_api import async_playwright
except ModuleNotFoundError:
    async_playwright = None

class FlightScraper:
    """Async Playwright scraper to extract real-time fare data from Google Flights."""

    def __init__(self, headless: bool = True):
        self.headless = headless

    async def fetch_route_flights(
        self, origin: str, destination:str, departure_date: str
    ) -> List[Dict]:
        """
        Navigates to Google Flights URL and extracts listings.

        :param origin: Airport IATA code (e.g., 'JFK')
        :param destination: Airport IATA code (e.g., 'LAX')
        :param departure_date: Date formatted as 'YYYY-MM-DD'
        """
        if async_playwright is None:
            raise RuntimeError(
                "Playwright is not installed. Run 'python -m pip install playwright' and "
                "'python -m playwright install chromium' before enabling live scraping."
            )

        search_url = (
            f"https://www.google.com/travel/flights?q=Flights%20to%20{destination}"
            f"%20from%20{origin}%20on%20{departure_date}%20one-way&curr=KSh"
        )

        extracted_flights = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            try:
                await page.goto(search_url, wait_until="documentloaded", timeout=30000)
                await asyncio.sleep(random.uniform(2, 4))

                consent_btn = page.locator("button:has-text('Accept all'), button:has-text('I agree')")
                if await consent_btn.count() > 0:
                    await consent_btn.first.click()
                    await page.wait_for_load_state("networkidle")

                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2)

                cards = page.locator("li.pI13d, div.pI13d, [role='listitem']")
                card_count = await cards.count()

                for i in range(min(card_count, 15)):
                    card = cards.nth(i)
                    text_content = await card.inner_text()

                    if not text_content or "$" not in text_content:
                        continue

                    lines = [line.strip() for line in text_content.split("\n") if line.strip()]

                    price_match = re.search(r"\$([0-9,]+)", text_content)
                    if not price_match:
                        continue
                    price = float(price_match.group(1).replace(",", ""))

                    normalized_text = text_content.lower()
                    layover_count = 0
                    if "nonstop" in normalized_text:
                        layover_count = 0
                    elif "stop" in normalized_text:
                        stop_match = re.search(r"(\d+)\s*stop", normalized_text)
                        layover_count = int(stop_match.group(1)) if stop_match else 1

                    airline = lines[0] if lines else "Unknown Airline"

                    extracted_flights.append({
                        "scrape_timestamp": datetime.now().isoformat(),
                        "origin": origin,
                        "destination": destination,
                        "departure_date": departure_date,
                        "airline": airline,
                        "price": price,
                        "layovers": layover_count,
                        "raw_summary": text_content[:100]
                    })
            except Exception as e:
                print(f"[Scraper Error] Failed to extract data: {e}")
            finally:
                await context.close()
                await browser.close()

        return extracted_flights

if __name__ == "__main__":
    # Test execution
    scraper = FlightScraper(headless=True)
    results = asyncio.run(scraper.fetch_route_flights("JFK", "LAX", "2026-10-15"))
    print(f"Scraped {len(results)} flight records.") 
        

