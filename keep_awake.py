from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import sys

# ⚠️ REPLACE THIS WITH YOUR ACTUAL STREAMLIT APP URL
url = "https://your-app-name.streamlit.app"

def wake_up_app():
    print(f"Attempting to wake up: {url}")
    
    # Configure headless Chrome (no UI)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    try:
        # Launch browser and visit the app
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        
        # Wait 10 seconds to ensure the WebSocket connection establishes
        time.sleep(10)
        
        print("Successfully pinged the application!")
        driver.quit()
        
    except Exception as e:
        print(f"Failed to ping the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    wake_up_app()
