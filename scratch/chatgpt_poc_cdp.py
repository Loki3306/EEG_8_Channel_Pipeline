import os
import subprocess
import time
from playwright.sync_api import sync_playwright

def main():
    print("Ensuring Chrome is completely closed...")
    os.system("taskkill /f /im chrome.exe >nul 2>&1")
    time.sleep(1)
    
    # Path to default Chrome profile
    user_data_dir = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data")
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    
    print("Launching Chrome manually with remote debugging enabled...")
    # Launch Chrome directly with only the remote debugging port and profile.
    # This avoids Playwright's automated flags which often crash Chrome when loaded with a real user profile.
    subprocess.Popen([
        chrome_path,
        f"--remote-debugging-port=9222",
        f"--user-data-dir={user_data_dir}",
        "--start-maximized"
    ])
    
    # Wait a bit for Chrome to start up and open the port
    time.sleep(3)
    
    print("Connecting Playwright to Chrome via CDP...")
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            context = browser.contexts[0]
            
            # Use existing tab if open, otherwise open a new one
            page = context.pages[0] if context.pages else context.new_page()
            
            print("Navigating to ChatGPT...")
            page.goto("https://chatgpt.com")
            
            print("Waiting for page input...")
            chat_input = page.locator("#prompt-textarea")
            chat_input.wait_for(state="visible", timeout=15000)
            
            print("Typing message...")
            chat_input.fill("Hello from Playwright")
            
            # Send message
            chat_input.press("Enter")
            
            # Wait for our message to appear in the chat history
            print("Waiting for message to be sent...")
            page.locator("text=Hello from Playwright").first.wait_for(state="visible", timeout=10000)
            
            print("Message sent successfully.")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
