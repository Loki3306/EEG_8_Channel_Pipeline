import os
import sys
from playwright.sync_api import sync_playwright

def chat_with_gpt(page, message):
    chat_input = page.locator("#prompt-textarea")
    chat_input.wait_for(state="visible", timeout=60000)
    
    assistant_messages = page.locator('[data-message-author-role="assistant"]')
    initial_count = assistant_messages.count()
    
    chat_input.fill(message)
    page.wait_for_timeout(500) # Give UI a moment to register text
    chat_input.press("Enter")
    
    # Wait for the assistant's reply block to appear
    for _ in range(60): # 30 seconds max
        if assistant_messages.count() > initial_count:
            break
        page.wait_for_timeout(500)
    else:
        # Failsafe: Try pressing Enter one more time
        chat_input.press("Enter")
        for _ in range(30):
            if assistant_messages.count() > initial_count:
                break
            page.wait_for_timeout(500)
        else:
            raise Exception("Timeout waiting for ChatGPT to start responding")
        
    last_message = assistant_messages.nth(-1)
    
    # Text stability check
    previous_text = None
    stable_count = 0
    for _ in range(240): # up to 120 seconds
        current_text = last_message.inner_text()
        if current_text and current_text == previous_text:
            stable_count += 1
            if stable_count >= 3: # Stable for 1.5 seconds
                break
        else:
            stable_count = 0
            previous_text = current_text
        page.wait_for_timeout(500)
        
    return last_message.inner_text()

def main():
    user_data_dir = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Playwright_ChatGPT")
    
    # Force stdout to be unbuffered and utf-8 encoded so emojis don't crash on Windows
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
    
    print("Initializing Playwright...")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            channel="chrome",
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"]
        )
        
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://chatgpt.com")
        
        # Wait for chat input to be visible before accepting inputs
        page.locator("#prompt-textarea").wait_for(state="visible", timeout=60000)
        
        print("\n" + "="*60)
        print("ChatGPT is ready!")
        print("="*60)
        print("READY_FOR_INPUT")
        
        while True:
            try:
                msg = input()
            except EOFError:
                break
                
            if msg.strip().lower() in ['exit', 'quit']:
                break
                
            if not msg.strip():
                continue
                
            print(f"Antigravity> {msg}")
            try:
                reply = chat_with_gpt(page, msg)
                print(f"\nChatGPT> {reply}\n")
            except Exception as e:
                print(f"\nError: {e}")
            
            print("READY_FOR_INPUT")
                
        context.close()

if __name__ == "__main__":
    main()
