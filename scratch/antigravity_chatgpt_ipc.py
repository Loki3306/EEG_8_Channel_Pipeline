import os
import time
import sys
from playwright.sync_api import sync_playwright

def chat_with_gpt(page, message):
    chat_input = page.locator("#prompt-textarea")
    chat_input.wait_for(state="visible", timeout=60000)
    
    assistant_messages = page.locator('[data-message-author-role="assistant"]')
    initial_count = assistant_messages.count()
    
    chat_input.fill(message)
    page.wait_for_timeout(500)
    chat_input.press("Enter")
    
    for _ in range(60): 
        if assistant_messages.count() > initial_count:
            break
        page.wait_for_timeout(500)
    else:
        chat_input.press("Enter")
        for _ in range(30):
            if assistant_messages.count() > initial_count:
                break
            page.wait_for_timeout(500)
        else:
            raise Exception("Timeout waiting for ChatGPT to start responding")
        
    last_message = assistant_messages.nth(-1)
    
    previous_text = None
    stable_count = 0
    for _ in range(240): 
        current_text = last_message.inner_text()
        if current_text and current_text == previous_text:
            stable_count += 1
            if stable_count >= 3:
                break
        else:
            stable_count = 0
            previous_text = current_text
        page.wait_for_timeout(500)
        
    return last_message.inner_text()

def main():
    user_data_dir = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Playwright_ChatGPT")
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')
    
    prompt_file = "scratch/prompt.txt"
    response_file = "scratch/response.txt"
    
    # Initialize files
    os.makedirs("scratch", exist_ok=True)
    with open(prompt_file, 'w', encoding='utf-8') as f: f.write("")
    with open(response_file, 'w', encoding='utf-8') as f: f.write("")
    
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
        
        page.locator("#prompt-textarea").wait_for(state="visible", timeout=60000)
        
        print("\n" + "="*60)
        print(f"ChatGPT is ready! Listening for messages in {prompt_file}...")
        print("="*60)
        
        while True:
            # Check prompt file
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    msg = f.read().strip()
            except Exception:
                msg = ""
                
            if msg:
                # Clear the prompt file so we don't process it twice
                with open(prompt_file, 'w', encoding='utf-8') as f: f.write("")
                
                if msg.lower() in ['exit', 'quit']:
                    print("Exiting...")
                    break
                    
                print(f"Antigravity (via file)> {msg}")
                try:
                    reply = chat_with_gpt(page, msg)
                    print(f"\nChatGPT> {reply}\n")
                    
                    # Write reply to response file
                    with open(response_file, 'w', encoding='utf-8') as f:
                        f.write(reply)
                except Exception as e:
                    print(f"Error: {e}")
                    with open(response_file, 'w', encoding='utf-8') as f:
                        f.write(f"ERROR: {e}")
                        
            time.sleep(1) # Poll every second
                
        context.close()

if __name__ == "__main__":
    main()
