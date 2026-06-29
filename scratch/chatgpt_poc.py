import os
from playwright.sync_api import sync_playwright

def main():
    # We use a dedicated, persistent profile directory just for this script.
    # This completely bypasses the lock/crash issues of the default Chrome profile,
    # but since it is persistent, you only need to log in to ChatGPT ONCE.
    user_data_dir = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Playwright_ChatGPT")
    
    with sync_playwright() as p:
        print("Launching Chrome with dedicated Playwright profile...")
        try:
            # Launch local Chrome with the dedicated persistent profile
            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                channel="chrome",
                headless=False,
                args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
                ignore_default_args=["--enable-automation"]
            )
            
            # Use the first open page or create a new one
            page = context.pages[0] if context.pages else context.new_page()
            
            print("Navigating to ChatGPT...")
            page.goto("https://chatgpt.com")
            
            print("="*60)
            print("ACTION REQUIRED: Please look at the Chrome window.")
            print("If you are not logged in, please click 'Log in' and log in now.")
            print("Since this is a dedicated profile, you only have to do this ONCE.")
            print("="*60)
            input("Press Enter in this terminal AFTER you are fully logged in (or to continue as guest)...")
            
            prompts = [
                "Hello from Playwright! Please reply with 'Hello, I am ChatGPT.' and nothing else.",
                "What is 2 + 2? Please reply with only the number."
            ]
            
            for i, prompt in enumerate(prompts):
                print(f"\n--- Sending Prompt {i+1} ---")
                
                chat_input = page.locator("#prompt-textarea")
                chat_input.wait_for(state="visible", timeout=60000)
                
                # Count the number of assistant messages before sending
                assistant_messages = page.locator('[data-message-author-role="assistant"]')
                initial_count = assistant_messages.count()
                
                print("Typing message...")
                chat_input.fill(prompt)
                
                # Send the message by pressing Enter
                chat_input.press("Enter")
                
                print("Waiting for message to be sent...")
                
                # Wait for the assistant's reply block to appear (count increases)
                print("Waiting for ChatGPT to start responding...")
                for _ in range(60): # 30 seconds max (60 * 500ms)
                    if assistant_messages.count() > initial_count:
                        break
                    page.wait_for_timeout(500)
                else:
                    raise Exception("Timeout waiting for ChatGPT to start responding")
                
                # Wait for the generation to finish.
                print("Waiting for response to finish generating...")
                last_message = assistant_messages.nth(-1)
                
                # Wait for the text to stabilize (stop changing for 1.5 seconds)
                previous_text = None
                stable_count = 0
                for _ in range(120): # up to 60 seconds
                    current_text = last_message.inner_text()
                    # Only consider it stable if it's not completely empty
                    if current_text and current_text == previous_text:
                        stable_count += 1
                        if stable_count >= 3: # Stable for 3 checks (1.5 seconds)
                            break
                    else:
                        stable_count = 0
                        previous_text = current_text
                    page.wait_for_timeout(500)
                
                # Extract the text from the last assistant message
                last_response = last_message.inner_text()
                
                print("\n" + "="*40)
                print(f"ChatGPT Reply {i+1}:")
                print("="*40)
                print(last_response)
                print("="*40 + "\n")
                
                # Wait a couple of seconds before sending the next prompt
                page.wait_for_timeout(2000)
            
            # Keep browser open for a few seconds to let you see it before closing
            page.wait_for_timeout(3000)
            context.close()
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
