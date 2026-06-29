import os
from playwright.sync_api import sync_playwright

def chat_with_tab(page, message):
    chat_input = page.locator("#prompt-textarea")
    chat_input.wait_for(state="visible", timeout=60000)
    
    assistant_messages = page.locator('[data-message-author-role="assistant"]')
    initial_count = assistant_messages.count()
    
    chat_input.fill(message)
    chat_input.press("Enter")
    
    # Wait for the assistant's reply block to appear
    for _ in range(60): # 30 seconds max
        if assistant_messages.count() > initial_count:
            break
        page.wait_for_timeout(500)
    else:
        raise Exception("Timeout waiting for ChatGPT to start responding")
        
    last_message = assistant_messages.nth(-1)
    
    # Text stability check
    previous_text = None
    stable_count = 0
    for _ in range(120): # up to 60 seconds
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
    
    with sync_playwright() as p:
        print("Launching Chrome...")
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            channel="chrome",
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"]
        )
        
        # Setup Tab 1 (GPT-A)
        page1 = context.pages[0] if context.pages else context.new_page()
        print("Opening Tab 1 (Pirate)...")
        page1.goto("https://chatgpt.com")
        
        # Setup Tab 2 (GPT-B)
        print("Opening Tab 2 (Robot)...")
        page2 = context.new_page()
        page2.goto("https://chatgpt.com")
        
        print("\n" + "="*60)
        input("Press Enter once BOTH tabs are fully loaded and ready...")
        print("="*60)
        
        print("Initializing personalities...")
        # Give them their initial prompts
        chat_with_tab(page1, "You are a rowdy Pirate. Answer every message like a pirate. Keep your responses to exactly one short sentence. Acknowledge this with 'Arrr'.")
        chat_with_tab(page2, "You are a highly logical, cold Robot. Answer every message like a robot. Keep your responses to exactly one short sentence. Acknowledge this with 'Beep boop'.")
        
        # The seed message to start the conversation
        current_message = "Hello! I am looking for buried treasure."
        
        print("\n=== THE CONVERSATION BEGINS ===")
        print(f"Seed Message: {current_message}\n")
        
        # Have them talk to each other for 3 rounds!
        for round_num in range(1, 4):
            print(f"--- Round {round_num} ---")
            
            # Send message to Tab 1 (Pirate)
            print("Pirate is typing...")
            # Bring tab 1 to front so you can watch it type
            page1.bring_to_front()
            reply1 = chat_with_tab(page1, current_message)
            print(f"🏴‍☠️ Pirate: {reply1}\n")
            
            # Use Pirate's reply as the prompt for Tab 2 (Robot)
            print("Robot is typing...")
            # Bring tab 2 to front
            page2.bring_to_front()
            reply2 = chat_with_tab(page2, reply1)
            print(f"🤖 Robot: {reply2}\n")
            
            # Set up the Robot's reply as the next message for the Pirate
            current_message = reply2
            
            # Pause slightly between turns
            page1.wait_for_timeout(2000)
            
        print("=== CONVERSATION FINISHED ===")
        print("\nClosing in 10 seconds...")
        page1.wait_for_timeout(10000)
        context.close()

if __name__ == "__main__":
    main()
