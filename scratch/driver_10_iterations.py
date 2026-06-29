import time
import sys

def main():
    # Force stdout to be unbuffered so logs stream live
    sys.stdout.reconfigure(line_buffering=True)
    
    prompts = [
        "What is 1+1? Answer with just the number.",
        "What is the capital of France? Answer with just the city.",
        "Name a primary color. Just one word.",
        "How many legs does a spider have? Just the number.",
        "What gas do plants absorb? Just the name.",
        "What is the largest mammal? Just the name.",
        "What planet is known as the Red Planet? Just the name.",
        "Who wrote Romeo and Juliet? Just the last name.",
        "What is the boiling point of water in Celsius? Just the number.",
        "Is Playwright awesome? Answer Yes or No."
    ]

    prompt_file = "scratch/prompt.txt"
    response_file = "scratch/response.txt"

    print("Starting 10 fast iterations...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/10] Sending: {prompt}")
        
        # Clear the response file first
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write("")
            
        # Write the new prompt for the IPC script to pick up
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
            
        # Poll until the response file is filled by the IPC script
        reply = ""
        while not reply:
            try:
                with open(response_file, 'r', encoding='utf-8') as f:
                    reply = f.read().strip()
            except Exception:
                pass
            time.sleep(0.5)
            
        print(f"[{i+1}/10] Received: {reply}")
        
        # Give a small 1-second pause so you can see it visually before the next one fires
        time.sleep(1)
        
    print("\nSuccessfully finished 10 iterations!")

if __name__ == "__main__":
    main()
