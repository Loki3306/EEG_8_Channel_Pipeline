import os
import time
import sys
from pathlib import Path

# Add the .mcp directory to sys.path so we can import LLMReviewer from server.py
mcp_dir = Path(__file__).resolve().parent.parent
if str(mcp_dir) not in sys.path:
    sys.path.append(str(mcp_dir))

from server import LLMReviewer

class BrowserChatGPTReviewer(LLMReviewer):
    def __init__(self):
        # We need absolute paths relative to repo root, assuming CWD is repo root
        repo_root = mcp_dir.parent
        self.prompt_file = repo_root / "scratch" / "prompt.txt"
        self.response_file = repo_root / "scratch" / "response.txt"
        self.timeout_sec = 300

    def review(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}".strip()
        
        # Write prompt to trigger the Playwright script
        with open(self.prompt_file, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
            
        # Wait for the response
        start_time = time.time()
        response_text = ""
        while time.time() - start_time < self.timeout_sec:
            if self.response_file.exists():
                try:
                    with open(self.response_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        # Our Playwright script clears the file when it starts processing,
                        # and writes the actual response when done.
                        if content and not content.startswith("ERROR:"):
                            response_text = content
                            break
                        elif content.startswith("ERROR:"):
                            raise RuntimeError(f"ChatGPT Bridge Error: {content}")
                except Exception:
                    pass
            time.sleep(1.0)
            
        if not response_text:
            raise TimeoutError("Timed out waiting for ChatGPT response from browser bridge.")
            
        # Clear response file for the next review
        with open(self.response_file, 'w', encoding='utf-8') as f:
            f.write("")
            
        return response_text
