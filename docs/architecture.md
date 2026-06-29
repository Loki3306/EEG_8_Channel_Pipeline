# System Architecture Specification

Below is the visual block diagram and communication interface overview for the EEG Autonomous Engineering Pipeline v1.0.

## 🔄 Interaction Diagram

```mermaid
graph TD
    User([User]) -->|Trigger review| AG[Antigravity Agent]
    AG -->|Invoke tool| MCP[EEG MCP Server]
    
    subgraph Local Validation Suite
        MCP -->|Static checks| RU[Ruff & PyCompile]
        MCP -->|Domain heuristics| EV[EEG Validator]
    end
    
    subgraph Persistent Storage
        MCP -->|Query / Save| RM[Research Memory]
        MCP -->|Immutably record| AT[Audit Trail]
    end
    
    subgraph Browser Bridge
        MCP -->|Write prompt.txt| IPC[antigravity_chatgpt_ipc.py]
        IPC -->|Chrome Debug Protocol| CG[ChatGPT Browser]
        CG -->|Return response| IPC
        IPC -->|Write response.txt| MCP
    end
    
    MCP -->|Perform double-check| IV[Independent Verifier]
    IV -->|Query through Bridge| IPC
    
    MCP -->|Final Status| Result{PASS / FAIL}
```

---

## 📞 Communication Protocols

1. **User ↔ Antigravity**: Uses MCP-enabled IDE commands to launch reviews.
2. **Antigravity ↔ MCP Server**: Uses JSON-RPC standard MCP commands over standard I/O pipes.
3. **MCP Server ↔ Browser Bridge**: Leverages standard file-system IPC (`scratch/prompt.txt` & `scratch/response.txt`) polled synchronously at 1-second intervals.
4. **Browser Bridge ↔ ChatGPT**: Playwright chromium context communicating directly with the ChatGPT DOM interface using Chrome DevTools Protocol.
