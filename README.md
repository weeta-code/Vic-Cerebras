# Grepal - The Snarky Debug Buddy

> *"Your code is bad, and you should feel bad... but here's how to fix it."*

Grepal is a **real-time debugging companion** that combines the lightning-fast inference of Cerebras LLMs with the brutal honesty your code deserves. Say goodbye to boring error messages and hello to a snarky AI that roasts your bugs while actually helping you fix them.

## What Makes Grepal Special

- **Real-time Analysis**: Powered by Cerebras' ultra-fast inference, Grepal analyzes your code as you type
- **Smart Bug Memory**: Uses HuggingFace embeddings to remember similar bugs and their fixes
- **Self-Learning**: Automatically updates its bug database as it encounters new issues
- **Snarky Personality**: Three levels of sass: Mild, Medium, or Savage
- **Multi-Language Support**: Works with JavaScript, TypeScript, Python, Java, C++, Go, and Rust

## Architecture

```
Grepal/
├── src/                    # TypeScript extension code
│   ├── extension.ts        # Main extension entry point
│   ├── grepalClient.ts     # Communication with Python server
│   └── ui/                 # VSCode UI components
├── server/                 # Python backend
│   ├── main.py             # FastAPI server
│   ├── cerebras_client.py  # Cerebras LLM integration
│   ├── embeddings.py       # HuggingFace embeddings
│   └── bug_db.py           # Bug database management
└── .vscode/                # Development configuration
```

## Quick Start

### Prerequisites
- Node.js 16+
- Python 3.8+
- VSCode 1.74+
- Cerebras API key

### Setup

1. **Install dependencies**:
   ```bash
   # Install Node.js dependencies
   npm install
   
   # Set up Python environment
   cd server && ./setup.sh
   ```

2. **Configure API keys**:
   ```bash
   # Create environment file
   echo "CEREBRAS_API_KEY=your_key_here" > server/.env
   ```

3. **Build and test**:
   ```bash
   # Compile TypeScript
   npm run compile
   
   # Start Python server (in separate terminal)
   cd server && python main.py
   
   # Test extension in VSCode
   # Press F5 to launch Extension Development Host
   ```

## Usage

1. **Enable Grepal**: Open Command Palette (`Cmd+Shift+P`) → "Enable Grepal"
2. **Start coding**: Grepal automatically analyzes your code as you type
3. **Adjust snark level**: Go to VSCode Settings → Extensions → Grepal → Snark Level
4. **View insights**: Command Palette → "Grepal: Show Debug Insights"


### Running the Extension
```bash
# Terminal 1: Start Python server
cd server && source grepal_env/bin/activate && python main.py

# Terminal 2: Watch TypeScript compilation
npm run watch

# VSCode: Press F5 to launch Extension Development Host
```

### Testing
```bash
# Run TypeScript linting
npm run lint

# Run Python tests
cd server && pytest
```

## The Grepal Experience

```typescript
// You type this buggy code:
const users = []
for (let i = 0; i <= users.length; i++) {
    console.log(users[i].name)
}

// Grepal immediately responds:
"Really? An off-by-one error in 2024? 
You're trying to access users[0] when the array is empty.
Also, that <= should be <, unless you enjoy undefined errors.
Here's the fix, genius: i < users.length"
```
## License

MIT License - Feel free to fork and improve (your code probably needs it anyway)

---

*Built with love (and a healthy dose of snark) 
