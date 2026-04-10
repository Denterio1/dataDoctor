# ✨ openAuton

**openAuton** is a sophisticated self-improving autonomous AI engineer that continuously evolves through experience. It combines meta-cognition, genetic algorithms, and multi-tool capability to accomplish complex tasks with minimal human intervention.

## 🚀 Features

- **Self-Evolution**: Uses genetic algorithms to evolve its own DNA and improve over time
- **Meta-Cognition**: Reflects on its own actions and generates insights for improvement
- **Multi-Tool Agent**: Web search, file operations, code execution, data analysis
- **Learning from Experience**: Records episodes and learns patterns for future tasks
- **Real-time Web UI**: Interactive dashboard for monitoring and controlling the agent
- **Data Training**: Built-in capability to train on CSV data for specialized tasks
- **Adaptive Strategy**: Dynamically adjusts approach based on task difficulty and available tools

## 📋 Quick Start

### Prerequisites
- Python 3.11+
- pip / venv

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/openAuton.git
cd openAuton

# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Agent

```bash
# Start the web server
python web/app.py

# Server will be available at http://127.0.0.1:8000
```

## 🎯 Usage Examples

### Via Web UI

1. Open http://127.0.0.1:8000 in your browser
2. Chat with openAuton in the sidebar
3. Watch in real-time as it plans and executes tasks
4. Monitor its learning progress and DNA evolution

### Training on Custom Data

1. Prepare a CSV file with your data
2. Use the "Train on File" feature in the web UI
3. openAuton will analyze the data and train a specialized model
4. View accuracy metrics and generated episodes

**Example CSV Structure:**
```
id,product_name,review_text,rating,verified_purchase,helpful_votes,review_date,category,price,customer_verified
1,Laptop X1,Great product,5,true,42,2024-01-15,Electronics,999.99,true
2,Mouse Pro,Works well,4,true,15,2024-01-16,Electronics,29.99,true
...
```

### API Endpoints

- `POST /api/chat` - Send a task to the agent
- `POST /api/train-file` - Upload CSV for training
- `GET /api/memory` - View learning episodes
- `GET /api/dna` - View current agent genes
- `GET /api/status` - Get agent status

## 📁 Project Structure

```
openAuton/
├── web/
│   ├── app.py              # FastAPI application & web UI
│   └── static/             # Frontend assets
├── src/
│   ├── core/
│   │   ├── agent.py        # Main agent orchestrator
│   │   ├── meta_agent.py   # Meta-cognition system
│   │   └── intuition.py    # Pattern matching engine
│   ├── genome/             # Genetic algorithm & DNA evolution
│   ├── experience/         # Episode recording system
│   ├── memory/             # Long-term experience memory
│   ├── llm/                # LLM provider integration
│   ├── tools/              # Multi-tool registry
│   └── training/           # Model training pipeline
├── experiments/
│   ├── episodes/           # Recorded learning episodes (JSON)
│   ├── dna_snapshots/      # DNA evolution history (YAML)
│   ├── universal_models/   # Trained models
│   └── uploads/            # Uploaded datasets
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── config/                 # Configuration files
│   └── prompts/            # LLM prompts for agent behavior
└── requirements.txt        # Python dependencies
```

## 🧠 How It Works

### The Agent Loop

openAuton operates in a continuous cycle:

1. **Observe** → Understand task requirements
2. **Plan** → Decide which tools to use
3. **Verify** → Check if tools are available
4. **Act** → Execute the tools
5. **Reflect** → Analyze what went well/poorly
6. **Evolve** → Generate genetic mutations for improvement
7. **Store** → Record episode for future learning

### Meta-Cognition

The meta-learning system operates at two levels:
- **Task Agent**: Executes specific tasks
- **Meta Agent**: Observes task agent performance and suggests improvements

This creates a hierarchy where the agent can improve itself by analyzing its own behavior.

### Genetic Evolution

The agent's "DNA" encodes:
- Tool usage preferences
- Risk assessment thresholds
- Learning rate parameters
- Communication style

These genes mutate based on task success/failure, creating a lineage of progressively better agents.

## 📊 Example Results

**Sentiment Analysis Training** (on 2000 samples):
- Accuracy: 91.86%
- Training Time: <1 minute
- Tools Used: Data preprocessing, model training, evaluation
- Episode Recorded: Successfully stored for future learning

## 🔧 Configuration

Edit `config/agent_dna.yaml` to customize:
- `budget_usd`: Maximum cost for tool usage
- `auto_save`: Auto-save episodes
- `verbose`: Debug logging level
- `risk_thresholds`: Task execution thresholds

Edit prompts in `config/prompts/` to modify agent behavior:
- `observe.txt`: Observation strategy
- `plan.txt`: Planning approach
- `reflect.txt`: Reflection prompts
- `evolve.txt`: Evolution guidelines

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_all.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📝 API Documentation

### POST /api/chat

Send a task to the agent.

**Request:**
```json
{
  "task": "Analyze this CSV file and tell me the main insights",
  "files": []
}
```

**Response:**
```json
{
  "response": "Agent's response here",
  "episode_id": "abc123",
  "tools_used": ["web_search", "code_execution"],
  "status": "success"
}
```

### POST /api/train-file

Upload CSV and train a model.

**Request:** (multipart/form-data)
- `file`: CSV file
- `epochs`: Number of training epochs (optional, default: 10)

**Response:**
```json
{
  "episode_id": "train_123",
  "accuracy": 0.919,
  "samples": 2000,
  "training_time": 0.45
}
```

### GET /api/memory

Get agent's learning episodes.

**Response:**
```json
{
  "total_episodes": 25,
  "recent_episodes": [...],
  "success_rate": 0.88
}
```

### GET /api/dna

View current agent genes.

**Response:**
```json
{
  "generation": 15,
  "genes": {
    "risk_threshold": 0.7,
    "tool_preference": {...},
    "learning_rate": 0.01
  }
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/openAuton/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/yourusername/openAuton/discussions)
- **Documentation**: Full docs at [Wiki](https://github.com/yourusername/openAuton/wiki)

## 🎓 Learning Resources

- [Architecture Overview](docs/architecture.md)
- [Agent DNA System](docs/dna.md)
- [Adding Custom Tools](docs/tools.md)
- [Training Custom Models](docs/training.md)

## 🚀 Roadmap

- [ ] Web UI dashboard with real-time charts
- [ ] Multi-agent collaboration
- [ ] Plugin system for custom tools
- [ ] Cloud deployment templates (Docker, Kubernetes)
- [ ] Advanced memory systems (RAG)
- [ ] Benchmark suite for agent evaluation
- [ ] Integration with popular APIs (OpenAI, Anthropic, etc.)

## ⚡ Performance

- **Startup Time**: ~2 seconds
- **Task Execution**: 1-30 seconds (depends on complexity)
- **Memory Usage**: ~500MB (base) + data-dependent
- **GPU Support**: PyTorch optimized for CUDA/cuDNN

## 🔐 Security

- No sensitive data is logged by default
- Episodes are stored locally
- API endpoints have no built-in authentication (add reverse proxy in production)
- Recommend using environment variables for API keys

## 📅 Changelog

### v1.0.0 (Initial Release)
- Core agent framework
- Meta-cognition system
- Genetic algorithm evolution
- Web UI dashboard
- CSV training capability
- Multi-tool integration

## 🙏 Acknowledgments

This project combines research from:
- Evolutionary algorithms (genetic programming)
- Meta-learning (learning to learn)
- Reinforcement learning (experience-based improvement)
- Natural language processing (task understanding)

---

**Happy automating!** 🤖✨

For questions or suggestions, please open an issue or discussion on GitHub.
