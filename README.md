# AI Research Assistant

A sophisticated multi-agent AI research system that creates diverse analyst personas to conduct comprehensive research interviews. Built with LangChain and LangGraph, featuring both sequential and parallel processing approaches with real-time web interface.

## 🎯 Overview

This project demonstrates advanced AI agent orchestration using two different approaches:

- **LangChain Version**: Sequential processing with simple chains, perfect for learning and prototyping
- **LangGraph Version**: Advanced parallel processing with human-in-the-loop capabilities, production-ready

## ✨ Key Features

### Core Functionality
- **Multi-Agent Research**: Creates 3-5 AI analyst personas with diverse expertise
- **Intelligent Interviews**: Each analyst conducts structured Q&A sessions with expert AI
- **Comprehensive Reports**: Automatically generates detailed research documents
- **Real-time Web Interface**: Live progress tracking and interactive controls

### Implementation Approaches
- **Sequential Processing** (LangChain): Linear workflow, beginner-friendly
- **Parallel Processing** (LangGraph): Concurrent interviews, enterprise-grade
- **Human Feedback Integration**: Interactive analyst refinement
- **Multi-source Research**: Wikipedia, Tavily web search, and fallback sources

### Web Interface Features
- Real-time terminal output
- Progress tracking and status indicators
- Interactive analyst cards
- Formatted report display
- Socket.IO for live updates

## 🏗️ Architecture

### LangChain Flow
```
Research Topic → API Setup → Create Analysts → Sequential Interviews → Final Report
```

### LangGraph Flow
```
Research Topic → Graph Build → Create Analysts → Human Feedback → Parallel Interviews → State Merge → Final Report
```

## 📋 Requirements

### Python Dependencies
```bash
# Core Flask dependencies
Flask==2.3.3
Flask-SocketIO==5.3.6

# LangChain and AI dependencies
langchain==0.1.0
langchain-anthropic==0.1.0
langchain-community==0.0.13
langgraph==0.0.21

# Optional for enhanced search
tavily-python==0.3.0

# Data processing
pydantic==2.5.0
wikipedia==1.4.0
```

### API Keys Required
- **Anthropic API Key** (Required): For Claude AI model
- **Tavily API Key** (Optional): For enhanced web search in LangGraph version

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repository-url>
cd langchain-langgraph-research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

#### Option 1: Environment Variables
```bash
# Set environment variables
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export TAVILY_API_KEY="your-tavily-api-key"  # Optional for LangGraph
```

#### Option 2: .env File (Recommended)
Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=your-anthropic-api-key
TAVILY_API_KEY=your-tavily-api-key
```

### 3. Run Applications

#### Web Interface (Recommended)
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

#### Command Line - LangChain Version
```bash
python LangChain_ResearchAssistant.py
```

#### Command Line - LangGraph Version
```bash
python LangGraph_ResearchAssistant1.py
```

## 💻 Usage Examples

### Web Interface
1. Open the web interface at `http://localhost:5000`
2. Enter your research topic (e.g., "Benefits of meditation for productivity")
3. Choose implementation (LangChain or LangGraph)
4. Set number of analysts (2-5)
5. Optionally provide human feedback
6. Click "Start Research" and monitor progress in real-time

### Command Line
```python
# LangChain Example
from LangChain_ResearchAssistant import SimpleResearchAssistant

assistant = SimpleResearchAssistant()
report = assistant.run_research("AI in healthcare", num_analysts=3)
print(report)

# LangGraph Example
from LangGraph_ResearchAssistant1 import SimpleResearchAssistant

assistant = SimpleResearchAssistant()
report = assistant.run_research("Climate change solutions", max_analysts=3)
print(report)
```

## 🔧 Configuration Options

### Research Parameters
- **Topic**: Any research subject
- **Analyst Count**: 2-5 diverse expert personas
- **Implementation**: Sequential (LangChain) or Parallel (LangGraph)
- **Human Feedback**: Optional analyst refinement

### System Configuration
- **API Endpoints**: Anthropic Claude, Tavily Search, Wikipedia
- **Output Format**: Markdown reports with structured sections
- **Processing Mode**: Sequential or parallel interview execution

## 📊 Comparison: LangChain vs LangGraph

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| **Processing** | Sequential | ✅ Parallel |
| **Data Sources** | Wikipedia only | ✅ Tavily + Wikipedia |
| **Human Interaction** | Manual | ✅ Built-in feedback loops |
| **Code Complexity** | ✅ Simple (~300 lines) | Advanced (~700 lines) |
| **State Management** | Manual | ✅ Automatic |
| **Best For** | ✅ Learning, prototypes | ✅ Production, enterprise |

## 🛠️ Technical Implementation

### Key Components

#### Data Models
```python
class Analyst(BaseModel):
    name: str = Field(description="Name of the analyst")
    role: str = Field(description="Professional role/title")
    affiliation: str = Field(description="Organization affiliation")
    description: str = Field(description="Expertise and focus areas")
```

#### LangChain Chains
- **Analyst Creation**: Generates diverse expert personas
- **Question Generation**: Creates interview questions
- **Expert Response**: Provides researched answers
- **Section Writing**: Formats interview content
- **Final Report**: Compiles comprehensive analysis

#### LangGraph State Management
```python
class ResearchGraphState(TypedDict):
    topic: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    final_report: str
```

### Web Interface Architecture
- **Flask Backend**: Research orchestration and API management
- **Socket.IO**: Real-time communication and progress updates
- **Frontend**: Responsive interface with live terminal and progress tracking

### Additional Files

- **visual_interview.html**: Interactive demo page with detailed architecture visualization and code examples
- **.env**: Store your API keys securely (create this file)
- **templates/**: Flask templates directory (if using template-based routing)
- **venv/**: Python virtual environment (automatically excluded from git)
- **__pycache__/**: Python bytecode cache (automatically excluded from git)


## 🔍 Research Process

### 1. Analyst Generation
The system creates 3-5 diverse AI personas with different:
- Professional backgrounds
- Areas of expertise
- Organizational affiliations
- Research perspectives

### 2. Interview Execution
Each analyst conducts structured interviews:
- Generates domain-specific questions
- Searches relevant information sources
- Receives expert-level responses
- Iterates through multiple conversation turns

### 3. Report Compilation
Final reports include:
- Executive summary
- Detailed analysis from each perspective
- Key takeaways and insights
- Properly formatted markdown output

## 🚨 Troubleshooting

### Common Issues

**API Key Errors**
```bash
# Verify environment variables
echo $ANTHROPIC_API_KEY
echo $TAVILY_API_KEY
```

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Web Interface Issues**
- Check that Flask-SocketIO is properly installed
- Ensure port 5000 is available
- Verify browser supports WebSocket connections

**Research Failures**
- Confirm API keys are valid and have sufficient credits
- Check internet connection for Wikipedia/Tavily searches
- Review terminal output for specific error messages


### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔗 Resources

- [LangChain Documentation](https://docs.langchain.com)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anthropic Claude API](https://docs.anthropic.com)
- [Tavily Search API](https://tavily.com)



**Built with ❤️ using LangChain, LangGraph, and Claude AI**