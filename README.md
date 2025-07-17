# LangChain Research Assistant

A comprehensive AI-powered research assistant that generates detailed reports by orchestrating interviews between specialized AI analyst personas and domain experts. This project demonstrates advanced LangChain/LangGraph implementations for multi-agent research workflows.

## Overview

The system creates diverse AI analyst personas, each with specific expertise areas, who conduct structured interviews with AI experts. These interviews are then synthesized into professional research reports with proper citations and comprehensive analysis.

**Two Implementation Approaches:**
- **LangChain Version**: Sequential processing with detailed documentation for educational purposes
- **LangGraph Version**: Production-ready implementation with parallel processing and human-in-the-loop capabilities

## Architecture

### Core Components

- **Analyst Generation**: Creates diverse expert personas based on research topic
- **Interview Orchestration**: Manages Q&A sessions between analysts and experts
- **Information Retrieval**: Integrates Wikipedia and web search for comprehensive data gathering
- **Report Synthesis**: Combines multiple perspectives into cohesive research documents

### Technology Stack

- **LangChain/LangGraph**: AI application framework and workflow orchestration
- **Anthropic Claude**: Primary language model for reasoning and content generation
- **Tavily Search**: Web search integration for current information
- **Wikipedia API**: Encyclopedic knowledge base access
- **Pydantic**: Data validation and structured output management

## Installation

### Prerequisites

- Python 3.8 or higher
- Anthropic API key (required)
- Tavily API key (optional, enhances search capabilities)

### Setup Instructions

1. **Clone Repository**
   ```bash
   git clone https://github.com/vinyakestur/langchain-langgraph-research-assistant.git
   cd langchain-langgraph-research-assistant
   ```

2. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configuration**
   
   Create `.env` file:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   TAVILY_API_KEY=your_tavily_api_key  # Optional
   LANGSMITH_API_KEY=your_langsmith_key  # Optional
   ```

## Usage

### Basic Implementation (LangChain)

```bash
python LangChain_ResearchAssistant.py
```

Features:
- Wikipedia-based research
- Sequential interview processing
- Comprehensive code documentation
- Minimal API requirements

### Advanced Implementation (LangGraph)

```bash
python LangGraph_ResearchAssistant.py
```

Features:
- Multi-source information retrieval
- Parallel interview execution
- Human feedback integration
- Advanced state management
- Production-ready architecture

## Configuration

### Environment Variables

| Variable | Status | Purpose |
|----------|--------|---------|
| `ANTHROPIC_API_KEY` | Required | Claude AI model access |
| `TAVILY_API_KEY` | Optional | Web search enhancement |
| `LANGSMITH_API_KEY` | Optional | Workflow debugging and tracing |

### Customizable Parameters

- **Analyst Count**: 1-10 analysts per research topic
- **Interview Depth**: Configurable question rounds per analyst
- **Search Sources**: Wikipedia, web search, or combined
- **Output Format**: Structured markdown with citations

## System Workflow

1. **Topic Analysis**: Analyzes research topic to determine key themes
2. **Persona Creation**: Generates specialized analyst personas with distinct perspectives
3. **Research Phase**: Each analyst conducts focused interviews with expert AI
4. **Information Gathering**: Searches multiple sources for comprehensive data
5. **Synthesis**: Combines all perspectives into structured research report
6. **Output Generation**: Produces professional markdown report with proper citations

## Output Example

```markdown
# Machine Learning in Financial Services

## Executive Summary
Analysis of ML implementation across banking, insurance, and investment sectors...

## Risk Management Perspective
Advanced algorithms are transforming traditional risk assessment models...

## Regulatory Compliance Analysis  
Current frameworks struggle to adapt to AI-driven decision making...

## Sources
[1] Federal Reserve AI Guidelines 2024
[2] McKinsey Financial AI Report 2024
```

## Dependencies

Core requirements are listed in `requirements.txt`:

```
langchain>=0.1.0
langchain-anthropic>=0.1.0
langchain-community>=0.0.25
langgraph>=0.0.30
anthropic>=0.8.0
tavily-python>=0.3.0
pydantic>=2.0.0
wikipedia>=1.4.0
```

## API Integration

### Anthropic Claude
Primary language model providing reasoning, analysis, and content generation capabilities.

### Tavily Search
Optional web search integration for accessing current information beyond Wikipedia's scope.

### Wikipedia
Reliable knowledge base providing comprehensive background information on research topics.

## Security Considerations

- API keys must be stored securely in environment variables
- Never commit sensitive credentials to version control
- Use `.gitignore` to exclude `.env` files from repository
- Regenerate API keys if accidentally exposed

## Error Handling

The system implements graceful degradation:
- Continues operation with Wikipedia-only search if Tavily is unavailable
- Provides detailed error messages for troubleshooting
- Maintains functionality with minimal API requirements

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Implement changes with appropriate tests
4. Submit pull request with detailed description

## License

MIT License - see LICENSE file for complete terms.

## Technical Documentation

Detailed implementation documentation is available within the source code comments, covering:
- LangChain chain construction patterns
- LangGraph state management principles
- Multi-agent coordination strategies
- Prompt engineering best practices