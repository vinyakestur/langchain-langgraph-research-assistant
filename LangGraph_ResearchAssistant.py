#!/usr/bin/env python3
"""
LangGraph Research Assistant - Complete Single File Implementation with Detailed Comments
Uses Anthropic Claude for AI reasoning and multiple sources for research

This script demonstrates:
1. LangGraph state management and graph construction
2. Multi-agent AI systems with different personas
3. Parallel processing using Send API
4. Human-in-the-loop interactions
5. Structured data models with Pydantic
6. External API integrations (Anthropic, Tavily, Wikipedia)
"""

# ================================
# IMPORTS AND DEPENDENCIES
# ================================

import os                    # For environment variables and file operations
import operator             # For type annotations with operators (like operator.add)
import getpass              # For secure password/API key input
from typing import List, Annotated    # For type hints
from typing_extensions import TypedDict  # For typed dictionaries (state definitions)
from pydantic import BaseModel, Field    # For data validation and structured models

# Try to import all required LangChain/LangGraph libraries
# If any are missing, show a helpful error message
try:
    from langchain_anthropic import ChatAnthropic  # Anthropic's Claude integration
    from langchain_community.tools.tavily_search import TavilySearchResults  # Web search tool
    from langchain_community.document_loaders import WikipediaLoader  # Wikipedia search
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string  # Message types
    from langgraph.graph import START, END, StateGraph  # Graph construction components
    from langgraph.checkpoint.memory import MemorySaver  # For saving state between runs
    from langgraph.constants import Send  # For parallel processing
    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install with: pip install langgraph langchain-anthropic langchain-community tavily-python wikipedia pydantic")
    exit(1)

# ================================
# ENVIRONMENT SETUP AND API KEYS
# ================================

def setup_environment():
    """
    Setup API keys from environment variables or user input
    
    This function handles API key management in a user-friendly way:
    1. First checks if keys are already set as environment variables
    2. If not, prompts the user to enter them
    3. Makes Tavily optional (can work with just Wikipedia)
    4. Makes LangSmith optional (used for tracing/debugging)
    
    Returns:
        dict: Dictionary containing the API keys
    """
    api_keys = {}
    
    # ANTHROPIC API KEY (Required)
    # This is needed to use Claude for AI reasoning
    api_keys['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
    if not api_keys['ANTHROPIC_API_KEY']:
        # Use getpass for secure input (won't show the key on screen)
        api_keys['ANTHROPIC_API_KEY'] = getpass.getpass("Enter your Anthropic API key: ")
        # Set it as environment variable for the session
        os.environ['ANTHROPIC_API_KEY'] = api_keys['ANTHROPIC_API_KEY']
    
    # TAVILY API KEY (Optional)
    # Tavily provides web search capabilities - makes research more comprehensive
    # But the system can work with just Wikipedia if Tavily isn't available
    api_keys['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    if not api_keys['TAVILY_API_KEY']:
        tavily_key = input("Enter your Tavily API key (optional, press Enter to skip and use only Wikipedia): ")
        if tavily_key.strip():
            api_keys['TAVILY_API_KEY'] = tavily_key
            os.environ['TAVILY_API_KEY'] = tavily_key
        else:
            print("🔍 Will use Wikipedia only for searches (Tavily skipped)")
    
    # LANGSMITH API KEY (Optional)
    # LangSmith is used for tracing and debugging LangChain/LangGraph applications
    # Very helpful for development but not required for functionality
    if not os.getenv('LANGSMITH_API_KEY'):
        langsmith_key = input("Enter LangSmith API key (optional, press Enter to skip): ")
        if langsmith_key.strip():
            os.environ['LANGSMITH_API_KEY'] = langsmith_key
            os.environ['LANGSMITH_TRACING'] = 'true'  # Enable tracing
            os.environ['LANGSMITH_PROJECT'] = 'research-assistant'  # Project name for organization
    
    return api_keys

# Initialize the environment and get API keys
api_keys = setup_environment()

# ================================
# AI MODEL AND TOOLS INITIALIZATION
# ================================

# Initialize the Claude language model
# claude-3-5-sonnet-20241022 is the latest Claude 3.5 Sonnet model
# temperature=0 makes the responses more deterministic and less creative
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

# Initialize Tavily search tool only if API key is available
# This demonstrates graceful degradation - the system works even without all tools
tavily_search = None
if api_keys.get('TAVILY_API_KEY'):
    try:
        # max_results=3 limits the number of search results to keep responses manageable
        tavily_search = TavilySearchResults(max_results=3)
        print("🌐 Web search enabled with Tavily")
    except Exception as e:
        print(f"⚠️ Tavily initialization failed: {e}")
        tavily_search = None
else:
    print("🔍 Using Wikipedia only for searches")

print("🚀 Research Assistant initialized with Anthropic Claude")

# ================================
# DATA MODELS USING PYDANTIC
# ================================

# Pydantic models provide data validation and structure
# They ensure that our data has the right types and required fields

class Analyst(BaseModel):
    """
    Represents an AI analyst persona for research
    
    Each analyst has a specific focus area and background, allowing for
    diverse perspectives on the research topic.
    """
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    
    @property
    def persona(self) -> str:
        """
        Creates a formatted string representation of the analyst's persona
        This is used in prompts to give Claude context about who it's supposed to be
        """
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    """
    Container for multiple analyst perspectives
    Used when Claude generates multiple analysts at once
    """
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")

class SearchQuery(BaseModel):
    """
    Structured format for search queries
    Ensures that Claude returns properly formatted search terms
    """
    search_query: str = Field(None, description="Search query for retrieval.")

# ================================
# STATE DEFINITIONS FOR LANGGRAPH
# ================================

# LangGraph uses TypedDict to define the structure of state that flows through the graph
# Each node can read from and write to this state

class GenerateAnalystsState(TypedDict):
    """
    State for the analyst generation sub-graph
    This manages the process of creating analyst personas
    """
    topic: str                    # The research topic
    max_analysts: int            # How many analysts to create
    human_analyst_feedback: str  # Optional feedback from user
    analysts: List[Analyst]      # The generated analysts

class InterviewState(TypedDict):
    """
    State for individual interview sub-graphs
    Each analyst gets their own interview state
    """
    max_num_turns: int                           # Maximum conversation turns
    context: Annotated[list, operator.add]      # Search results (accumulated with +)
    analyst: Analyst                             # The analyst conducting the interview
    interview: str                               # Full interview transcript
    sections: list                               # Final report sections
    messages: Annotated[list, operator.add]     # Conversation messages (accumulated with +)

class ResearchGraphState(TypedDict):
    """
    Main state for the entire research process
    This flows through the main graph and accumulates all research data
    """
    topic: str                                   # Research topic
    max_analysts: int                           # Number of analysts
    human_analyst_feedback: str                 # User feedback on analysts
    analysts: List[Analyst]                     # All analysts
    sections: Annotated[list, operator.add]    # Report sections from all interviews
    introduction: str                           # Report introduction
    content: str                                # Main report content
    conclusion: str                             # Report conclusion
    final_report: str                           # Complete formatted report

# ================================
# PROMPT TEMPLATES
# ================================

# These are carefully crafted prompts that guide Claude's behavior
# Good prompts are crucial for getting consistent, high-quality results

ANALYST_INSTRUCTIONS = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

QUESTION_INSTRUCTIONS = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

SEARCH_INSTRUCTIONS = SystemMessage(content="""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well-structured web search query""")

ANSWER_INSTRUCTIONS = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.
3. The context contain sources at the topic of each individual document.
4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 
5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/> then just list: 
   [1] assistant/docs/llama3_1.pdf, page 7 
   And skip the addition of the brackets as well as the Document source preamble in your citation."""

SECTION_WRITER_INSTRUCTIONS = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.

7. Be sure to combine sources and avoid redundancy.
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

REPORT_WRITER_INSTRUCTIONS = """You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

Here are the memos from your analysts to build your report from: 

{context}"""

INTRO_CONCLUSION_INSTRUCTIONS = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.
For your introduction, use ## Introduction as the section header. 
For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

# ================================
# ANALYST CREATION FUNCTIONS
# ================================

def create_analysts(state: GenerateAnalystsState):
    """
    Creates AI analyst personas based on the research topic
    
    This function:
    1. Takes the topic and any human feedback
    2. Uses Claude to generate diverse analyst personas
    3. Returns the analysts in the state
    
    Args:
        state: Current state containing topic, max_analysts, and feedback
        
    Returns:
        dict: Updated state with generated analysts
    """
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    
    # Use structured output to ensure Claude returns properly formatted analyst data
    structured_llm = llm.with_structured_output(Perspectives)
    
    # Format the prompt with the current parameters
    system_message = ANALYST_INSTRUCTIONS.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback, 
        max_analysts=max_analysts
    )
    
    # Call Claude to generate the analysts
    analysts = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of analysts.")
    ])
    
    # Return the analysts in the expected state format
    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """
    No-op node for human feedback interruption
    
    This is a placeholder node where LangGraph will pause execution
    to allow human input. The actual feedback is handled by the main
    application logic, not this function.
    """
    pass

def should_continue(state: GenerateAnalystsState):
    """
    Determines the next step after human feedback
    
    This is a conditional edge function that decides whether to:
    1. Regenerate analysts (if feedback was provided)
    2. End the analyst generation process (if no feedback)
    
    Args:
        state: Current state
        
    Returns:
        str: Next node name or END
    """
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"  # Regenerate with feedback
    return END  # Proceed to next stage

# ================================
# INTERVIEW FUNCTIONS
# ================================

def generate_question(state: InterviewState):
    """
    Generates interview questions from the analyst persona
    
    This function makes Claude take on the role of the specific analyst
    and ask relevant questions to an expert about their focus area.
    
    Args:
        state: Interview state containing analyst and message history
        
    Returns:
        dict: Updated state with new question message
    """
    analyst = state["analyst"]
    messages = state["messages"]

    # Use the analyst's persona to guide question generation
    system_message = QUESTION_INSTRUCTIONS.format(goals=analyst.persona)
    
    # Generate the question using the conversation history
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
        
    # Add the question to the message history
    return {"messages": [question]}

def search_web(state: InterviewState):
    """
    Searches the web for relevant information using Tavily
    
    This function:
    1. Analyzes the conversation to determine what to search for
    2. Performs the web search if Tavily is available
    3. Formats the results for use by the expert
    
    Args:
        state: Interview state with conversation messages
        
    Returns:
        dict: Updated state with search results in context
    """
    # Check if Tavily is available
    if not tavily_search:
        # Graceful degradation - return empty context if no web search
        return {"context": ["<Document>Web search not available - using Wikipedia only</Document>"]}
    
    try:
        # Use Claude to generate a good search query from the conversation
        structured_llm = llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([SEARCH_INSTRUCTIONS] + state['messages'])
        
        # Perform the actual web search
        search_docs = tavily_search.invoke(search_query.search_query)

        # Format the search results with proper document tags
        # This makes it easy for Claude to cite sources properly
        formatted_search_docs = "\n\n---\n\n".join([
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ])

        return {"context": [formatted_search_docs]}
    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
        return {"context": ["<Document>Web search failed - using Wikipedia only</Document>"]}

def search_wikipedia(state: InterviewState):
    """
    Searches Wikipedia for relevant information
    
    Wikipedia provides high-quality, encyclopedic information
    and serves as a reliable fallback when web search isn't available.
    
    Args:
        state: Interview state with conversation messages
        
    Returns:
        dict: Updated state with Wikipedia search results in context
    """
    # Generate search query from conversation
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SEARCH_INSTRUCTIONS] + state['messages'])
    
    try:
        # Search Wikipedia (load_max_docs=2 to keep it manageable)
        search_docs = WikipediaLoader(
            query=search_query.search_query, 
            load_max_docs=2
        ).load()

        # Format Wikipedia results with source information
        formatted_search_docs = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    except Exception as e:
        print(f"⚠️ Wikipedia search failed: {e}")
        formatted_search_docs = "<Document>No Wikipedia results found</Document>"

    return {"context": [formatted_search_docs]}

def generate_answer(state: InterviewState):
    """
    Generates expert answers to analyst questions
    
    This function makes Claude take on the role of an expert
    who answers the analyst's questions using the search context.
    
    Args:
        state: Interview state with analyst, messages, and context
        
    Returns:
        dict: Updated state with expert answer
    """
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Create the expert persona prompt with search context
    system_message = ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)
    
    # Generate the expert's answer
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
            
    # Mark the message as coming from the expert (for routing logic)
    answer.name = "expert"
    
    # Add to message history
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    """
    Saves the complete interview transcript
    
    This converts the message history into a readable transcript
    for use in report generation.
    
    Args:
        state: Interview state with complete message history
        
    Returns:
        dict: Updated state with interview transcript
    """
    messages = state["messages"]
    
    # Convert messages to a readable string format
    interview = get_buffer_string(messages)
    
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    """
    Routes between question and answer phases of the interview
    
    This is a conditional edge function that determines whether to:
    1. Continue asking questions
    2. End the interview and save it
    
    The logic checks:
    - Maximum number of turns reached
    - Analyst said "Thank you" (indicates satisfaction)
    
    Args:
        state: Interview state
        name: Name of the expert (for counting responses)
        
    Returns:
        str: Next node name
    """
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    # Count how many times the expert has responded
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    # End if we've reached the maximum number of turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # Check if analyst indicated satisfaction with "Thank you"
    if len(messages) >= 2:
        last_question = messages[-2]
        if "Thank you so much for your help" in last_question.content:
            return 'save_interview'
    
    # Continue the interview
    return "ask_question"

def write_section(state: InterviewState):
    """
    Writes a report section based on the interview and research
    
    This function takes the completed interview and search context
    and generates a structured report section with proper citations.
    
    Args:
        state: Interview state with interview transcript and context
        
    Returns:
        dict: Updated state with completed report section
    """
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
   
    # Use the analyst's focus area to guide section writing
    system_message = SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)
    
    # Generate the section using both interview and search context
    section = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source to write your section: {context}")
    ])
                
    # Add the section to the state
    return {"sections": [section.content]}

# ================================
# REPORT GENERATION FUNCTIONS
# ================================

def initiate_all_interviews(state: ResearchGraphState):
    """
    Initiates parallel interviews using LangGraph's Send API
    
    This is a key function that demonstrates LangGraph's parallel processing.
    It either:
    1. Returns to analyst creation (if feedback provided)
    2. Starts multiple interviews in parallel (using Send API)
    
    The Send API allows each analyst to conduct their interview
    simultaneously, making the process much faster.
    
    Args:
        state: Main research state
        
    Returns:
        str or list: Next node name or list of Send objects for parallel execution
    """
    human_analyst_feedback = state.get('human_analyst_feedback')
    
    # If there's feedback, go back to regenerate analysts
    if human_analyst_feedback:
        return "create_analysts"
    else:
        # Start parallel interviews using Send API
        topic = state["topic"]
        
        # Create a Send object for each analyst
        # Each will run independently in parallel
        return [Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
        }) for analyst in state["analysts"]]

def write_report(state: ResearchGraphState):
    """
    Writes the main report content from all analyst sections
    
    This is the "reduce" step that combines all the parallel
    interview results into a cohesive report.
    
    Args:
        state: Research state with all completed sections
        
    Returns:
        dict: Updated state with main report content
    """
    sections = state["sections"]
    topic = state["topic"]

    # Combine all sections into one context
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Generate the main report content
    system_message = REPORT_WRITER_INSTRUCTIONS.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Write a report based upon these memos.")
    ])
    
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    """
    Writes an engaging introduction for the report
    
    Args:
        state: Research state with completed sections
        
    Returns:
        dict: Updated state with introduction
    """
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=f"Write the report introduction")
    ])
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    """
    Writes a compelling conclusion for the report
    
    Args:
        state: Research state with completed sections
        
    Returns:
        dict: Updated state with conclusion
    """
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=f"Write the report conclusion")
    ])
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """
    Combines all report parts into the final formatted document
    
    This function:
    1. Takes the introduction, content, and conclusion
    2. Handles source formatting
    3. Creates the final markdown report
    
    Args:
        state: Research state with all report components
        
    Returns:
        dict: Updated state with final_report
    """
    content = state["content"]
    
    # Clean up the content formatting
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    
    # Handle sources section
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    # Combine all parts with proper formatting
    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    
    # Add sources if they exist
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
        
    return {"final_report": final_report}

# ================================
# GRAPH CONSTRUCTION
# ================================

def build_interview_graph():
    """
    Builds the interview sub-graph for individual analyst interviews
    
    This graph handles the conversation flow for a single analyst:
    1. Ask question
    2. Search for information (both web and Wikipedia in parallel)
    3. Generate expert answer
    4. Decide whether to continue or finish
    5. Save interview and write section
    
    Returns:
        CompiledGraph: The compiled interview graph
    """
    interview_builder = StateGraph(InterviewState)
    
    # Add all the nodes (functions) to the graph
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)

    # Define the edges (flow) between nodes
    interview_builder.add_edge(START, "ask_question")           # Start with a question
    interview_builder.add_edge("ask_question", "search_web")    # Search web after question
    interview_builder.add_edge("ask_question", "search_wikipedia")  # Also search Wikipedia (parallel)
    interview_builder.add_edge("search_web", "answer_question")     # Answer after web search
    interview_builder.add_edge("search_wikipedia", "answer_question")  # Answer after Wikipedia search
    
    # Conditional edge: decide whether to continue or finish interview
    interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
    
    interview_builder.add_edge("save_interview", "write_section")  # Write section after saving
    interview_builder.add_edge("write_section", END)              # End after writing section

    # Compile the graph with memory for state persistence
    return interview_builder.compile(checkpointer=MemorySaver())

def build_research_graph():
    """
    Builds the main research graph that orchestrates the entire process
    
    This is the top-level graph that:
    1. Creates analysts
    2. Gets human feedback (with interruption)
    3. Conducts interviews in parallel
    4. Writes introduction, content, and conclusion in parallel
    5. Finalizes the complete report
    
    Returns:
        CompiledGraph: The compiled main research graph
    """
    # First build the interview sub-graph
    interview_graph = build_interview_graph()
    
    # Create the main graph using ResearchGraphState
    main_builder = StateGraph(ResearchGraphState)
    
    # Add all nodes to the main graph
    main_builder.add_node("create_analysts", create_analysts)
    main_builder.add_node("human_feedback", human_feedback)        # Interruption point
    main_builder.add_node("conduct_interview", interview_graph)    # Sub-graph for interviews
    main_builder.add_node("write_report", write_report)
    main_builder.add_node("write_introduction", write_introduction)
    main_builder.add_node("write_conclusion", write_conclusion)
    main_builder.add_node("finalize_report", finalize_report)

    # Define the main workflow edges
    main_builder.add_edge(START, "create_analysts")                # Start by creating analysts
    main_builder.add_edge("create_analysts", "human_feedback")    # Then get feedback
    
    # Conditional edge: either regenerate analysts or start interviews
    main_builder.add_conditional_edges("human_feedback", initiate_all_interviews, 
                                     ["create_analysts", "conduct_interview"])
    
    # After interviews complete, write different parts in parallel
    main_builder.add_edge("conduct_interview", "write_report")
    main_builder.add_edge("conduct_interview", "write_introduction")
    main_builder.add_edge("conduct_interview", "write_conclusion")
    
    # Wait for all three writing tasks to complete, then finalize
    main_builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    main_builder.add_edge("finalize_report", END)

    # Compile with memory and interruption before human_feedback
    return main_builder.compile(interrupt_before=['human_feedback'], checkpointer=MemorySaver())

# ================================
# MAIN RESEARCH ASSISTANT CLASS
# ================================

class ResearchAssistant:
    """
    Main class that orchestrates the research process
    
    This class provides a clean interface for users while managing
    the complex LangGraph execution underneath.
    """
    
    def __init__(self):
        """Initialize the research assistant with the main graph"""
        self.graph = build_research_graph()
        
    def run_research(self, topic: str, max_analysts: int = 3):
        """
        Runs the complete research process
        
        This method:
        1. Starts the graph execution
        2. Handles human-in-the-loop feedback
        3. Monitors progress
        4. Returns the final report
        
        Args:
            topic: The research topic
            max_analysts: Number of analysts to create
            
        Returns:
            str: The complete research report in markdown format
        """
        # Create a unique thread for this research session
        # This allows multiple concurrent research sessions
        thread = {"configurable": {"thread_id": "1"}}
        
        print(f"\n🔬 Starting research on: {topic}")
        print(f"📊 Creating {max_analysts} analysts...")
        
        # PHASE 1: Create analysts and get them displayed
        for event in self.graph.stream({
            "topic": topic,
            "max_analysts": max_analysts
        }, thread, stream_mode="values"):
            
            # Check if analysts were generated in this event
            analysts = event.get('analysts', '')
            if analysts:
                print("\n👥 Generated Analysts:")
                for i, analyst in enumerate(analysts, 1):
                    print(f"\n{i}. {analyst.name}")
                    print(f"   Role: {analyst.role}")
                    print(f"   Affiliation: {analyst.affiliation}")
                    print(f"   Focus: {analyst.description}")
        
        # PHASE 2: Handle human feedback (human-in-the-loop)
        state = self.graph.get_state(thread)
        if 'human_feedback' in state.next:
            # The graph is paused at the human_feedback node
            feedback = input("\n💭 Any feedback on the analysts? (Press Enter to continue or provide feedback): ")
            
            if feedback.strip():
                # User provided feedback - regenerate analysts
                print(f"📝 Applying feedback: {feedback}")
                self.graph.update_state(thread, {
                    "human_analyst_feedback": feedback
                }, as_node="human_feedback")
                
                # Re-run the graph to regenerate analysts
                for event in self.graph.stream(None, thread, stream_mode="values"):
                    analysts = event.get('analysts', '')
                    if analysts:
                        print("\n👥 Updated Analysts:")
                        for i, analyst in enumerate(analysts, 1):
                            print(f"\n{i}. {analyst.name}")
                            print(f"   Role: {analyst.role}")
                            print(f"   Affiliation: {analyst.affiliation}")
                            print(f"   Focus: {analyst.description}")
            else:
                # No feedback - continue with current analysts
                self.graph.update_state(thread, {
                    "human_analyst_feedback": None
                }, as_node="human_feedback")
        
        print("\n🎤 Conducting interviews...")
        
        # PHASE 3: Continue with interviews and report generation
        # Use stream_mode="updates" to see which nodes are executing
        for event in self.graph.stream(None, thread, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            
            # Provide user feedback on progress
            if node_name == "conduct_interview":
                print("  ✅ Interview completed")
            elif node_name == "write_report":
                print("  📝 Writing main report...")
            elif node_name == "write_introduction":
                print("  📄 Writing introduction...")
            elif node_name == "write_conclusion":
                print("  🏁 Writing conclusion...")
            elif node_name == "finalize_report":
                print("  ✨ Finalizing report...")
        
        # PHASE 4: Get the final report
        final_state = self.graph.get_state(thread)
        report = final_state.values.get('final_report')
        
        print("\n🎉 Research completed!")
        return report

# ================================
# MAIN APPLICATION ENTRY POINT
# ================================

def main():
    """
    Main function that provides the user interface
    
    This function:
    1. Creates the research assistant
    2. Gets user input for topic and analysts
    3. Runs the research process
    4. Saves and displays results
    """
    
    try:
        # Initialize the research assistant
        assistant = ResearchAssistant()
        
        # Display welcome message
        print("\n" + "="*60)
        print("🤖 AI Research Assistant")
        print("Powered by LangGraph + Anthropic Claude")
        print("="*60)
        
        # Get research parameters from user
        topic = input("\n🔍 What topic would you like me to research? ")
        if not topic.strip():
            print("❌ Please provide a valid topic")
            return
            
        max_analysts_input = input("👥 How many analysts should I create? (default 3): ")
        max_analysts = int(max_analysts_input) if max_analysts_input.strip() else 3
        
        # Run the research process
        report = assistant.run_research(topic, max_analysts)
        
        # Save the report to a file
        # Create a safe filename from the topic
        safe_filename = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"research_report_{safe_filename.replace(' ', '_').lower()}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 Report saved to: {filename}")
        
        # Ask if user wants to see the report in terminal
        show_report = input("\n📋 Display the full report? (y/n): ").lower().strip()
        if show_report in ['y', 'yes']:
            print("\n" + "="*60)
            print("📋 RESEARCH REPORT")
            print("="*60)
            print(report)
        else:
            print(f"✅ Report saved successfully to {filename}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Research interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        print("Please check your API keys and try again")

# ================================
# QUICK TEST FUNCTION
# ================================

def quick_test():
    """
    Runs a quick test with predefined parameters
    
    Useful for testing the system without user input.
    Uncomment the call to this function in __main__ to use it.
    """
    try:
        assistant = ResearchAssistant()
        
        # Test with a simple topic and fewer analysts for speed
        report = assistant.run_research("Benefits of Python for beginners", 2)
        
        # Save test report
        with open("quick_test_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("✅ Quick test completed! Report saved to quick_test_report.md")
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")

# ================================
# SCRIPT EXECUTION
# ================================

if __name__ == "__main__":
    """
    Main execution block
    
    This runs when the script is executed directly (not imported).
    You can switch between interactive mode and quick test mode.
    """
    
    # Uncomment the next line to run a quick test instead of interactive mode
    # quick_test()
    
    # Interactive mode (default)
    main()

# ================================
# KEY CONCEPTS EXPLAINED
# ================================

"""
LANGGRAPH CONCEPTS DEMONSTRATED:

1. STATE MANAGEMENT:
   - TypedDict defines the structure of data flowing through the graph
   - Annotated[list, operator.add] automatically merges lists from parallel branches
   - State persists between node executions

2. GRAPH CONSTRUCTION:
   - StateGraph defines the workflow
   - Nodes are functions that process and update state
   - Edges define the flow between nodes
   - Conditional edges allow dynamic routing based on state

3. PARALLEL PROCESSING:
   - Send API enables parallel execution of multiple sub-graphs
   - Each analyst interview runs independently and simultaneously
   - Results are automatically merged back into the main state

4. HUMAN-IN-THE-LOOP:
   - interrupt_before parameter pauses execution at specified nodes
   - update_state allows external modification of graph state
   - Graph resumes from the interruption point

5. SUB-GRAPHS:
   - Complex workflows can be broken into smaller, reusable graphs
   - Sub-graphs have their own state and can be compiled independently
   - Main graph can include sub-graphs as nodes

6. MEMORY AND PERSISTENCE:
   - MemorySaver checkpointer saves state between runs
   - Threads allow multiple concurrent executions
   - State can be retrieved and modified externally

AI AGENT CONCEPTS:

1. PERSONAS:
   - Each analyst has a distinct role and focus area
   - Prompts are tailored to maintain consistent character
   - Multiple perspectives provide comprehensive coverage

2. TOOL USE:
   - Integration with external APIs (Tavily, Wikipedia)
   - Graceful degradation when tools are unavailable
   - Structured output ensures reliable data parsing

3. MULTI-AGENT ORCHESTRATION:
   - Analysts work independently but contribute to shared goal
   - Results are synthesized into coherent final output
   - Human oversight ensures quality and relevance

4. RESEARCH METHODOLOGY:
   - Interview format encourages deep exploration
   - Multiple information sources provide comprehensive coverage
   - Structured report format ensures readability and citations
"""
'''cd /Users/vinyakestur/Desktop/langchain-langgraph-research-assistant
source venv/bin/activate
export ANTHROPIC_API_KEY=your_actual_api_key_here
python LangGraph_ResearchAssistant.py
'''