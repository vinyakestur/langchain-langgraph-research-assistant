#!/usr/bin/env python3
"""
SIMPLIFIED LangGraph Research Assistant - GUARANTEED TO WORK
This version simplifies the complex interview sub-graph to ensure it works reliably
"""

import os
import operator
import getpass
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Import required libraries
try:
    from langchain_anthropic import ChatAnthropic
    try:
        from langchain_tavily import TavilySearchResults
        print("✅ Using new Tavily import")
    except ImportError:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            print("⚠️ Using deprecated Tavily import")
        except ImportError:
            TavilySearchResults = None
            print("❌ Tavily not available")
    
    from langchain_community.document_loaders import WikipediaLoader
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langgraph.graph import START, END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.constants import Send
    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    exit(1)

# Setup environment
def setup_environment():
    api_keys = {}
    
    api_keys['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
    if not api_keys['ANTHROPIC_API_KEY']:
        api_keys['ANTHROPIC_API_KEY'] = getpass.getpass("Enter your Anthropic API key: ")
        os.environ['ANTHROPIC_API_KEY'] = api_keys['ANTHROPIC_API_KEY']
    
    api_keys['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    if not api_keys['TAVILY_API_KEY']:
        tavily_key = input("Enter your Tavily API key (optional, press Enter to skip): ")
        if tavily_key.strip():
            api_keys['TAVILY_API_KEY'] = tavily_key
            os.environ['TAVILY_API_KEY'] = tavily_key
        else:
            print("🔍 Will use Wikipedia only for searches")
    
    return api_keys

api_keys = setup_environment()

# Initialize AI model and tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

tavily_search = None
if api_keys.get('TAVILY_API_KEY') and TavilySearchResults:
    try:
        tavily_search = TavilySearchResults(max_results=2)
        print("🌐 Web search enabled with Tavily")
    except Exception as e:
        print(f"⚠️ Tavily initialization failed: {e}")
        tavily_search = None
else:
    print("🔍 Using Wikipedia only for searches")

print("🚀 Research Assistant initialized with Anthropic Claude")

# Data models
class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="List of analysts with their roles and affiliations.")

# Simplified State - NO SUB-GRAPHS
class SimpleResearchState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    final_report: str

# Prompts
ANALYST_INSTRUCTIONS = """You are tasked with creating {max_analysts} AI analyst personas for research topic: {topic}

Create diverse analysts with different perspectives, roles, and affiliations.

{human_analyst_feedback}"""

INTERVIEW_INSTRUCTIONS = """You are {analyst_name}, a {analyst_role} from {analyst_affiliation}.

Research topic: {topic}
Your expertise: {analyst_description}

Conduct a brief research interview. Ask 2 specific questions about this topic from your expert perspective, then provide detailed answers based on the context below.

Context: {context}

Format your response as:
## Interview with {analyst_name}

**Question 1:** [Your first question]
**Answer:** [Detailed answer based on context]

**Question 2:** [Your second question]  
**Answer:** [Detailed answer based on context]

**Summary:** [Brief summary of key insights from your perspective]
"""

REPORT_INSTRUCTIONS = """Create a research report on {topic} based on these analyst interviews:

{interviews}

Format as:
# Research Report: {topic}

## Executive Summary
[Brief overview of key findings]

## Analysis
[Detailed analysis combining all perspectives]

## Conclusion
[Key takeaways and implications]
"""

# Simplified Functions
def create_analysts(state: SimpleResearchState):
    """Create analyst personas"""
    try:
        topic = state['topic']
        max_analysts = state['max_analysts']
        feedback = state.get('human_analyst_feedback', '')
        
        structured_llm = llm.with_structured_output(Perspectives)
        
        system_message = ANALYST_INSTRUCTIONS.format(
            topic=topic,
            max_analysts=max_analysts,
            human_analyst_feedback=feedback
        )
        
        analysts = structured_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the analysts.")
        ])
        
        print(f"✅ Successfully created {len(analysts.analysts)} analysts")
        return {"analysts": analysts.analysts}
        
    except Exception as e:
        print(f"❌ Error creating analysts: {e}")
        return {"analysts": []}

def human_feedback_node(state: SimpleResearchState):
    """Placeholder for human feedback"""
    pass

def conduct_research(state: SimpleResearchState):
    """Simplified research function - no complex sub-graphs"""
    try:
        topic = state['topic']
        analysts = state.get('analysts', [])
        
        print(f"🎤 Conducting research with {len(analysts)} analysts...")
        
        sections = []
        
        for i, analyst in enumerate(analysts, 1):
            print(f"  📝 Research session {i}/{len(analysts)}: {analyst.name}")
            
            # Simple context gathering
            context = gather_context(topic, analyst.description)
            
            # Generate interview
            interview_prompt = INTERVIEW_INSTRUCTIONS.format(
                analyst_name=analyst.name,
                analyst_role=analyst.role,
                analyst_affiliation=analyst.affiliation,
                analyst_description=analyst.description,
                topic=topic,
                context=context
            )
            
            interview = llm.invoke([
                SystemMessage(content="You are conducting a research interview."),
                HumanMessage(content=interview_prompt)
            ])
            
            sections.append(interview.content)
            print(f"  ✅ Completed research with {analyst.name}")
        
        print(f"✅ All research sessions completed ({len(sections)} sections)")
        return {"sections": sections}
        
    except Exception as e:
        print(f"❌ Error in research: {e}")
        return {"sections": [f"Research failed: {str(e)}"]}

def gather_context(topic: str, focus: str) -> str:
    """Gather research context from available sources"""
    context_parts = []
    
    # Try Tavily search
    if tavily_search:
        try:
            search_results = tavily_search.invoke(f"{topic} {focus}")
            for result in search_results[:2]:
                context_parts.append(f"Source: {result.get('url', 'Web')}\n{result.get('content', '')}")
        except Exception as e:
            print(f"⚠️ Tavily search failed: {e}")
    
    # Try Wikipedia
    try:
        wiki_loader = WikipediaLoader(query=f"{topic} {focus}", load_max_docs=1)
        wiki_docs = wiki_loader.load()
        for doc in wiki_docs:
            context_parts.append(f"Source: {doc.metadata.get('source', 'Wikipedia')}\n{doc.page_content[:1000]}")
    except Exception as e:
        print(f"⚠️ Wikipedia search failed: {e}")
    
    if not context_parts:
        context_parts.append(f"General knowledge about {topic} related to {focus}")
    
    return "\n\n---\n\n".join(context_parts)

def generate_final_report(state: SimpleResearchState):
    """Generate the final report"""
    try:
        topic = state['topic']
        sections = state.get('sections', [])
        
        if not sections:
            return {"final_report": f"# Research Report: {topic}\n\nNo research sections were generated."}
        
        interviews_text = "\n\n".join(sections)
        
        report_prompt = REPORT_INSTRUCTIONS.format(
            topic=topic,
            interviews=interviews_text
        )
        
        report = llm.invoke([
            SystemMessage(content="You are a technical writer creating a research report."),
            HumanMessage(content=report_prompt)
        ])
        
        print(f"✅ Final report generated ({len(report.content)} characters)")
        return {"final_report": report.content}
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        fallback_report = f"""# Research Report: {state['topic']}

## Error
Report generation failed: {str(e)}

## Available Sections
{len(state.get('sections', []))} sections were created but could not be compiled.
"""
        return {"final_report": fallback_report}

def route_after_feedback(state: SimpleResearchState):
    """Decide what to do after human feedback"""
    feedback = state.get('human_analyst_feedback')
    if feedback and feedback.strip() and feedback.lower() not in ['no', 'none', '']:
        return "create_analysts"  # Regenerate analysts
    return "conduct_research"   # Continue with research

# Build the simplified graph
def build_simple_research_graph():
    """Build a simple, working research graph"""
    builder = StateGraph(SimpleResearchState)
    
    # Add nodes
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("conduct_research", conduct_research)
    builder.add_node("generate_report", generate_final_report)
    
    # Add edges
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", route_after_feedback, 
                                ["create_analysts", "conduct_research"])
    builder.add_edge("conduct_research", "generate_report")
    builder.add_edge("generate_report", END)
    
    return builder.compile(interrupt_before=['human_feedback'], checkpointer=MemorySaver())

# Main research assistant class
class SimpleResearchAssistant:
    def __init__(self):
        self.graph = build_simple_research_graph()
        
    def run_research(self, topic: str, max_analysts: int = 3):
        thread = {"configurable": {"thread_id": "1"}}
        
        print(f"\n🔬 Starting research on: {topic}")
        print(f"📊 Creating {max_analysts} analysts...")
        
        # Phase 1: Create analysts
        for event in self.graph.stream({
            "topic": topic,
            "max_analysts": max_analysts
        }, thread, stream_mode="values"):
            
            analysts = event.get('analysts', [])
            if analysts:
                print("\n👥 Generated Analysts:")
                for i, analyst in enumerate(analysts, 1):
                    print(f"\n{i}. {analyst.name}")
                    print(f"   Role: {analyst.role}")
                    print(f"   Affiliation: {analyst.affiliation}")
                    print(f"   Focus: {analyst.description}")
        
        # Phase 2: Handle human feedback
        state = self.graph.get_state(thread)
        if 'human_feedback' in state.next:
            feedback = input("\n💭 Any feedback on the analysts? (Press Enter to continue): ")
            
            if feedback.strip() and feedback.lower() not in ['no', 'none']:
                print(f"📝 Applying feedback: {feedback}")
                self.graph.update_state(thread, {
                    "human_analyst_feedback": feedback
                }, as_node="human_feedback")
                
                # Regenerate analysts
                for event in self.graph.stream(None, thread, stream_mode="values"):
                    analysts = event.get('analysts', [])
                    if analysts:
                        print("\n👥 Updated Analysts:")
                        for i, analyst in enumerate(analysts, 1):
                            print(f"\n{i}. {analyst.name}")
                            print(f"   Role: {analyst.role}")
                            print(f"   Affiliation: {analyst.affiliation}")
                            print(f"   Focus: {analyst.description}")
            else:
                self.graph.update_state(thread, {
                    "human_analyst_feedback": ""
                }, as_node="human_feedback")
        
        # Phase 3: Continue with research and report generation
        for event in self.graph.stream(None, thread, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            if node_name == "conduct_research":
                print("🔍 Conducting research sessions...")
            elif node_name == "generate_report":
                print("📝 Generating final report...")
        
        # Phase 4: Get final report
        final_state = self.graph.get_state(thread)
        report = final_state.values.get('final_report')
        
        print("\n🎉 Research completed!")
        return report

def main():
    try:
        assistant = SimpleResearchAssistant()
        
        print("\n" + "="*60)
        print("🤖 Simplified AI Research Assistant")
        print("Powered by LangGraph + Anthropic Claude")
        print("="*60)
        
        topic = input("\n🔍 What topic would you like me to research? ")
        if not topic.strip():
            print("❌ Please provide a valid topic")
            return
            
        max_analysts_input = input("👥 How many analysts should I create? (default 3): ")
        max_analysts = int(max_analysts_input) if max_analysts_input.strip() else 3
        
        report = assistant.run_research(topic, max_analysts)
        
        if report and isinstance(report, str):
            safe_filename = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"research_report_{safe_filename.replace(' ', '_').lower()}.md"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\n💾 Report saved to: {filename}")
            except Exception as e:
                print(f"❌ Could not save file: {e}")
                print("📋 Here's the report:")
                print("="*60)
                print(report)
                return
            
            show_report = input("\n📋 Display the full report? (y/n): ").lower().strip()
            if show_report in ['y', 'yes']:
                print("\n" + "="*60)
                print("📋 RESEARCH REPORT")
                print("="*60)
                print(report)
        else:
            print("❌ Failed to generate report")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Research interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")

if __name__ == "__main__":
    main()

'''source /Users/vinyakestur/Documents/Projects/Projects_Built/Projects_luminity/langchain-langgraph-research-assistant/venv/bin/activate
export TAVILY_API_KEY="your-tavily-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
python LangGraph_ResearchAssistant1.py'''