#!/usr/bin/env python3
"""
LangChain Research Assistant - Beginner-Friendly Version
This version only requires an Anthropic API key and includes detailed explanations

WHAT THIS DOES:
1. Creates AI analyst personas for different perspectives
2. Each analyst "interviews" an expert (another AI) about your topic
3. Combines all interviews into a comprehensive research report
4. Uses only Wikipedia for research (no paid APIs needed)

FLOW:
Topic → Create Analysts → Conduct Interviews → Generate Report
"""

import os
import getpass
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# ================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ================================

# These are the core LangChain libraries we need
try:
    from langchain_anthropic import ChatAnthropic  # For Claude integration
    from langchain_community.document_loaders import WikipediaLoader  # For Wikipedia search
    from langchain_core.prompts import ChatPromptTemplate  # For creating structured prompts
    from langchain_core.output_parsers import PydanticOutputParser  # For parsing structured outputs
    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install with: pip install langchain langchain-anthropic langchain-community wikipedia pydantic")
    exit(1)

# ================================
# STEP 2: SETUP ANTHROPIC API KEY
# ================================

def setup_anthropic_key():
    """
    Get Anthropic API key from environment or user input
    This is the only API key we need for this simplified version
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("🔑 Anthropic API key not found in environment variables")
        api_key = getpass.getpass("Enter your Anthropic API key: ")
        os.environ['ANTHROPIC_API_KEY'] = api_key
    
    print("✅ Anthropic API key configured")
    return api_key

# Initialize the API key and create our LLM instance
api_key = setup_anthropic_key()
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",  # Using Claude 3.5 Sonnet
    temperature=0  # Low temperature for consistent, factual responses
)

print("🤖 Claude AI model initialized")

# ================================
# STEP 3: DEFINE DATA STRUCTURES
# ================================

class Analyst(BaseModel):
    """
    Data structure for an AI analyst persona
    Think of this as a character profile for different types of experts
    """
    name: str = Field(description="Name of the analyst")
    role: str = Field(description="Professional role/title")
    affiliation: str = Field(description="Organization or field they represent")
    description: str = Field(description="Their expertise and focus areas")
    
    @property
    def persona(self) -> str:
        """
        Creates a formatted string representation of the analyst
        This will be used in prompts to give Claude context about who it's playing
        """
        return f"""
Name: {self.name}
Role: {self.role}
Affiliation: {self.affiliation}
Expertise: {self.description}
"""

class Analysts(BaseModel):
    """Container to hold multiple analyst personas"""
    analysts: List[Analyst] = Field(description="List of analyst personas")

class InterviewResult(BaseModel):
    """Stores the result of one analyst's interview"""
    analyst_name: str
    interview_transcript: str  # Full Q&A conversation
    research_context: str     # Background info used
    report_section: str       # Final formatted section for the report

# ================================
# STEP 4: CREATE LANGCHAIN CHAINS
# ================================

# A "chain" in LangChain is a sequence: Prompt → LLM → Output Parser
# Think of it as a reusable workflow for a specific task

# CHAIN 1: Analyst Creation
# This chain creates diverse expert personas for our research topic
analyst_creation_prompt = ChatPromptTemplate.from_template("""
You are an expert at creating diverse research analyst personas.

Research Topic: {topic}
Number of analysts needed: {num_analysts}

Create {num_analysts} different AI analyst personas who would have unique perspectives on this topic.
Make sure they represent different:
- Professional backgrounds
- Areas of expertise  
- Organizational affiliations
- Viewpoints and concerns

For each analyst, provide:
- A realistic name
- Their professional role
- Their affiliation (company, university, organization, etc.)
- A description of their specific expertise and what they'd focus on

Return your response as a JSON object matching this structure:
{{
  "analysts": [
    {{
      "name": "Dr. Sarah Chen",
      "role": "Senior Research Scientist",
      "affiliation": "MIT AI Lab",
      "description": "Focuses on ethical implications and safety concerns in AI development"
    }}
  ]
}}
""")

# Create the output parser - this converts LLM text output into our Python objects
analyst_parser = PydanticOutputParser(pydantic_object=Analysts)

# Combine prompt + LLM + parser into a chain
analyst_creation_chain = analyst_creation_prompt | llm | analyst_parser

# CHAIN 2: Question Generation
# This generates questions that each analyst would ask about the topic
question_generation_prompt = ChatPromptTemplate.from_template("""
You are an analyst conducting an interview about a research topic.

Your persona:
{analyst_persona}

Research Topic: {topic}
Interview turn: {turn_number}
Previous conversation: {conversation_history}

Based on your expertise and role, generate ONE specific, insightful question about this topic.
- If this is turn 1, introduce yourself briefly and ask your first question
- If this is turn 2 or later, ask a follow-up question based on previous answers
- Focus on your area of expertise
- Make it specific and actionable
- If you feel you have enough information, you can say "Thank you for your insights!"

Generate only the question (or thank you), nothing else.
""")

question_generation_chain = question_generation_prompt | llm

# CHAIN 3: Expert Response
# This generates expert answers to analyst questions using research context
expert_response_prompt = ChatPromptTemplate.from_template("""
You are an expert being interviewed about: {topic}

The analyst asking the question has this background:
{analyst_persona}

Their question: {question}

Research context from Wikipedia:
{research_context}

Based on the research context provided, give a comprehensive, expert-level answer to the question.
Guidelines:
- Use information from the research context
- Be specific and provide concrete examples
- Stay focused on the analyst's area of interest
- Include relevant facts, figures, and details
- Keep the response conversational but informative

Your expert response:
""")

expert_response_chain = expert_response_prompt | llm

# CHAIN 4: Section Writing
# This converts interview transcripts into polished report sections
section_writing_prompt = ChatPromptTemplate.from_template("""
You are a technical writer creating a section of a research report.

Analyst's area of expertise: {analyst_expertise}
Interview transcript: {interview_transcript}
Research context: {research_context}

Create a well-structured report section based on this interview.

Format:
## [Create an engaging title based on the analyst's focus area]

### Key Insights
[2-3 paragraphs summarizing the main findings from this perspective]

### Details
[Additional details, examples, and supporting information]

Use clear, professional language. Focus on the insights and information, not on the interview process itself.
""")

section_writing_chain = section_writing_prompt | llm

# CHAIN 5: Final Report Compilation
# This combines all sections into a comprehensive final report
final_report_prompt = ChatPromptTemplate.from_template("""
You are a senior research analyst compiling a comprehensive report.

Research Topic: {topic}
Individual analyst sections:
{all_sections}

Create a professional research report that synthesizes all the analyst perspectives.

Format:
# [Creative, engaging title for the research topic]

## Executive Summary
[2-3 paragraphs highlighting the most important findings across all perspectives]

## Detailed Analysis
[Integrate and organize the content from all analyst sections into a coherent narrative]

## Key Takeaways
[3-5 bullet points with the most important insights]

## Conclusion
[Final thoughts and implications]

Make sure to:
- Avoid redundancy between sections
- Create smooth transitions between different perspectives
- Highlight areas where analysts agree or disagree
- Maintain a professional, analytical tone
""")

final_report_chain = final_report_prompt | llm

# ================================
# STEP 5: RESEARCH FUNCTIONS
# ================================

def search_wikipedia(query: str, max_docs: int = 2) -> str:
    """
    Search Wikipedia for information about a topic
    
    Args:
        query: Search term
        max_docs: Maximum number of Wikipedia pages to retrieve
    
    Returns:
        Combined text from Wikipedia articles
    """
    try:
        print(f"🔍 Searching Wikipedia for: {query}")
        
        # WikipediaLoader is a LangChain tool that fetches Wikipedia content
        loader = WikipediaLoader(query=query, load_max_docs=max_docs)
        docs = loader.load()
        
        if not docs:
            return "No Wikipedia results found for this query."
        
        # Combine all documents into one text block
        combined_text = ""
        for doc in docs:
            combined_text += f"\n--- Source: {doc.metadata['source']} ---\n"
            combined_text += doc.page_content
            combined_text += "\n" + "="*50 + "\n"
        
        return combined_text
        
    except Exception as e:
        print(f"⚠️ Wikipedia search failed: {e}")
        return f"Wikipedia search failed: {str(e)}"

def conduct_interview(analyst: Analyst, topic: str, max_turns: int = 3) -> InterviewResult:
    """
    Conduct a full interview between an analyst and expert
    
    This is the core function that simulates a conversation:
    1. Analyst asks a question
    2. We search for relevant information
    3. Expert answers using that information
    4. Repeat for several turns
    
    Args:
        analyst: The analyst persona conducting the interview
        topic: Research topic
        max_turns: Maximum number of question-answer rounds
    
    Returns:
        InterviewResult containing the full interview and formatted section
    """
    print(f"  🎤 Starting interview with {analyst.name} ({analyst.role})")
    
    # Storage for the conversation
    conversation_history = []
    interview_transcript = ""
    all_research_context = ""
    
    for turn in range(1, max_turns + 1):
        print(f"    📝 Turn {turn}/{max_turns}")
        
        # STEP 1: Generate analyst question
        question_response = question_generation_chain.invoke({
            "analyst_persona": analyst.persona,
            "topic": topic,
            "turn_number": turn,
            "conversation_history": "\n".join(conversation_history)
        })
        
        question = question_response.content
        conversation_history.append(f"Analyst: {question}")
        interview_transcript += f"\n**Analyst ({analyst.name}):** {question}\n"
        
        # Check if analyst is done
        if "Thank you" in question or "thank you" in question:
            print(f"    ✅ {analyst.name} concluded the interview")
            break
        
        # STEP 2: Search for relevant information
        # Create a search query based on the question
        search_query = f"{topic} {question}"
        research_context = search_wikipedia(search_query)
        all_research_context += f"\n--- Turn {turn} Research ---\n{research_context}\n"
        
        # STEP 3: Generate expert response
        expert_response = expert_response_chain.invoke({
            "topic": topic,
            "analyst_persona": analyst.persona,
            "question": question,
            "research_context": research_context
        })
        
        answer = expert_response.content
        conversation_history.append(f"Expert: {answer}")
        interview_transcript += f"**Expert:** {answer}\n"
    
    print(f"    ✅ Interview completed with {analyst.name}")
    
    # STEP 4: Convert interview to report section
    print(f"    📋 Writing report section for {analyst.name}")
    section_response = section_writing_chain.invoke({
        "analyst_expertise": analyst.description,
        "interview_transcript": interview_transcript,
        "research_context": all_research_context
    })
    
    return InterviewResult(
        analyst_name=analyst.name,
        interview_transcript=interview_transcript,
        research_context=all_research_context,
        report_section=section_response.content
    )

# ================================
# STEP 6: MAIN RESEARCH CLASS
# ================================

class SimpleResearchAssistant:
    """
    Main class that orchestrates the entire research process
    
    This class ties together all the chains and functions to:
    1. Create analyst personas
    2. Conduct interviews with each analyst
    3. Generate a comprehensive final report
    """
    
    def __init__(self):
        self.llm = llm
        print("🚀 Research Assistant initialized")
    
    def create_analysts(self, topic: str, num_analysts: int = 3) -> List[Analyst]:
        """
        Create diverse analyst personas for the research topic
        
        Args:
            topic: What we're researching
            num_analysts: How many different perspectives we want
        
        Returns:
            List of Analyst objects
        """
        print(f"👥 Creating {num_analysts} analyst personas for: {topic}")
        
        try:
            # Use our analyst creation chain
            result = analyst_creation_chain.invoke({
                "topic": topic,
                "num_analysts": num_analysts
            })
            
            # Display the created analysts
            print(f"✅ Successfully created {len(result.analysts)} analysts:")
            for i, analyst in enumerate(result.analysts, 1):
                print(f"  {i}. {analyst.name} - {analyst.role}")
                print(f"     Focus: {analyst.description}")
            
            return result.analysts
            
        except Exception as e:
            print(f"❌ Error creating analysts: {e}")
            return []
    
    def conduct_all_interviews(self, analysts: List[Analyst], topic: str) -> List[InterviewResult]:
        """
        Conduct interviews with all analysts sequentially
        
        Args:
            analysts: List of analyst personas
            topic: Research topic
        
        Returns:
            List of interview results
        """
        print(f"\n🎤 Conducting interviews with {len(analysts)} analysts")
        
        results = []
        for i, analyst in enumerate(analysts, 1):
            print(f"\n📋 Interview {i}/{len(analysts)}: {analyst.name}")
            
            try:
                result = conduct_interview(analyst, topic)
                results.append(result)
                print(f"✅ Successfully completed interview {i}")
                
            except Exception as e:
                print(f"❌ Error in interview {i}: {e}")
                continue
        
        return results
    
    def generate_final_report(self, topic: str, interview_results: List[InterviewResult]) -> str:
        """
        Combine all interview results into a comprehensive report
        
        Args:
            topic: Research topic
            interview_results: Results from all interviews
        
        Returns:
            Final formatted report as a string
        """
        print(f"\n📝 Generating final report from {len(interview_results)} interviews")
        
        # Combine all sections
        all_sections = []
        for result in interview_results:
            all_sections.append(result.report_section)
        
        sections_text = "\n\n".join(all_sections)
        
        # Generate the final report
        final_report = final_report_chain.invoke({
            "topic": topic,
            "all_sections": sections_text
        })
        
        return final_report.content
    
    def run_research(self, topic: str, num_analysts: int = 3) -> str:
        """
        Run the complete research process
        
        This is the main function that orchestrates everything:
        1. Creates analysts
        2. Conducts interviews
        3. Generates final report
        
        Args:
            topic: What to research
            num_analysts: How many different perspectives
        
        Returns:
            Complete research report
        """
        print(f"\n🔬 Starting research on: {topic}")
        print("="*60)
        
        try:
            # STEP 1: Create analysts
            analysts = self.create_analysts(topic, num_analysts)
            if not analysts:
                return "❌ Failed to create analysts"
            
            # STEP 2: Conduct interviews
            interview_results = self.conduct_all_interviews(analysts, topic)
            if not interview_results:
                return "❌ No successful interviews completed"
            
            # STEP 3: Generate final report
            final_report = self.generate_final_report(topic, interview_results)
            
            print("\n🎉 Research completed successfully!")
            return final_report
            
        except Exception as e:
            error_msg = f"❌ Research failed: {str(e)}"
            print(error_msg)
            return error_msg

# ================================
# STEP 7: MAIN EXECUTION
# ================================

def main():
    """
    Main function - this is where the program starts
    """
    print("\n" + "="*60)
    print("🤖 LangChain Research Assistant")
    print("Simple version using only Anthropic API + Wikipedia")
    print("="*60)
    
    # Create our research assistant
    assistant = SimpleResearchAssistant()
    
    # Get input from user
    print("\n📝 Let's start your research!")
    topic = input("🔍 What topic would you like to research? ")
    
    if not topic.strip():
        print("❌ Please provide a research topic")
        return
    
    # Get number of analysts (optional)
    try:
        num_analysts = int(input("👥 How many different perspectives? (default: 3) ") or "3")
        if num_analysts < 1 or num_analysts > 10:
            print("⚠️ Using default of 3 analysts")
            num_analysts = 3
    except ValueError:
        num_analysts = 3
    
    # Run the research
    print(f"\n🚀 Starting research with {num_analysts} analysts...")
    report = assistant.run_research(topic, num_analysts)
    
    # Save and display results
    if report and not report.startswith("❌"):
        # Save to file
        filename = f"research_report_{topic.replace(' ', '_').lower()}.md"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n💾 Report saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save file: {e}")
        
        # Ask if user wants to see the report
        show_report = input("\n📋 Would you like to see the report now? (y/n): ").lower().strip()
        if show_report == 'y':
            print("\n" + "="*60)
            print("📋 RESEARCH REPORT")
            print("="*60)
            print(report)
    else:
        print(f"\n{report}")

if __name__ == "__main__":
    main()

# ================================
# FLOW SUMMARY FOR BEGINNERS
# ================================

"""
🔄 COMPLETE FLOW EXPLANATION:

1. SETUP PHASE:
   - Import LangChain libraries
   - Get your Anthropic API key
   - Create Claude AI model instance

2. ANALYST CREATION:
   - You provide a research topic
   - AI creates 3 different expert personas (e.g., scientist, policy maker, industry expert)
   - Each has different expertise and perspective

3. INTERVIEW PHASE (for each analyst):
   - Analyst asks a question about the topic
   - System searches Wikipedia for relevant information
   - Expert (AI) answers using the research context
   - Repeat for 2-3 rounds per analyst

4. REPORT GENERATION:
   - Each interview gets converted to a report section
   - All sections are combined into one comprehensive report
   - Final report includes summary, analysis, and conclusions

5. OUTPUT:
   - Complete research report saved as markdown file
   - Option to display on screen

🎯 KEY CONCEPTS:
- Chains: Reusable workflows (Prompt → LLM → Parser)
- Personas: Different AI characters for diverse perspectives
- Context: Background information fed to AI for better answers
- Pydantic: Data validation and structure
- Sequential Processing: One step after another (vs parallel)

🔧 WHAT YOU NEED:
- Just an Anthropic API key (no other paid services)
- Python packages: langchain, langchain-anthropic, langchain-community, pydantic, wikipedia
"""

'''cd /Users/vinyakestur/Desktop/langchain-langgraph-research-assistant
source venv/bin/activate
export ANTHROPIC_API_KEY=your_actual_api_key_here
python LangChain_ResearchAssistant.py
'''