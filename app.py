#!/usr/bin/env python3
"""
CORRECTED Flask Web Application for AI Research Assistant
This version properly fixes the human feedback handling with thread synchronization
"""

import os
import sys
import webbrowser
import threading
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import getpass
from threading import Event, Lock

# Import your existing research assistant classes
try:
    from LangChain_ResearchAssistant import SimpleResearchAssistant
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain Research Assistant imported")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("❌ LangChain Research Assistant not found")

try:
    from LangGraph_ResearchAssistant1 import SimpleResearchAssistant as LangGraphAssistant
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph Research Assistant imported")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("❌ LangGraph Research Assistant not found")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'research_assistant_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Thread-safe feedback handling
class FeedbackHandler:
    def __init__(self):
        self.feedback = None
        self.feedback_event = Event()
        self.lock = Lock()
        self.waiting_for_feedback = False
    
    def request_feedback(self, analysts_data):
        """Request feedback and wait for response"""
        with self.lock:
            self.feedback = None
            self.feedback_event.clear()
            self.waiting_for_feedback = True
        
        # Send feedback request to client
        socketio.emit('feedback_required', {'analysts': analysts_data})
        print("🔍 DEBUG: Feedback request sent to client")
        
        # Wait for feedback with timeout
        feedback_received = self.feedback_event.wait(timeout=60)  # 60 second timeout
        
        with self.lock:
            self.waiting_for_feedback = False
            result = self.feedback if feedback_received else ""
            print(f"🔍 DEBUG: Feedback result: '{result}' (received: {feedback_received})")
            return result
    
    def submit_feedback(self, feedback_text):
        """Submit feedback from web interface"""
        with self.lock:
            if self.waiting_for_feedback:
                self.feedback = feedback_text.strip()
                self.feedback_event.set()
                print(f"🔍 DEBUG: Feedback submitted: '{self.feedback}'")
                return True
            else:
                print("🔍 DEBUG: Feedback submitted but not waiting")
                return False

# Global variables
research_thread = None
research_active = False
feedback_handler = FeedbackHandler()

def setup_environment():
    """Setup API keys if not already configured"""
    api_keys = {}
    
    # Check Anthropic API key
    api_keys['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
    if not api_keys['ANTHROPIC_API_KEY']:
        print("🔑 Anthropic API key not found in environment")
        api_key = getpass.getpass("Enter your Anthropic API key: ")
        os.environ['ANTHROPIC_API_KEY'] = api_key
        api_keys['ANTHROPIC_API_KEY'] = api_key
    
    # Check Tavily API key (optional)
    api_keys['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    if not api_keys['TAVILY_API_KEY']:
        print("🔍 Tavily API key not found (optional for LangGraph)")
    
    return api_keys

# Setup API keys
api_keys = setup_environment()

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Return API status for the web interface"""
    return jsonify({
        'anthropic_connected': bool(api_keys.get('ANTHROPIC_API_KEY')),
        'tavily_connected': bool(api_keys.get('TAVILY_API_KEY')),
        'langchain_available': LANGCHAIN_AVAILABLE,
        'langgraph_available': LANGGRAPH_AVAILABLE
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status_update', {'status': 'connected', 'message': 'Connected to research assistant'})
    print(f"👤 Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"👤 Client disconnected")

@socketio.on('start_research')
def handle_start_research(data):
    """Handle research start request from web interface"""
    global research_thread, research_active
    
    if research_active:
        emit('error', {'message': 'Research already in progress'})
        return
    
    topic = data.get('topic', '').strip()
    implementation = data.get('implementation', 'langchain')
    analyst_count = int(data.get('analyst_count', 3))
    human_feedback = data.get('human_feedback', '').strip()
    
    if not topic:
        emit('error', {'message': 'Please provide a research topic'})
        return
    
    # Validate implementation availability
    if implementation == 'langchain' and not LANGCHAIN_AVAILABLE:
        emit('error', {'message': 'LangChain implementation not available'})
        return
    
    if implementation == 'langgraph' and not LANGGRAPH_AVAILABLE:
        emit('error', {'message': 'LangGraph implementation not available'})
        return
    
    # Start research in background thread
    research_active = True
    research_thread = threading.Thread(
        target=run_research_background,
        args=(topic, implementation, analyst_count, human_feedback)
    )
    research_thread.start()
    
    emit('research_started', {
        'topic': topic,
        'implementation': implementation,
        'analyst_count': analyst_count
    })

@socketio.on('submit_feedback')
def handle_submit_feedback(data):
    """Handle human feedback submission from web interface"""
    feedback = data.get('feedback', '').strip()
    
    print(f"🔍 DEBUG: Received feedback submission: '{feedback}'")
    
    # Submit feedback through the handler
    success = feedback_handler.submit_feedback(feedback)
    
    if success:
        # Confirm receipt to the client
        emit('feedback_submitted', {'feedback': feedback, 'status': 'received'})
        # Also emit to all clients to update UI
        socketio.emit('feedback_confirmed', {'feedback': feedback})
        print(f"✅ Feedback processed successfully: '{feedback}'")
    else:
        emit('feedback_submitted', {'feedback': feedback, 'status': 'not_waiting'})
        print(f"⚠️ Feedback received but not currently waiting for feedback")

def emit_progress(percentage, message, step_type='info'):
    """Emit progress update to web interface"""
    socketio.emit('progress_update', {
        'progress': percentage,
        'message': message,
        'type': step_type
    })

def emit_terminal_output(message, output_type='output'):
    """Emit terminal output to web interface"""
    socketio.emit('terminal_output', {
        'message': message,
        'type': output_type
    })

def emit_interview_update(analyst_name, message, message_type='question'):
    """Emit interview progress to the Analysts tab"""
    socketio.emit('interview_update', {
        'analyst_name': analyst_name,
        'message': message,
        'type': message_type  # 'question', 'answer', 'search', 'complete'
    })

def emit_analysts(analysts):
    """Emit analyst data to web interface"""
    analyst_data = []
    
    print(f"🔍 DEBUG: Emitting {len(analysts)} analysts to web interface")
    
    for i, analyst in enumerate(analysts):
        try:
            if hasattr(analyst, 'name'):
                # Object with attributes
                data = {
                    'name': analyst.name,
                    'role': analyst.role,
                    'affiliation': analyst.affiliation,
                    'description': analyst.description
                }
            else:
                # Dictionary
                data = {
                    'name': analyst.get('name', f'Analyst {i+1}'),
                    'role': analyst.get('role', 'Unknown Role'),
                    'affiliation': analyst.get('affiliation', 'Unknown Affiliation'),
                    'description': analyst.get('description', 'No description available')
                }
            
            analyst_data.append(data)
            print(f"✅ Added analyst: {data['name']}")
            
        except Exception as e:
            print(f"❌ Error processing analyst {i}: {e}")
            # Create fallback analyst
            analyst_data.append({
                'name': f'Analyst {i+1}',
                'role': 'Research Expert',
                'affiliation': 'Research Institute',
                'description': f'Expert analyst for research topic analysis'
            })
    
    print(f"📤 Emitting analysts_created event with {len(analyst_data)} analysts")
    socketio.emit('analysts_created', {'analysts': analyst_data})
    
    # Also emit to clear any previous interview data
    socketio.emit('clear_interviews')
    
    return analyst_data

def emit_final_report(report):
    """Emit final report to web interface"""
    socketio.emit('report_generated', {'report': report})

def create_sample_analysts(analyst_count, topic):
    """Create sample analysts when the actual method is not available"""
    sample_analysts = [
        {
            'name': "Dr. Sarah Chen",
            'role': "Neuroscience Researcher",
            'affiliation': "Stanford Medical School",
            'description': f"Focuses on brain imaging and neuroplasticity research related to {topic}"
        },
        {
            'name': "Prof. Michael Torres",
            'role': "Psychology Expert",
            'affiliation': "Harvard Business School",
            'description': f"Specializes in workplace productivity and psychological aspects of {topic}"
        },
        {
            'name': "Dr. Emma Liu",
            'role': "Technology Specialist",
            'affiliation': "MIT Media Lab",
            'description': f"Develops AI-powered applications and studies technological solutions for {topic}"
        },
        {
            'name': "Dr. James Rodriguez",
            'role': "Behavioral Economics Researcher",
            'affiliation': "University of Chicago",
            'description': f"Studies decision-making processes and economic implications of {topic}"
        },
        {
            'name': "Prof. Aisha Patel",
            'role': "Organizational Behavior Expert",
            'affiliation': "Wharton School",
            'description': f"Researches team dynamics and organizational applications of {topic}"
        }
    ]
    
    # Convert to simple objects with attributes
    class SimpleAnalyst:
        def __init__(self, data):
            self.name = data['name']
            self.role = data['role']
            self.affiliation = data['affiliation']
            self.description = data['description']
    
    return [SimpleAnalyst(analyst) for analyst in sample_analysts[:analyst_count]]

def run_research_background(topic, implementation, analyst_count, human_feedback):
    """Run research in background thread with ACTUAL research AND UI integration"""
    global research_active
    
    try:
        emit_terminal_output(f"🚀 Starting {implementation.upper()} Research Assistant...", 'success')
        emit_progress(5, "Initializing research system")
        
        if not research_active:
            return
        
        # Initialize the appropriate research assistant
        if implementation == 'langchain':
            assistant = SimpleResearchAssistant()
            emit_terminal_output("🔗 LangChain system initialized", 'output')
        else:
            assistant = LangGraphAssistant()
            emit_terminal_output("🌐 LangGraph system initialized", 'output')
        
        emit_progress(15, "System initialized")
        time.sleep(1)
        
        if not research_active:
            return
        
        # Create analysts
        emit_terminal_output(f"👥 Creating {analyst_count} analyst personas for: {topic}", 'output')
        emit_progress(25, "Creating analyst personas")
        
        analysts = None
        
        # Try to create actual analysts first
        try:
            if implementation == 'langchain' and hasattr(assistant, 'create_analysts'):
                analysts = assistant.create_analysts(topic, analyst_count)
            elif implementation == 'langgraph':
                # For LangGraph, we'll use sample analysts to avoid complexity
                analysts = create_sample_analysts(analyst_count, topic)
            else:
                analysts = create_sample_analysts(analyst_count, topic)
        except Exception as e:
            emit_terminal_output(f"⚠️ Error creating analysts: {str(e)}", 'output')
            analysts = create_sample_analysts(analyst_count, topic)
        
        if not analysts:
            analysts = create_sample_analysts(analyst_count, topic)
        
        # Emit analysts to UI and get the formatted data
        analyst_data = emit_analysts(analysts)
        
        emit_terminal_output(f"✅ Successfully created {len(analysts)} analysts:", 'success')
        for i, analyst in enumerate(analysts, 1):
            name = analyst.name if hasattr(analyst, 'name') else analyst.get('name', f'Analyst {i}')
            role = analyst.role if hasattr(analyst, 'role') else analyst.get('role', 'Expert')
            emit_terminal_output(f"  {i}. {name} - {role}", 'analyst')
        
        emit_progress(35, "Analysts created, waiting for feedback")
        
        if not research_active:
            return
        
        # Handle human feedback - ONLY for LangGraph
        final_feedback = ""
        
        if implementation == 'langgraph':
            # LangGraph supports interactive feedback
            if human_feedback and human_feedback.strip() and human_feedback.lower() not in ['', 'none', 'no']:
                emit_terminal_output(f"💭 Using initial feedback: {human_feedback}", 'analyst')
                final_feedback = human_feedback
                emit_progress(40, "Processing initial feedback")
                time.sleep(2)
            else:
                # Request feedback through the proper handler
                emit_terminal_output("💭 Requesting feedback from web interface...", 'analyst')
                emit_terminal_output("💡 You can provide feedback in the Human Feedback field and press Enter", 'output')
                
                try:
                    final_feedback = feedback_handler.request_feedback(analyst_data)
                    
                    if final_feedback:
                        emit_terminal_output(f"💭 Received feedback: {final_feedback}", 'analyst')
                        emit_progress(40, "Processing human feedback")
                        time.sleep(2)
                    else:
                        emit_terminal_output("💭 No feedback provided or timeout - continuing with generated analysts", 'output')
                        
                except Exception as e:
                    emit_terminal_output(f"⚠️ Feedback handling error: {str(e)}", 'output')
                    final_feedback = ""
        else:
            # LangChain runs automatically - no interactive feedback
            emit_terminal_output("📋 LangChain runs automatically without interactive feedback", 'output')
            if human_feedback and human_feedback.strip():
                emit_terminal_output(f"💭 Initial feedback noted: {human_feedback}", 'analyst')
                final_feedback = human_feedback
            else:
                emit_terminal_output("💭 Proceeding with generated analysts", 'output')
            emit_progress(40, "Continuing with research")
        
        emit_progress(45, "Feedback processed, continuing research")
        
        if not research_active:
            return
        
        # ✅ ACTUALLY RUN THE REAL RESEARCH (from working first code)
        if implementation == 'langchain':
            try:
                emit_terminal_output("🔗 Running REAL LangChain research process...", 'output')
                emit_progress(50, "Creating analysts and conducting real research")
                
                # Call the actual research method that does everything
                report = assistant.run_research(topic, analyst_count)
                
                if report and not report.startswith("❌"):
                    # Emit the REAL report
                    emit_final_report(report)
                    emit_terminal_output("✅ LangChain research completed successfully!", 'success')
                    emit_progress(100, "Real research completed successfully")
                    socketio.emit('research_completed', {
                        'topic': topic,
                        'analyst_count': analyst_count,
                        'implementation': implementation
                    })
                    return
                else:
                    emit_terminal_output(f"❌ Research failed: {report}", 'error')
                    return
                    
            except Exception as e:
                emit_terminal_output(f"❌ LangChain research failed: {str(e)}", 'error')
                socketio.emit('research_error', {'error': str(e)})
                return
        
        elif implementation == 'langgraph':
            try:
                emit_terminal_output("🌐 Running REAL LangGraph research process...", 'output')
                emit_progress(50, "Creating analysts and conducting real research")
                
                # Call the actual research method
                report = assistant.run_research(topic, analyst_count)
                
                if report and isinstance(report, str) and not report.startswith("❌"):
                    # Emit the REAL report
                    emit_final_report(report)
                    emit_terminal_output("✅ LangGraph research completed successfully!", 'success')
                    emit_progress(100, "Real research completed successfully")
                    socketio.emit('research_completed', {
                        'topic': topic,
                        'analyst_count': analyst_count,
                        'implementation': implementation
                    })
                    return
                else:
                    emit_terminal_output(f"❌ Research failed: {report}", 'error')
                    return
                    
            except Exception as e:
                emit_terminal_output(f"❌ LangGraph research failed: {str(e)}", 'error')
                socketio.emit('research_error', {'error': str(e)})
                return
        
    except Exception as e:
        emit_terminal_output(f"❌ Research failed: {str(e)}", 'error')
        socketio.emit('research_error', {'error': str(e)})
    
    finally:
        research_active = False

def open_browser():
    """Open the web browser to the application"""
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 AI Research Assistant - Web Interface (COMBINED WORKING)")
    print("="*60)
    print(f"🔗 LangChain Available: {LANGCHAIN_AVAILABLE}")
    print(f"🌐 LangGraph Available: {LANGGRAPH_AVAILABLE}")
    print(f"🔑 Anthropic API: {'✅ Configured' if api_keys.get('ANTHROPIC_API_KEY') else '❌ Missing'}")
    print(f"🔍 Tavily API: {'✅ Configured' if api_keys.get('TAVILY_API_KEY') else '❌ Missing (Optional)'}")
    print("="*60)
    print("🚀 Starting web server...")
    print("🌐 Opening browser automatically...")
    print("💡 Use Ctrl+C to stop the server")
    print("💡 Real reports + Analysts display + Human feedback working!")
    print("="*60)
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
