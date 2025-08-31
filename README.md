# 🎙️ Complete Podcast Generator with Voice Cloning

A comprehensive system for generating podcasts using XTTS-v2 voice cloning technology. This system can create professional-quality podcasts with multiple speakers using AI-generated voice samples, integrated with RAG-based script generation and an intuitive interactive workflow.

## ✨ Features

- **🎭 Dynamic Voice Switching**: Intelligent speaker mapping with user-selected voice samples
- **🎨 Tone Control**: Professional, Academic, Sports Commentator, and Narrator styles
- **📚 Smart Document Selection**: Numbered list selection for RAG documents with range support
- **📁 Enhanced Script Management**: Numbered list selection for scripts with file info
- **🎙️ Voice Cloning**: Uses XTTS-v2 for high-quality voice replication
- **👥 Multiple Speakers**: Support for 1-5 speakers with custom voice assignments
- **🤖 Automatic Script Generation**: Creates podcast scripts based on topics using RAG and LLM
- **⚡ Batch Processing**: Generate multiple podcast segments efficiently
- **🎵 Voice Sample Management**: Built-in tools for creating and managing voice samples
- **📊 Quality Analysis**: Audio quality assessment and optimization
- **📝 Custom Scripts**: Support for custom podcast scripts and dialogue
- **🔍 RAG Integration**: Retrieval-augmented generation for context-aware content
- **🧠 LLM Integration**: Uses Ollama with Qwen 2.5 14B for script generation
- **🧹 Intelligent Script Cleanup**: Removes time annotations, section headers, and markdown
- **🎯 Interactive Workflow**: User-friendly guided interface for all operations

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Podcast-check

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Ollama (Required for Script Generation)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required model
ollama pull qwen2.5:14b-instruct
```

### 3. Basic Usage

#### **Interactive Mode (Recommended)**
```bash
# Start the guided interactive workflow
python main.py --interactive
```

#### **Command Line Mode**
```bash
# Check system status
python main.py --status

# List available voice samples
python main.py --list-voices

# List available prompts
python main.py --list-prompts

# Generate audio from the latest script
python main.py --generate-audio
```

## 🎯 Complete Command Reference

### **Interactive Workflow (Recommended)**
```bash
# Start the guided interactive workflow
python main.py --interactive
```

**Interactive Workflow Features:**
- **Script Availability Check**: Choose between existing scripts or new generation
- **Smart Document Selection**: Numbered list with range support (e.g., "1-3")
- **Enhanced Script Selection**: Numbered list with file info for existing scripts
- **Comprehensive Preferences**: Speed, Energy, Tone, Duration, Speaker count
- **Voice Sample Assignment**: Select specific voice files for each speaker
- **Starting Speaker Control**: Choose who begins the conversation
- **Real-time Validation**: Immediate feedback on all inputs
- **Summary & Confirmation**: Review all selections before execution

### **System Information Commands**
```bash
# Check complete system status
python main.py --status

# List available voice samples with quality scores
python main.py --list-voices

# List available prompt templates
python main.py --list-prompts
```

### **Voice Sample Management**
```bash
# Create default voice samples (first time setup)
python main.py --create-voice-samples

# List available voice samples
python main.py --list-voices
```

### **Script Generation**
```bash
# Process dataset and build RAG index (first time)
python main.py --process-dataset


# Reprocess dataset and rebuild index
python main.py --reprocess
```

### **Audio Generation**
```bash
# Generate audio from latest script (auto-selects voice)
python main.py --generate-audio

# Generate audio with specific voice
python main.py --generate-audio --voice male_reporter_default
python main.py --generate-audio --voice female_reporter_default
python main.py --generate-audio --voice neutral_default
```

### **Complete Workflow Example**
```bash
# Step 1: Generate script
python main.py --generate long-desc --length medium

# Step 2: Generate audio from the script
python main.py --generate-audio

# Step 3: Check system status
python main.py --status
```

## 🏗️ System Architecture

### Core Components

1. **Interactive Workflow** (`src/interactive_workflow.py`)
   - **User Experience**: Guided, numbered selection interface
   - **Input Validation**: Real-time validation with retry logic
   - **Workflow Orchestration**: Seamless integration of all components
   - **Metadata Management**: JSON generation for debugging and audit

2. **Voice Sample Manager** (`src/voice_sample_manager.py`)
   - Creates and manages voice samples
   - Supports XTTS-v2 voice cloning
   - Validates audio quality
   - Provides sample optimization

3. **XTTS Integration** (`src/xtts_integration.py`)
   - Interfaces with XTTS-v2 model
   - Handles voice cloning operations
   - Manages audio generation with voice switching
   - Provides quality analysis and post-processing

4. **Complete Lightweight System** (`src/complete_lightweight_system.py`)
   - Orchestrates RAG/LLM integration
   - Generates scripts with user preferences
   - Manages content optimization and validation
   - Handles speech rate analysis and content planning

5. **LLM Interface** (`src/llm_interface.py`)
   - Manages Ollama integration
   - Generates conversational scripts
   - Handles script formatting and cleanup
   - Ensures alternating dialogue patterns

6. **Main Application** (`main.py`)
   - Unified command-line interface
   - Integrates all system components
   - Provides interactive and command-line modes
   - Automatic file selection and processing

## 🎙️ Voice Sample Capabilities

### **Voice Sample Naming Convention**
- **Format**: `<gender>_<name>.wav` (e.g., `male_john.wav`, `female_sarah.wav`)
- **Automatic Detection**: System automatically detects male/female voices by prefix
- **Flexible Assignment**: Map any voice sample to any speaker role

### **Built-in Voice Types**
- **Male Voices**: Professional male voices for hosting and reporting
- **Female Voices**: Professional female voices for hosting and reporting
- **Custom Types**: User-defined voice characteristics and samples

### **Voice Sample Sources**
1. **XTTS-v2 Voice Cloning**: High-quality voice replication from audio samples
2. **Online Sources**: Freesound, Internet Archive, LibriVox, Common Voice
3. **Custom Creation**: Specific text-to-speech for unique voices
4. **Quality Validation**: Ensures samples meet XTTS-v2 requirements

### **Quality Features**
- **Duration Optimization**: 3-30 second ideal range
- **Audio Analysis**: Energy, zero-crossing rate, quality scoring
- **Automatic Validation**: Ensures samples are suitable for voice cloning
- **Optimization Tools**: Noise reduction and audio enhancement

## 📝 Script Generation Features

### **RAG-Based Content**
- **Context-Aware**: Uses your PDF documents for relevant information
- **Intelligent Retrieval**: Finds most relevant content for scripts
- **Professional Quality**: Generates natural dialogue between speakers

### **Enhanced User Preferences**
- **Speed Control**: Slow (0.8x), Normal (1.0x), Fast (1.3x), Very Fast (1.5x)
- **Energy Levels**: Low (calm/analytical), Medium (balanced/professional), High (exciting/dynamic)
- **Tone Styles**: Professional, Academic, Sports Commentator, Narrator
- **Duration Options**: Summary (2-3 min), Short (3-5 min), Medium (5-8 min), Long (9-11 min), Detailed (12+ min)
- **Speaker Control**: 1-5 speakers with custom voice assignments
- **Starting Speaker**: Choose who begins the conversation

### **Intelligent Content Generation**
- **Speech Rate Analysis**: Analyzes voice samples to determine optimal content length
- **Content Planning**: Generates target word counts based on actual speech speed
- **Iterative Optimization**: Multiple generation passes with smart content adjustment
- **Quality Validation**: Ensures content meets duration requirements naturally


### **Script Types**
- **Conversational**: Natural dialogue between multiple speakers
- **Monologue**: Single speaker format
- **Discussion**: Multiple speaker format with alternating dialogue
- **Custom**: User-defined structure and style

## 🔧 Configuration

### **Environment Setup**
```bash
# Required: Ollama service running
ollama serve

# Required: Qwen 2.5 14B model
ollama pull qwen2.5:14b-instruct

# Optional: Custom voice samples
python main.py --create-voice-samples
```

### **Voice Sample Requirements**
- **Format**: WAV files (recommended for XTTS-v2)
- **Naming**: Use `<gender>_<name>.wav` format for automatic detection
- **Duration**: 3-30 seconds (optimal for voice cloning)
- **Quality**: Clear speech, minimal background noise
- **Location**: Store in `voice_samples/` directory

### **Voice Sample Settings**
- **Sample Rate**: 22050 Hz (optimized for XTTS-v2)
- **Format**: WAV (preferred by XTTS-v2)
- **Duration**: 3-30 seconds (ideal range)
- **Quality**: High energy, clear speech

### **XTTS-v2 Parameters**
- **Model**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Language**: English (default)
- **Speed**: Adjustable (0.5x to 2.0x)
- **Output**: High-quality WAV files

### **Intelligent Script Cleanup & Review Workflow**
- **Time Annotations**: Removes `[Time: X minutes]` and similar markers
- **Section Headers**: Removes `**Introduction:**`, `**Background:**` etc.
- **Markdown Formatting**: Cleans `**bold**` and `*italic*` while preserving content
- **Section Dividers**: Removes `---` separators
- **Meta-commentary**: Removes end-of-script notes and explanations
- **Speaker Labels**: Preserves and formats "Person 1:" and "Person 2:" labels
- **File Saving**: Saves cleaned scripts to `output/cleanup-scripts/` with timestamp
- **User Review**: Allows users to edit cleaned scripts before TTS generation
- **Smart Loading**: Automatically loads updated files or uses memory version based on user choice

## 📁 Output Structure

```
Podcast-check/
├── output/                    # Generated scripts (.txt files)
│   ├── scripts/              # Original generated scripts
│   └── cleanup-scripts/      # Cleaned scripts for user review
├── audio_output/             # Generated audio files (.wav)
├── voice_samples/            # Voice samples for cloning
├── dataset/                  # PDF documents for RAG
├── prompts/                  # Prompt templates
├── logs/                     # Detailed operation logs
└── generated_podcasts/       # Complete podcast outputs
```

## 🎯 Interactive Workflow Guide

### **Getting Started**
```bash
python main.py --interactive
```

### **Workflow Steps**

#### **Step 1: Script Availability**
- **Option A**: Use existing script from `output/scripts/`
- **Option B**: Generate new script with RAG documents

#### **Step 2A: Existing Script Path**
- Select script by number from numbered list
- Shows file size and modification date
- Automatic validation and confirmation

#### **Step 2B: New Script Generation**
- **Document Selection**: Choose from `dataset/` folder
  - Numbered list with range support (e.g., "1-3")
  - Ignores subdirectories
  - Immediate validation
- **Prompt Selection**: Choose from available prompts
  - Numbered list selection
  - Automatic loading and confirmation

#### **Step 3: User Preferences**
1. **Speed**: Slow (0.8x) to Very Fast (1.5x)
2. **Energy**: Low to High
3. **Tone**: Professional, Academic, Sports Commentator, Narrator
4. **Duration**: Short (2-3 min) to Long (10-15 min)
5. **Speakers**: 1-5 speakers
6. **Voice Assignment**: Select specific voice files for each speaker
7. **Starting Speaker**: Choose who begins the conversation

#### **Step 4: Summary & Confirmation**
- Review all selections
- Confirm or abort execution
- Metadata saved to JSON for debugging

#### **Step 5: Script Cleanup & Review**
- **Automatic Cleanup**: Removes time annotations, section headers, markdown
- **File Saving**: Saves cleaned script to `output/cleanup-scripts/`
- **User Review**: Edit script if needed, then confirm to proceed
- **Smart Loading**: Loads updated file or uses memory version

#### **Step 6: Execution**
- **Script Generation**: RAG + LLM with preferences
- **Audio Generation**: XTTS-v2 with voice switching
- **Output**: Clean script + continuous audio file

### **Voice Types & Assignment**
- **Flexible Mapping**: Assign any voice sample to any speaker role
- **Gender Detection**: Automatic male/female voice detection by filename
- **Custom Samples**: Add your own voice samples in WAV format
- **Voice Sharing**: System informs when multiple speakers share voices

### **Script Generation & Style**
- **Tone Control**: Professional, Academic, Sports Commentator, Narrator
- **Speed & Energy**: Fine-tune conversation pace and enthusiasm
- **Duration Planning**: Intelligent content generation based on speech rate
- **Speaker Control**: Choose starting speaker and voice assignments

### **Audio Quality & Processing**
- **Voice Cloning**: High-quality XTTS-v2 voice replication
- **Post-processing**: Speed adjustment and energy enhancement
- **Continuous Output**: Single audio file with seamless voice switching
- **Format Optimization**: WAV format with optimal settings for XTTS-v2

## Troubleshooting

### Common Issues

1. **Ollama Not Running**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Check status
   ollama list
   ```

2. **Model Not Available**
   ```bash
   # Pull required model
   ollama pull qwen2.5:14b-instruct
   ```

3. **Voice Samples Missing**
   ```bash
   # Create default samples
   python main.py --create-voice-samples
   
   # Check samples
   python main.py --list-voices
   ```

4. **XTTS-v2 Initialization Failed**
   - Check TTS installation: `pip install TTS==0.21.0`
   - Verify PyTorch version: `torch==2.2.0`
   - Check Transformers version: `transformers==4.49.0`
   - Verify model availability
   - Check system resources

### Interactive Workflow Issues

5. **Voice Switching Not Working**
   - Ensure voice samples use correct naming: `male_*.wav`, `female_*.wav`
   - Check that voice samples are valid WAV files
   - Verify speaker mapping in generated JSON metadata

6. **Script Cleanup Issues**
   - Check that time annotations are properly removed
   - Verify speaker labels are preserved as "Person 1:" and "Person 2:"
   - Review script before and after cleanup in logs

7. **Content Length Issues**
   - Speech rate analysis may need recalibration
   - Check voice sample quality and duration
   - Verify target duration settings in preferences

### Performance Tips

- **Batch Processing**: Generate multiple segments together
- **Sample Optimization**: Use optimized voice samples
- **Resource Management**: Monitor memory usage during generation

##  Future Enhancements

- **Real-time Recording**: Microphone input support
- **Video Integration**: Video podcast generation
- **Advanced Scripting**: AI-powered script generation
- **Voice Training**: Custom voice model training
- **Streaming Support**: Live podcast generation
- **Multi-language**: Support for multiple languages

### Main Application

```bash
python main.py [OPTIONS]

Options:
  --interactive           Start interactive workflow (RECOMMENDED)
  --status               Show system status
  --list-voices          List available voice samples
  --list-prompts         List available prompts
  --create-voice-samples Create default voice samples
  --process-dataset      Process PDF dataset
  --generate PROMPT      Generate script using prompt
  --length {short,medium,long}  Script length
  --generate-audio       Generate audio from latest script
  --voice VOICE          Specify voice type for audio
  --reprocess           Reprocess dataset
```

### Interactive Workflow

```python
class InteractiveWorkflow:
    def __init__(self, system)
    def run_workflow()
    def _collect_user_preferences()
    def _collect_speaker_assignments()
    def _collect_starting_speaker()
    def _collect_tone_preference()
    def _execute_new_script_workflow()
    def _execute_existing_script_workflow()
```

### XTTS Integration

```python
class XTTSIntegration:
    def __init__(voice_samples_dir)
    def initialize_xtts(model_name)
    def clone_voice(text, voice_sample_path, output_path)
    def create_podcast_segment(script_segment, voice_type, output_dir)
    def batch_generate(script_segments, output_dir)
    def create_continuous_podcast_audio(script_path, output_path, speaker_mapping)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- **XTTS-v2**: Coqui AI for the voice cloning technology
- **SoundFile**: Audio file I/O library
- **Ollama**: Local LLM inference framework
- **Qwen 2.5**: Alibaba Cloud for the language model

---

**Happy Podcasting! 🎙️✨** 
