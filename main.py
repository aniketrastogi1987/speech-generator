#!/usr/bin/env python3
"""
Podcast Script Generation System with TTS Integration

Main application entry point for generating podcast scripts using RAG, LLM, and TTS voice cloning.
"""

import sys
import argparse
import os
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from complete_lightweight_system import CompleteLightweightSystem

# Import TTS components
from voice_sample_manager import VoiceSampleManager
from src.xtts_integration import XTTSIntegration
from complete_podcast_generator import CompletePodcastGenerator


def setup_logging():
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/podcast_generator.log",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - <level>{message}</level>",
        level="DEBUG"
    )


def get_latest_output_file(output_dir):
    """Get the latest output file from the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    # First try to look in the scripts subdirectory
    scripts_dir = output_path / "scripts"
    if scripts_dir.exists():
        text_files = list(scripts_dir.glob("*.txt"))
        if text_files:
            # Return the most recently modified file from scripts directory
            latest_file = max(text_files, key=lambda x: x.stat().st_mtime)
            return latest_file
    
    # Fallback to main output directory
    text_files = list(output_path.glob("*.txt"))
    if not text_files:
        return None
    
    # Return the most recently modified file
    latest_file = max(text_files, key=lambda x: x.stat().st_mtime)
    return latest_file


def collect_script_preferences() -> dict:
    """Collect user preferences for script generation with speech rate optimization."""
    print("\n🎭 === PODCAST SCRIPT CUSTOMIZATION === 🎭")
    print("Let's customize your podcast script before generation!")
    
    # 1. Speed of conversation
    print("\n1️⃣ SPEED OF CONVERSATION:")
    print("   • Slow (0.8x): Detailed analysis, complex topics")
    print("   • Normal (1.0x): Balanced pace, general discussion")
    print("   • Fast (1.3x): Quick insights, breaking news")
    print("   • Very Fast (1.5x): High-energy, rapid-fire discussion")
    
    speed_input = input("Select speed [slow/normal/fast/very_fast] (default: fast): ").strip().lower()
    speed_mapping = {
        'slow': 0.8, 'normal': 1.0, 'fast': 1.3, 'very_fast': 1.5
    }
    speed_factor = speed_mapping.get(speed_input, 1.3)
    speed_name = speed_input if speed_input in speed_mapping else 'fast'
    
    # 2. Energy level
    print("\n2️⃣ ENERGY LEVEL:")
    print("   • Low: Calm, serious, analytical")
    print("   • Medium: Balanced, professional, engaging")
    print("   • High: Exciting, dynamic, passionate")
    
    energy_input = input("Select energy [low/medium/high] (default: high): ").strip().lower()
    energy_level = energy_input if energy_input in ['low', 'medium', 'high'] else 'high'
    energy_name = energy_level
    
    # 3. Duration of conversation
    print("\n3️⃣ DURATION OF CONVERSATION:")
    print("   • Summary: 2-3 minutes, key points only")
    print("   • Short: 3-5 minutes, brief overview")
    print("   • Medium: 5-8 minutes, balanced coverage")
    print("   • Long: 9-11 minutes, comprehensive discussion")
    print("   • Detailed: More than 12 minutes, in-depth analysis")
    
    duration_input = input("Select duration [summary/short/medium/long/detailed] (default: medium): ").strip().lower()
    conversation_length = duration_input if duration_input in ['summary', 'short', 'medium', 'long', 'detailed'] else 'medium'
    duration_name = conversation_length
    
    # 4. Voice selection
    print("\n4️⃣ VOICE SELECTION:")
    print("   • Male: Single male reporter")
    print("   • Female: Single female reporter")
    print("   • Both: Interactive dialogue between male and female reporters")
    
    voice_input = input("Select voice [male/female/both] (default: both): ").strip().lower()
    voice_type = voice_input if voice_input in ['male', 'female', 'both'] else 'both'
    
    # 5. Speech rate optimization
    print("\n5️⃣ SPEECH RATE OPTIMIZATION:")
    print("   • System will analyze your voice samples")
    print("   • Generate optimal content length for target duration")
    print("   • Ensure natural pacing and engagement")
    
    # Create preferences dictionary
    preferences = {
        'speed_factor': speed_factor,
        'energy_level': energy_level,
        'conversation_length': conversation_length,
        'voice_type': voice_type,
        'speed_name': speed_name,
        'energy_name': energy_name,
        'duration_name': duration_name
    }
    
    # Display selected preferences
    print("\n🎯 === YOUR SCRIPT PREFERENCES === 🎯")
    print(f"   Speed: {speed_name.title()} ({speed_factor}x)")
    print(f"   Energy: {energy_level.title()}")
    print(f"   Duration: {duration_name.title()}")
    print(f"   Voice: {voice_type.title()}")
    
    # Confirm preferences
    confirm = input("\n✅ Generate script with these preferences? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("❌ Script generation cancelled.")
        return None
    
    return preferences

def analyze_speech_rate_for_preferences(preferences: dict) -> dict:
    """
    Analyze speech rate and generate content planning guide for the given preferences.
    
    Args:
        preferences: User preferences including duration
        
    Returns:
        Enhanced preferences with content guide
    """
    try:
        import sys
        sys.path.append('.')  # Add current directory to path
        try:
            from speech_rate_analyzer import SpeechRateAnalyzer
        except ImportError as e:
            logger.error(f"Failed to import speech rate analyzer: {e}")
            return preferences
        
        # Extract target duration
        duration_mapping = {
            'summary': 2.5, 
            'short': 4.0, 
            'medium': 6.5, 
            'long': 10.0, 
            'detailed': 15.0
        }
        target_duration = duration_mapping.get(preferences.get('conversation_length', 'medium'), 6.5)
        
        logger.info(f"🔍 Analyzing speech rate for {target_duration} minute target...")
        
        # Initialize speech rate analyzer
        analyzer = SpeechRateAnalyzer()
        
        # Analyze voice samples and generate content guide
        content_guide = analyzer.generate_content_planning_guide(target_duration)
        
        if content_guide:
            logger.info(f"✅ Speech rate analysis complete:")
            logger.info(f"   Target words: {content_guide['target_word_count']:,}")
            logger.info(f"   Speech rate: {content_guide['speech_rate_analysis']['average_wpm']:.1f} WPM")
            
            # Add content guide to preferences
            preferences['content_guide'] = content_guide
            preferences['target_duration_minutes'] = target_duration
            
            return preferences
        else:
            logger.warning("⚠️ Speech rate analysis failed - using default settings")
            return preferences
            
    except ImportError:
        logger.warning("⚠️ Speech rate analyzer not available - using default settings")
        return preferences
    except Exception as e:
        logger.error(f"Speech rate analysis error: {e}")
        return preferences


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Podcast Script Generation System with TTS Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive workflow (recommended for new users)
  python main.py --interactive
  
  
  # Generate audio from latest script
  python main.py --generate-audio
  
  # Generate audio with specific voice
  python main.py --generate-audio --voice male_reporter_default
  
  # Generate audio with custom speed and energy
  python main.py --generate-audio --speed 1.5 --energy high
  
  # Create voice samples
  python main.py --create-voice-samples
  
  # Check system status
  python main.py --status
  
  # List available prompts
  python main.py --list-prompts
  
  # List available voices
  python main.py --list-voices
        """
    )
    
    parser.add_argument(
        "--process-dataset",
        action="store_true",
        help="Process the dataset and build RAG index"
    )
    
    parser.add_argument(
        "--generate",
        type=str,
        metavar="PROMPT_NAME",
        help="Generate script using specified prompt"
    )
    
    parser.add_argument(
        "--length",
        choices=["summary", "short", "medium", "long", "detailed"],
        default="medium",
        help="Conversation length: summary (2-3 min), short (3-5 min), medium (5-8 min), long (9-11 min), detailed (12+ min) (default: medium)"
    )
    
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate audio from the latest generated script"
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        default="auto",
        help="Voice type for audio generation (default: auto-select)"
    )
    
    parser.add_argument(
        "--speed",
        type=float,
        default=1.3,
        help="Audio speed factor (1.0 = normal, 1.3 = fast, 1.5 = very fast)"
    )
    
    parser.add_argument(
        "--energy",
        choices=["low", "medium", "high"],
        default="high",
        help="Audio energy level (default: high for podcast-style)"
    )
    
    parser.add_argument(
        "--create-voice-samples",
        action="store_true",
        help="Create default voice samples for TTS"
    )
    
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voice samples"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive workflow for podcast generation"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status"
    )
    
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompts"
    )
    
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess dataset and rebuild RAG index"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated scripts (default: output)"
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Dataset directory containing PDFs (default: dataset)"
    )
    
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="prompts",
        help="Prompts directory (default: prompts)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Initialize script generator
        logger.info("Initializing Podcast Script Generation System...")
        generator = CompleteLightweightSystem(
            dataset_folder=args.dataset_dir,
            prompts_folder=args.prompts_dir,
            output_folder=args.output_dir
        )
        
        # Initialize TTS components
        logger.info("Initializing TTS components...")
        voice_manager = VoiceSampleManager()
        xtts = XTTSIntegration(voice_manager=voice_manager)
        
        # Process dataset if requested
        if args.process_dataset:
            logger.info("Processing dataset...")
            try:
                # Use smaller chunk sizes and batch processing for memory efficiency
                success = generator.process_dataset(pdf_chunk_size=5000, text_chunk_size=500, overlap=100)
                if success:
                    logger.info("Dataset processed successfully!")
                else:
                    logger.error("Failed to process dataset")
                    return 1
            except KeyboardInterrupt:
                logger.info("Dataset processing interrupted by user")
                return 1
            except Exception as e:
                logger.error(f"Unexpected error during dataset processing: {e}")
                return 1
        
        # Reprocess dataset if requested
        if args.reprocess:
            logger.info("Reprocessing dataset...")
            success = generator.reprocess_dataset()
            if success:
                logger.info("Dataset reprocessed successfully!")
            else:
                logger.error("Failed to reprocess dataset")
                return 1
        
        # Create voice samples if requested
        if args.create_voice_samples:
            logger.info("Creating default voice samples...")
            try:
                samples = voice_manager.create_default_samples()
                logger.info(f"Created {len(samples)} voice samples")
                print(f"✅ Created {len(samples)} voice samples:")
                for voice_type, path in samples.items():
                    print(f"  • {voice_type}: {path}")
            except Exception as e:
                logger.error(f"Failed to create voice samples: {e}")
                return 1
        
        # List available voices
        if args.list_voices:
            voices = voice_manager.get_available_samples()
            print("\n=== Available Voice Samples ===")
            if voices:
                for name, path in voices.items():
                    info = voice_manager.get_sample_info(path)
                    if "error" not in info:
                        print(f"  • {name}: {info['duration']:.1f}s, Quality: {info['quality_score']:.2f}")
                    else:
                        print(f"  • {name}: {path}")
            else:
                print("  No voice samples found. Use --create-voice-samples to create them.")
            print()
        
        # Show system status
        if args.status:
            try:
                status = generator.get_system_status()
                print("\n=== System Status ===")
                
                # Handle the actual structure returned by the lightweight system
                if 'error' in status:
                    print(f"❌ System Error: {status['error']}")
                    return 1
                
                # Extract information from the correct structure
                dataset_info = status.get('dataset', {})
                rag_info = status.get('rag', {})
                output_info = status.get('output', {})
                components_info = status.get('components', {})
                
                print(f"System: {status.get('system', 'Unknown')}")
                print(f"Dataset Folder: {dataset_info.get('folder', 'Unknown')}")
                print(f"PDFs Found: {dataset_info.get('pdf_count', 'Unknown')}")
                print(f"RAG Documents: {rag_info.get('document_count', 'Unknown')}")
                print(f"Output Folder: {output_info.get('folder', 'Unknown')}")
                
                # Component status
                print(f"\nComponent Status:")
                for component, status_text in components_info.items():
                    print(f"  {component.replace('_', ' ').title()}: {status_text}")
                
                # TTS Status
                voices = voice_manager.get_available_samples()
                print(f"\nVoice Samples: {len(voices)} available")
                print(f"TTS Status: {'Ready' if voices else 'No voice samples'}")
                
                # Show TTS model info if available
                try:
                    tts_info = xtts.get_model_info()
                    if 'error' not in tts_info:
                        print(f"TTS Model: {tts_info.get('model_type', 'Unknown')}")
                        print(f"TTS Status: {tts_info.get('status', 'Unknown')}")
                except Exception as e:
                    print(f"TTS Info: Could not retrieve (Error: {e})")
                
                print()
                
            except Exception as e:
                logger.error(f"Failed to get system status: {e}")
                print(f"\n❌ Error getting system status: {e}")
                print("Some components may not be fully initialized.")
                print()
        
        # List available prompts
        if args.list_prompts:
            prompts = generator.get_available_prompts()
            print("\n=== Available Prompts ===")
            if prompts:
                for prompt in prompts:
                    print(f"  • {prompt}")
            else:
                print("  No prompts found")
            print()
        
        # Generate script if requested
        if args.generate:
            logger.info(f"Generating script using prompt: {args.generate}")
            
            # Check if prompt exists
            available_prompts = generator.get_available_prompts()
            if args.generate not in available_prompts:
                logger.error(f"Prompt '{args.generate}' not found. Available prompts: {available_prompts}")
                return 1
            
            # Collect user preferences for script customization
            preferences = collect_script_preferences()
            if not preferences:
                return 1
            
            # STEP 1: Analyze speech rate FIRST to determine content requirements
            logger.info("🔍 Step 1: Analyzing speech rate for content planning...")
            enhanced_preferences = analyze_speech_rate_for_preferences(preferences)
            
            if not enhanced_preferences:
                logger.error("Failed to analyze speech rate")
                return 1
            
            # STEP 2: Generate script with speech rate optimization
            logger.info("🎭 Step 2: Generating script with speech rate optimization...")
            script_path = generator.generateAndSaveWithPreferences(args.generate, enhanced_preferences)
            
            if script_path:
                logger.info(f"Script generated and saved to: {script_path}")
                
                # Display success message with speech rate analysis results
                print("\n✓ Script generated successfully with your preferences!")
                print(f"📁 Saved to: {script_path}")
                print("🎯 Preferences applied:")
                print(f"   • Speed: {preferences['speed_name'].title()} ({preferences['speed_factor']}x)")
                print(f"   • Energy: {preferences['energy_level'].title()}")
                print(f"   • Duration: {preferences['duration_name'].title()}")
                print(f"   • Voice: {preferences['voice_type'].title()}")
                
                # Display speech rate analysis results
                if 'content_guide' in enhanced_preferences:
                    content_guide = enhanced_preferences['content_guide']
                    print(f"🎯 Speech Rate Analysis Results:")
                    print(f"   • Measured Speech Rate: {content_guide['speech_rate_analysis']['average_wpm']:.1f} WPM")
                    print(f"   • Target Word Count: {content_guide['target_word_count']:,} words")
                    print(f"   • Target Duration: {content_guide['target_duration_minutes']:.1f} minutes")
                    print(f"   • Content Multiplier: {content_guide['content_multiplier']:.2f}x")
                    
                    # Show section breakdown
                    print(f"📝 Content Structure:")
                    for section_name, section_info in content_guide['section_breakdown'].items():
                        print(f"   • {section_name.replace('_', ' ').title()}: {section_info['word_count']} words")
                else:
                    print("   • Content Guide: Using default settings (speech rate analysis not available)")
            else:
                logger.error("Failed to generate script")
                return 1
        
        # Generate audio if requested
        if args.generate_audio:
            logger.info("Generating audio from latest script...")
            
            # Get the latest output file
            latest_file = get_latest_output_file(args.output_dir)
            if not latest_file:
                logger.error("No output files found. Generate a script first using --generate")
                return 1
            
            logger.info(f"Using script file: {latest_file}")
            
            # Check if voice samples exist
            voices = voice_manager.get_available_samples()
            if not voices:
                logger.error("No voice samples found. Create them first using --create-voice-samples")
                return 1
            
            # Initialize TTS - XTTS-v2 is required for voice cloning
            import platform
            if platform.system() == "Darwin" and platform.machine().startswith("arm"):
                logger.info("🍎 Detected Apple Silicon MacBook. Initializing XTTS-v2...")
                if not xtts.initialize_macbook_tts_fallback():
                    logger.error("❌ XTTS-v2 initialization failed. Voice cloning is required.")
                    return 1
                use_macbook_tts = True
                logger.info("✅ XTTS-v2 initialized successfully")
            else:
                # Try XTTS-v2 on other systems
                if not xtts.initialize_xtts():
                    logger.warning("Failed to initialize XTTS-v2. Trying alternative initialization...")
                    
                    # Try alternative XTTS-v2 initialization
                    if not xtts.initialize_macbook_tts_fallback():
                        logger.error("❌ XTTS-v2 initialization failed. Voice cloning is required.")
                        return 1
                    else:
                        logger.info("✅ XTTS-v2 initialized successfully")
                        use_macbook_tts = True
                else:
                    use_macbook_tts = False
                    logger.info("✅ XTTS-v2 initialized successfully")
            
            # Read the script content
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                
                # Try to read metadata to get user preferences
                metadata_file = str(latest_file).replace('.txt', '.json')
                user_preferences = None
                
                if os.path.exists(metadata_file):
                    try:
                        import json
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            user_preferences = metadata.get('preferences', {})
                            logger.info(f"📋 Loaded user preferences: {user_preferences}")
                    except Exception as e:
                        logger.warning(f"Failed to load metadata: {e}")
                
                # Use user preferences or defaults
                speed_factor = user_preferences.get('speed_factor', 1.3) if user_preferences else 1.3
                energy_level = user_preferences.get('energy_name', 'high') if user_preferences else 'high'
                
                logger.info(f"⚡ Using speed factor: {speed_factor}x")
                logger.info(f"🔥 Using energy level: {energy_level}")
                
                # Clean the script for audio generation (remove time annotations and speaker labels)
                logger.info("🧹 Cleaning script for audio generation...")
                from src.llm_interface import LLMInterface
                llm_interface = LLMInterface()
                cleaned_script, cleaned_filename = llm_interface.clean_script_for_audio(script_content)
                
                logger.info(f"Original script length: {len(script_content)} characters")
                logger.info(f"Cleaned script length: {len(cleaned_script)} characters")
                
                if cleaned_filename:
                    logger.info(f"📁 Cleaned script saved to: output/cleanup-scripts/{cleaned_filename}")
                    
                    # Prompt user to review the cleaned script
                    print("\n📝 SCRIPT REVIEW STEP:")
                    print("The cleaned script has been saved to a file.")
                    print("You can now review and edit the script if needed.")
                    
                    proceed = input("Proceed with TTS generation? (y/n): ").strip().lower()
                    if proceed != 'y':
                        updates_made = input("Confirm if updates were made to the file? (y/n): ").strip().lower()
                        if updates_made == 'y':
                            # Load the updated file from disk
                            try:
                                updated_filepath = f"output/cleanup-scripts/{cleaned_filename}"
                                with open(updated_filepath, 'r', encoding='utf-8') as f:
                                    cleaned_script = f.read()
                                logger.info(f"✅ Loaded updated script from file: {len(cleaned_script)} characters")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to load updated file, using cleaned script from memory: {e}")
                        else:
                            logger.info("✅ Using cleaned script from memory")
                    else:
                        logger.info("✅ Proceeding with TTS generation using cleaned script")
                
                # Option to preview cleaned script before audio generation
                preview_choice = input("\n🔍 Preview cleaned script before audio generation? [y/N]: ").strip().lower()
                if preview_choice in ['y', 'yes']:
                    print("\n📝 === CLEANED SCRIPT FOR AUDIO ===")
                    print("(Time annotations and speaker labels removed)")
                    print("=" * 50)
                    print(cleaned_script)
                    print("\n" + "=" * 50)
                    
                    continue_choice = input("\n✅ Continue with audio generation? [Y/n]: ").strip().lower()
                    if continue_choice in ['n', 'no']:
                        print("❌ Audio generation cancelled.")
                        return 1
                
                # Determine voice type and apply script preferences
                if args.voice == "auto":
                    # Auto-select voice based on content and script preferences
                    if "Male Reporter:" in script_content and "Female Reporter:" in script_content:
                        voice_type = "both_reporters"  # Special case for dialogue
                        logger.info("🎭 Detected dialogue script - will alternate between male and female voices")
                    elif "Male Reporter:" in script_content:
                        voice_type = "male_reporter_default"
                    elif "Female Reporter:" in script_content:
                        voice_type = "female_reporter_default"
                    else:
                        voice_type = "male_reporter_default"  # Default fallback
                else:
                    voice_type = args.voice
                
                # Generate ONE continuous audio file with proper voice switching
                logger.info("🎙️ Generating continuous audio with voice switching...")
                
                if use_macbook_tts:
                    # Use XTTS-v2 with user preferences
                    output_path = xtts.create_continuous_podcast_audio(
                        full_script=cleaned_script,
                        original_script=script_content,  # Keep original for speaker detection
                        voice_type=voice_type,
                        speed_factor=speed_factor,  # Apply actual user speed preference
                        energy_level=energy_level,  # Apply actual user energy preference
                        output_dir="audio_output"
                    )
                else:
                    # Use XTTS-v2
                    output_path = xtts.create_continuous_podcast_audio(
                        full_script=cleaned_script,
                        original_script=script_content,
                        voice_type=voice_type,
                        speed_factor=speed_factor,
                        energy_level=energy_level,
                        output_dir="audio_output"
                    )
                
                if output_path:
                    print(f"\n🎉 Audio generation completed!")
                    print(f"📁 Generated continuous audio file: {output_path}")
                    print(f"🎙️ Voice pattern: {voice_type}")
                    print(f"⚡ Speed: {speed_factor}x ({user_preferences.get('speed_name', 'fast') if user_preferences else 'fast'})")
                    print(f"🔥 Energy: {energy_level.title()}")
                    print(f"📂 Output directory: audio_output/")
                else:
                    logger.error("Failed to generate continuous audio file")
                    return 1
                
            except Exception as e:
                logger.error(f"Failed to process script file: {e}")
                return 1
        
        # Handle interactive mode
        if args.interactive:
            logger.info("Starting interactive workflow...")
            try:
                from src.interactive_workflow import InteractiveWorkflow
                workflow = InteractiveWorkflow()
                success = workflow.run_workflow()
                if success:
                    logger.info("Interactive workflow completed successfully")
                    return 0
                else:
                    logger.error("Interactive workflow failed")
                    return 1
            except Exception as e:
                logger.error(f"Interactive workflow failed: {e}")
                return 1
        
        # If no specific action requested, show help
        if not any([args.process_dataset, args.generate, args.generate_audio, 
                   args.create_voice_samples, args.list_voices, args.status, 
                   args.list_prompts, args.reprocess, args.interactive]):
            parser.print_help()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 