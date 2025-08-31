#!/usr/bin/env python3
"""
Interactive Workflow Module for Podcast Generation
Handles user input, validation, and workflow orchestration
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from loguru import logger

class InteractiveWorkflow:
    """Interactive workflow for podcast generation with user customization."""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.voice_samples_dir = Path("voice_samples")
        self.audio_output_dir = Path("audio_output")
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.audio_output_dir.mkdir(exist_ok=True)
        
        # User selections storage
        self.user_selections = {}
        self.workflow_type = None  # 'script_exists' or 'generate_new'
        
    def run_workflow(self) -> bool:
        """Main workflow execution."""
        try:
            print("🎙️ === PODCAST GENERATION WORKFLOW === 🎙️")
            print("Welcome to the Interactive Podcast Generator!")
            print("=" * 50)
            
            # Step 1: Check if user has existing script
            if not self._check_script_availability():
                return False
            
            # Step 2: Handle script path or RAG processing
            if not self._handle_script_or_rag():
                return False
            
            # Step 3: Collect user preferences
            if not self._collect_user_preferences():
                return False
            
            # Step 4: Show summary and confirm execution
            if not self._show_summary_and_confirm():
                return False
            
            # Step 5: Execute the workflow
            return self._execute_workflow()
            
        except KeyboardInterrupt:
            print("\n\n❌ Operation cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            print(f"\n❌ Workflow failed: {e}")
            return False
    
    def _check_script_availability(self) -> bool:
        """Ask user if they have an existing script."""
        print("\n📝 STEP 1: Script Availability Check")
        print("-" * 30)
        
        while True:
            try:
                choice = input("Do you have an existing script? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    self.workflow_type = 'script_exists'
                    return True
                elif choice in ['n', 'no']:
                    self.workflow_type = 'generate_new'
                    return True
                else:
                    print("❌ Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                raise
    
    def _handle_script_or_rag(self) -> bool:
        """Handle existing script path or RAG document collection."""
        if self.workflow_type == 'script_exists':
            return self._handle_existing_script()
        else:
            return self._handle_rag_documents()
    
    def _handle_existing_script(self) -> bool:
        """Handle existing script selection."""
        print("\n📁 STEP 2A: Existing Script Selection")
        print("-" * 30)
        
        # Use the new script selection method
        return self._collect_script_selection()
    
    def _handle_rag_documents(self) -> bool:
        """Handle RAG document collection."""
        print("\n📚 STEP 2B: RAG Document Collection")
        print("-" * 30)
        
        # Collect document selection using the new method
        if not self._collect_document_selection():
            return False
        
        # Get prompt path
        if not self._collect_prompt_path():
            return False
        
        return True
    
    def _collect_document_path(self, doc_num: int, document_paths: List[str]) -> bool:
        """Collect a single document path with validation."""
        print(f"Document {doc_num} - Enter the full path to your PDF file")
        print("Example: dataset/document.pdf or /full/path/to/document.pdf")
        
        retry_count = 0
        while retry_count < 3:
            try:
                doc_path = input(f"Document {doc_num} path: ").strip()
                
                if not doc_path:
                    print("❌ Document path cannot be empty")
                    retry_count += 1
                    continue
                
                # Validate document path
                if self._validate_document_path(doc_path):
                    document_paths.append(doc_path)
                    print(f"✅ Document {doc_num} validated: {doc_path}")
                    return True
                else:
                    retry_count += 1
                    print(f"❌ Invalid document path. Retries remaining: {3 - retry_count}")
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _collect_prompt_path(self) -> bool:
        """Collect prompt path for LLM."""
        print("Available prompts:")
        prompts_dir = Path("prompts")
        if prompts_dir.exists():
            prompt_files = list(prompts_dir.glob("*.txt"))
            for i, prompt_file in enumerate(prompt_files, 1):
                print(f"   {i}. {prompt_file.name}")
        
        retry_count = 0
        while retry_count < 3:
            try:
                prompt_input = input("Enter prompt number or name (e.g., 1 or long-desc): ").strip()
                
                if not prompt_input:
                    print("❌ Prompt selection cannot be empty")
                    retry_count += 1
                    continue
                
                # Check if user entered a number
                try:
                    prompt_num = int(prompt_input)
                    if 1 <= prompt_num <= len(prompt_files):
                        # User selected by number
                        selected_prompt = prompt_files[prompt_num - 1].stem  # Remove .txt extension
                        self.user_selections['prompt_path'] = selected_prompt
                        print(f"✅ Prompt selected: {selected_prompt}")
                        return True
                    else:
                        print(f"❌ Please enter a number between 1 and {len(prompt_files)}")
                        retry_count += 1
                        continue
                except ValueError:
                    # User entered text, treat as prompt name
                    prompt_path = prompt_input
                
                # Validate prompt path
                if self._validate_prompt_path(prompt_path):
                    self.user_selections['prompt_path'] = prompt_path
                    print(f"✅ Prompt validated: {prompt_path}")
                    return True
                else:
                    retry_count += 1
                    print(f"❌ Invalid prompt name. Retries remaining: {3 - retry_count}")
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _collect_user_preferences(self) -> bool:
        """Collect user preferences for script generation and audio."""
        print("\n🎭 STEP 3: User Preferences")
        print("-" * 30)
        
        # Speed options
        if not self._collect_speed_preference():
            return False
        
        # Energy options
        if not self._collect_energy_preference():
            return False
        
        # Tone options
        if not self._collect_tone_preference():
            return False
        
        # Duration (only for new script generation)
        if self.workflow_type == 'generate_new':
            if not self._collect_duration_preference():
                return False
        
        # Number of speakers
        if not self._collect_speaker_count():
            return False
        
        return True
    
    def _collect_speed_preference(self) -> bool:
        """Collect speed preference."""
        print("\n1️⃣ SPEED OF CONVERSATION:")
        print("   1. Slow (0.8x): Detailed analysis, complex topics")
        print("   2. Normal (1.0x): Balanced pace, general discussion")
        print("   3. Fast (1.3x): Quick insights, breaking news")
        print("   4. Very Fast (1.5x): High-energy, rapid-fire discussion")
        
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input("Select speed [1-4] (default: 3): ").strip()
                
                if not choice:
                    choice = "3"  # Default to fast
                
                if choice in ['1', '2', '3', '4']:
                    speed_map = {
                        '1': ('slow', 0.8),
                        '2': ('normal', 1.0),
                        '3': ('fast', 1.3),
                        '4': ('very_fast', 1.5)
                    }
                    speed_name, speed_factor = speed_map[choice]
                    self.user_selections['speed'] = {
                        'name': speed_name,
                        'factor': speed_factor
                    }
                    print(f"✅ Speed selected: {speed_name} ({speed_factor}x)")
                    return True
                else:
                    print("❌ Please enter a number between 1 and 4")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _collect_energy_preference(self) -> bool:
        """Collect energy preference."""
        print("\n2️⃣ ENERGY LEVEL:")
        print("   1. Low: Calm, serious, analytical")
        print("   2. Medium: Balanced, professional, engaging")
        print("   3. High: Exciting, dynamic, passionate")
        
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input("Select energy [1-3] (default: 3): ").strip()
                
                if not choice:
                    choice = "3"  # Default to high
                
                if choice in ['1', '2', '3']:
                    energy_map = {
                        '1': 'low',
                        '2': 'medium',
                        '3': 'high'
                    }
                    energy = energy_map[choice]
                    self.user_selections['energy'] = energy
                    print(f"✅ Energy selected: {energy}")
                    return True
                else:
                    print("❌ Please enter a number between 1 and 3")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _collect_tone_preference(self) -> bool:
        """Collect tone preference."""
        print("\n3️⃣ TONE OF CONVERSATION:")
        print("   1. Professional: Formal, business-like, authoritative")
        print("   2. Academic: Educational, analytical, research-focused")
        print("   3. Sports Commentator: Energetic, exciting, play-by-play")
        print("   4. Narrator: Storytelling, engaging, descriptive")
        
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input("Select tone [1-4] (default: 1): ").strip()
                
                if not choice:
                    choice = "1"  # Default to professional
                
                if choice in ['1', '2', '3', '4']:
                    tone_map = {
                        '1': 'professional',
                        '2': 'academic',
                        '3': 'sports_commentator',
                        '4': 'narrator'
                    }
                    tone = tone_map[choice]
                    self.user_selections['tone'] = tone
                    print(f"✅ Tone selected: {tone}")
                    return True
                else:
                    print("❌ Please enter a number between 1 and 4")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _collect_duration_preference(self) -> bool:
        """Collect duration preference."""
        print("\n4️⃣ DURATION OF CONVERSATION:")
        print("   1. Summary: 2-3 minutes, key points only")
        print("   2. Short: 3-5 minutes, brief overview")
        print("   3. Medium: 5-8 minutes, balanced coverage")
        print("   4. Long: 9-11 minutes, comprehensive discussion")
        print("   5. Detailed: More than 12 minutes, in-depth analysis")
        
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input("Select duration [1-5] (default: 3): ").strip()
                
                if not choice:
                    choice = "3"  # Default to medium
                
                if choice in ['1', '2', '3', '4', '5']:
                    duration_map = {
                        '1': 'summary',
                        '2': 'short',
                        '3': 'medium',
                        '4': 'long',
                        '5': 'detailed'
                    }
                    duration = duration_map[choice]
                    self.user_selections['duration'] = duration
                    print(f"✅ Duration selected: {duration}")
                    return True
                else:
                    print("❌ Please enter a number between 1 and 5")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _collect_speaker_count(self) -> bool:
        """Collect number of speakers for the podcast."""
        print("\n5️⃣ NUMBER OF SPEAKERS:")
        print("   How many people will be speaking in this podcast?")
        
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input("Enter number of speakers [1-5] (default: 2): ").strip()
                
                if not choice:
                    choice = "2"  # Default to 2 speakers
                
                try:
                    speaker_count = int(choice)
                    if 1 <= speaker_count <= 5:
                        self.user_selections['speaker_count'] = speaker_count
                        print(f"✅ Number of speakers: {speaker_count}")
                        
                        # Collect speaker voice assignments
                        if speaker_count > 1:
                            if not self._collect_speaker_assignments(speaker_count):
                                return False
                            
                            # Ask who starts first
                            if not self._collect_starting_speaker():
                                return False
                        
                        return True
                    else:
                        print("❌ Please enter a number between 1 and 5")
                        retry_count += 1
                except ValueError:
                    print("❌ Please enter a valid number")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _collect_speaker_assignments(self, speaker_count: int) -> bool:
        """Collect speaker voice assignments."""
        print(f"\n🎭 SPEAKER VOICE ASSIGNMENTS ({speaker_count} speakers):")
        
        # List available voice samples
        voice_samples = self._list_voice_samples()
        if not voice_samples:
            print("❌ No voice samples found in voice_samples/ directory")
            return False
        
        print("Available voice samples:")
        for i, sample in enumerate(voice_samples, 1):
            print(f"   {i}. {sample}")
        
        # Assign voices to speakers
        speakers = [f"Person {i}" for i in range(1, speaker_count + 1)]
        speaker_assignments = {}
        
        for speaker in speakers:
            retry_count = 0
            while retry_count < 3:
                try:
                    choice = input(f"Select voice for {speaker} [1-{len(voice_samples)}]: ").strip()
                    
                    try:
                        choice = int(choice)
                        if 1 <= choice <= len(voice_samples):
                            selected_sample = voice_samples[choice - 1]
                            speaker_assignments[speaker] = selected_sample
                            print(f"✅ {speaker} assigned to: {selected_sample}")
                            break
                        else:
                            print(f"❌ Please enter a number between 1 and {len(voice_samples)}")
                            retry_count += 1
                    except ValueError:
                        print("❌ Please enter a valid number")
                        retry_count += 1
                        
                except KeyboardInterrupt:
                    raise
            
            if retry_count >= 3:
                print("❌ Maximum retries exceeded. Please restart the program.")
                return False
        
        self.user_selections['speaker_assignments'] = speaker_assignments
        return True
    
    def _collect_starting_speaker(self) -> bool:
        """Collect which speaker starts first."""
        print(f"\n🎬 STARTING SPEAKER:")
        print("   Which speaker would you like to start the podcast?")
        
        # Get the speaker names from assignments
        speakers = list(self.user_selections['speaker_assignments'].keys())
        
        print("Available speakers:")
        for i, speaker in enumerate(speakers, 1):
            voice_sample = self.user_selections['speaker_assignments'][speaker]
            print(f"   {i}. {speaker} (using voice: {voice_sample})")
        
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input(f"Select starting speaker [1-{len(speakers)}]: ").strip()
                
                try:
                    choice = int(choice)
                    if 1 <= choice <= len(speakers):
                        starting_speaker = speakers[choice - 1]
                        self.user_selections['starting_speaker'] = starting_speaker
                        print(f"✅ {starting_speaker} will start the podcast")
                        return True
                    else:
                        print(f"❌ Please enter a number between 1 and {len(speakers)}")
                        retry_count += 1
                except ValueError:
                    print("❌ Please enter a valid number")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _list_voice_samples(self) -> List[str]:
        """List available voice samples."""
        if not self.voice_samples_dir.exists():
            return []
        
        wav_files = list(self.voice_samples_dir.glob("*.wav"))
        return [f.name for f in wav_files]
    
    def _show_summary_and_confirm(self) -> bool:
        """Show summary of all selections and confirm execution."""
        print("\n📋 STEP 4: Summary and Confirmation")
        print("=" * 50)
        
        print("🎯 WORKFLOW TYPE:")
        print(f"   {'Script exists' if self.workflow_type == 'script_exists' else 'Generate new script'}")
        
        if self.workflow_type == 'script_exists':
            print(f"   Script path: {self.user_selections['script_path']}")
        else:
            print(f"   Documents: {len(self.user_selections['document_paths'])}")
            print(f"   Prompt: {self.user_selections['prompt_path']}")
        
        print("\n⚙️ USER PREFERENCES:")
        print(f"   Speed: {self.user_selections['speed']['name']} ({self.user_selections['speed']['factor']}x)")
        print(f"   Energy: {self.user_selections['energy']}")
        print(f"   Tone: {self.user_selections['tone']}")
        if 'duration' in self.user_selections:
            print(f"   Duration: {self.user_selections['duration']}")
        print(f"   Speakers: {self.user_selections['speaker_count']}")
        
        if 'speaker_assignments' in self.user_selections:
            print("\n🎭 SPEAKER ASSIGNMENTS:")
            for speaker, voice in self.user_selections['speaker_assignments'].items():
                print(f"   {speaker}: {voice}")
            
            if 'starting_speaker' in self.user_selections:
                print(f"   🎬 Starting Speaker: {self.user_selections['starting_speaker']}")
        
        print("\n" + "=" * 50)
        
        # Confirm execution
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input("Proceed with execution? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    print("✅ Proceeding with execution...")
                    return True
                elif choice in ['n', 'no']:
                    print("❌ Execution cancelled by user")
                    return False
                else:
                    print("❌ Please enter 'y' for yes or 'n' for no")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _execute_workflow(self) -> bool:
        """Execute the workflow based on user selections."""
        print("\n🚀 STEP 5: Executing Workflow")
        print("=" * 50)
        
        try:
            # Generate timestamp for unique identification
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.workflow_type == 'script_exists':
                # Handle existing script workflow
                success = self._execute_existing_script_workflow(timestamp)
            else:
                # Handle new script generation workflow
                success = self._execute_new_script_workflow(timestamp)
            
            if success:
                # Save workflow metadata
                self._save_workflow_metadata(timestamp)
                print("\n🎉 Workflow completed successfully!")
                return True
            else:
                print("\n❌ Workflow execution failed")
                return False
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            print(f"\n❌ Workflow execution failed: {e}")
            return False
    
    def _execute_existing_script_workflow(self, timestamp: str) -> bool:
        """Execute workflow for existing script."""
        print("📖 Processing existing script...")
        
        try:
            from xtts_integration import XTTSIntegration
            from voice_sample_manager import VoiceSampleManager
            from llm_interface import LLMInterface
            
            # Initialize components
            voice_manager = VoiceSampleManager()
            xtts = XTTSIntegration(voice_manager)
            
            # Initialize XTTS-v2 model
            if not xtts.initialize_macbook_tts_fallback():
                logger.error("Failed to initialize XTTS-v2")
                return False
            
            # Read the script content
            script_path = Path(self.user_selections['script_path'])
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            print(f"✅ Script loaded: {len(script_content)} characters")
            
            # Clean the script for audio generation
            llm_interface = LLMInterface()
            cleaned_script, cleaned_filename = llm_interface.clean_script_for_audio(script_content)
            print(f"✅ Script cleaned: {len(cleaned_script)} characters")
            
            if cleaned_filename:
                print(f"📁 Cleaned script saved to: output/cleanup-scripts/{cleaned_filename}")
                
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
                            print(f"✅ Loaded updated script from file: {len(cleaned_script)} characters")
                        except Exception as e:
                            print(f"⚠️ Failed to load updated file, using cleaned script from memory: {e}")
                    else:
                        print("✅ Using cleaned script from memory")
                else:
                    print("✅ Proceeding with TTS generation using cleaned script")
            
            # Determine voice type based on speaker count
            if self.user_selections['speaker_count'] == 1:
                voice_type = "male_reporter_default"  # Default to male for single speaker
            else:
                voice_type = "both_reporters"  # Use both voices for multiple speakers
            
            # Create dynamic speaker mapping for TTS (same logic as new script workflow)
            tts_speaker_mapping = None
            if self.user_selections['speaker_count'] > 1 and 'speaker_assignments' in self.user_selections:
                starting_speaker = self.user_selections.get('starting_speaker', 'Person 1')
                speaker_assignments = self.user_selections['speaker_assignments']
                
                # Create the mapping: Person 1/2 -> actual voice files
                if starting_speaker == "Person 1":
                    # Person 1 starts first
                    tts_speaker_mapping = {
                        "Person 1": speaker_assignments["Person 1"],
                        "Person 2": speaker_assignments["Person 2"]
                    }
                else:
                    # Person 2 starts first - swap the mapping
                    tts_speaker_mapping = {
                        "Person 1": speaker_assignments["Person 2"],  # Person 2's voice for Person 1
                        "Person 2": speaker_assignments["Person 1"]   # Person 1's voice for Person 2
                    }
                
                logger.info(f"🎭 Created speaker mapping for existing script: {tts_speaker_mapping}")
            else:
                logger.warning("🚨 No speaker assignments found for existing script workflow")
            
            # Generate audio with user preferences
            print("🎙️ Generating audio with voice switching...")
            output_path = xtts.create_continuous_podcast_audio(
                full_script=cleaned_script,
                original_script=script_content,
                voice_type=voice_type,
                speed_factor=self.user_selections['speed']['factor'],
                energy_level=self.user_selections['energy'],
                output_dir="audio_output",
                speaker_mapping=tts_speaker_mapping
            )
            
            if output_path:
                print(f"🎉 Audio generation completed!")
                print(f"📁 Generated file: {output_path}")
                print(f"⚡ Speed: {self.user_selections['speed']['name']} ({self.user_selections['speed']['factor']}x)")
                print(f"🔥 Energy: {self.user_selections['energy']}")
                return True
            else:
                print("❌ Audio generation failed")
                return False
                
        except Exception as e:
            logger.error(f"Existing script workflow failed: {e}")
            print(f"❌ Workflow failed: {e}")
            return False
    
    def _execute_new_script_workflow(self, timestamp: str) -> bool:
        """Execute workflow for new script generation."""
        print("🆕 Generating new script...")
        
        try:
            from complete_lightweight_system import CompleteLightweightSystem
            
            # Initialize the complete system
            print("🔧 Initializing system components...")
            system = CompleteLightweightSystem()
            
            # Generate script with user preferences
            print("📝 Generating script with user preferences...")
            
            # Map user preferences to system parameters
            speed_name = self.user_selections['speed']['name']
            energy_name = self.user_selections['energy']
            duration_name = self.user_selections['duration']
            
            # Determine voice type based on speaker count
            if self.user_selections['speaker_count'] == 1:
                voice_type = "male"
            else:
                voice_type = "both"
            
            # Create preferences dictionary for the system
            preferences = {
                'speed_name': speed_name,
                'speed_factor': self.user_selections['speed']['factor'],
                'energy_name': energy_name,
                'tone': self.user_selections['tone'],
                'conversation_length': duration_name,
                'voice_type': voice_type,
                'speaker_count': self.user_selections['speaker_count'],
                'starting_speaker': self.user_selections.get('starting_speaker', 'Person 1'),
                'speaker_assignments': self.user_selections.get('speaker_assignments', {})
            }
            
            # Create dynamic speaker mapping for TTS
            if voice_type == "both" and 'speaker_assignments' in self.user_selections:
                starting_speaker = self.user_selections['starting_speaker']
                speaker_assignments = self.user_selections['speaker_assignments']
                
                # Create the mapping: Person 1/2 -> actual voice files
                tts_speaker_mapping = {}
                if starting_speaker == "Person 1":
                    # Person 1 starts first
                    tts_speaker_mapping = {
                        "Person 1": speaker_assignments["Person 1"],
                        "Person 2": speaker_assignments["Person 2"]
                    }
                else:
                    # Person 2 starts first - swap the mapping
                    tts_speaker_mapping = {
                        "Person 1": speaker_assignments["Person 2"],  # Person 2's voice for Person 1
                        "Person 2": speaker_assignments["Person 1"]   # Person 1's voice for Person 2
                    }
                
                preferences['tts_speaker_mapping'] = tts_speaker_mapping
                # Also store in user_selections for the existing script workflow
                self.user_selections['tts_speaker_mapping'] = tts_speaker_mapping
            
            # Generate script using the existing system
            script_content = system.generate_script_with_preferences(
                prompt_name=self.user_selections['prompt_path'],
                preferences=preferences
            )
            
            if not script_content:
                print("❌ Script generation failed")
                return False
            
            # Save the generated script
            script_filename = f"script_{self.user_selections['prompt_path']}_{timestamp}.txt"
            script_path = self.output_dir / "scripts" / script_filename
            script_path.parent.mkdir(exist_ok=True)
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            print(f"✅ Script saved: {script_path}")
            
            # Now generate audio from the script
            print("🎙️ Generating audio from generated script...")
            
            # Update user selections to include the generated script path
            self.user_selections['script_path'] = str(script_path)
            
            # Execute the existing script workflow
            return self._execute_existing_script_workflow(timestamp)
            
        except Exception as e:
            logger.error(f"New script workflow failed: {e}")
            print(f"❌ Workflow failed: {e}")
            return False
    
    def _save_workflow_metadata(self, timestamp: str) -> bool:
        """Save workflow metadata for debugging and audit."""
        try:
            metadata = {
                'timestamp': timestamp,
                'workflow_type': self.workflow_type,
                'user_selections': self.user_selections,
                'execution_status': 'completed'
            }
            
            # Create filename based on workflow type
            if self.workflow_type == 'script_exists':
                filename = f"workflow_existing_script_{timestamp}.json"
            else:
                filename = f"workflow_new_script_{timestamp}.json"
            
            metadata_path = self.output_dir / filename
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Workflow metadata saved: {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save workflow metadata: {e}")
            return False
    
    # Validation methods
    def _get_available_scripts(self) -> List[Path]:
        """Get list of available scripts from output/scripts folder."""
        scripts_dir = Path("output/scripts")
        if not scripts_dir.exists():
            return []
        
        # Get all text files in scripts folder
        script_files = [f for f in scripts_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() == '.txt']
        
        return sorted(script_files, key=lambda x: x.stat().st_mtime, reverse=True)  # Sort by modification time, newest first
    
    def _collect_script_selection(self) -> bool:
        """Collect script selection from available scripts."""
        available_scripts = self._get_available_scripts()
        
        if not available_scripts:
            print("❌ No scripts found in output/scripts folder")
            print("   Please generate a script first or check the folder path")
            return False
        
        print(f"\n📁 Available Scripts ({len(available_scripts)} files):")
        for i, script_path in enumerate(available_scripts, 1):
            # Get file size and modification time
            stat = script_path.stat()
            size_kb = stat.st_size / 1024
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"   {i}. {script_path.name} ({size_kb:.1f} KB, {mod_time})")
        
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input(f"\nSelect script by number [1-{len(available_scripts)}]: ").strip()
                
                if not choice:
                    print("❌ Please select a script")
                    retry_count += 1
                    continue
                
                try:
                    script_num = int(choice)
                    if 1 <= script_num <= len(available_scripts):
                        selected_script = available_scripts[script_num - 1]
                        self.user_selections['script_path'] = str(selected_script)
                        print(f"✅ Script selected: {selected_script.name}")
                        return True
                    else:
                        print(f"❌ Please enter a number between 1 and {len(available_scripts)}")
                        retry_count += 1
                except ValueError:
                    print("❌ Please enter a valid number")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _validate_script_path(self, script_path: str) -> bool:
        """Validate script path."""
        path = Path(script_path)
        return path.exists() and path.is_file() and path.suffix == '.txt'
    
    def _get_available_documents(self) -> List[Path]:
        """Get list of available PDF documents from dataset folder."""
        dataset_dir = Path("dataset")
        if not dataset_dir.exists():
            return []
        
        # Get all PDF files in dataset folder (ignore subdirectories)
        pdf_files = [f for f in dataset_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() == '.pdf']
        
        return sorted(pdf_files)
    
    def _collect_document_selection(self) -> bool:
        """Collect document selection from available PDFs."""
        available_docs = self._get_available_documents()
        
        if not available_docs:
            print("❌ No PDF documents found in dataset folder")
            print("   Please add PDF files to the dataset folder and try again")
            return False
        
        print(f"\n📚 Available Documents ({len(available_docs)} PDFs):")
        for i, doc_path in enumerate(available_docs, 1):
            print(f"   {i}. {doc_path.name}")
        
        retry_count = 0
        while retry_count < 3:
            try:
                choice = input(f"\nSelect documents by number (e.g., 1,2 or 1-3) [1-{len(available_docs)}]: ").strip()
                
                if not choice:
                    print("❌ Please select at least one document")
                    retry_count += 1
                    continue
                
                # Parse the selection (support comma-separated and range)
                selected_docs = []
                for part in choice.split(','):
                    part = part.strip()
                    if '-' in part:
                        # Handle range (e.g., "1-3")
                        try:
                            start, end = map(int, part.split('-'))
                            if 1 <= start <= end <= len(available_docs):
                                selected_docs.extend(range(start, end + 1))
                            else:
                                print(f"❌ Invalid range: {part}")
                                retry_count += 1
                                continue
                        except ValueError:
                            print(f"❌ Invalid range format: {part}")
                            retry_count += 1
                            continue
                    else:
                        # Handle single number
                        try:
                            num = int(part)
                            if 1 <= num <= len(available_docs):
                                selected_docs.append(num)
                            else:
                                print(f"❌ Invalid document number: {num}")
                                retry_count += 1
                                continue
                        except ValueError:
                            print(f"❌ Invalid number: {part}")
                            retry_count += 1
                            continue
                
                if selected_docs:
                    # Convert to unique, sorted list
                    selected_docs = sorted(list(set(selected_docs)))
                    selected_paths = [available_docs[i-1] for i in selected_docs]
                    
                    self.user_selections['document_paths'] = [str(p) for p in selected_paths]
                    print(f"✅ Selected {len(selected_paths)} documents:")
                    for path in selected_paths:
                        print(f"   📄 {path.name}")
                    return True
                else:
                    print("❌ No valid documents selected")
                    retry_count += 1
                    
            except KeyboardInterrupt:
                raise
        
        print("❌ Maximum retries exceeded. Please restart the program.")
        return False
    
    def _validate_document_path(self, doc_path: str) -> bool:
        """Validate document path."""
        path = Path(doc_path)
        if not path.exists() or not path.is_file():
            return False
        
        # Check if it's a PDF
        if path.suffix.lower() != '.pdf':
            print(f"❌ File format not supported: {path.suffix}")
            print("   Only PDF files are supported for RAG processing")
            return False
        
        return True
    
    def _validate_prompt_path(self, prompt_path: str) -> bool:
        """Validate prompt path."""
        if prompt_path.startswith("prompts/"):
            # User provided full path, use as is
            prompt_file = Path(prompt_path)
        else:
            # User provided just the name, add prompts folder and .txt extension
            if not prompt_path.endswith('.txt'):
                prompt_path += '.txt'
            prompt_file = Path("prompts") / prompt_path
        
        if not prompt_file.exists():
            return False
        
        # Check if it's a text file
        if prompt_file.suffix.lower() not in ['.txt', '.md']:
            return False
        
        return True
    
 