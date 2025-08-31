#!/usr/bin/env python3
"""
XTTS-v2 Integration Module for Podcast Script Generation
Provides high-quality text-to-speech conversion using Coqui TTS
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch

logger = logging.getLogger(__name__)

class XTTSIntegration:
    """Integration class for XTTS-v2 voice cloning."""
    
    def __init__(self, voice_manager=None):
        """Initialize XTTS integration.
        
        Args:
            voice_manager: VoiceSampleManager instance for voice sample management
        """
        self.voice_manager = voice_manager
        self.xtts_model = None
        self.xtts_available = False
        self.macbook_tts_available = False
        self.device = "cpu"  # Default to CPU for compatibility
        self.tts = None
        self.output_folder = Path("output")
        self.output_folder.mkdir(exist_ok=True)
    
    def _initialize_tts(self):
        """Initialize the TTS model using the working TTS.api approach."""
        try:
            from TTS.api import TTS
            logger.info("Initializing TTS with XTTS-v2 model")
            logger.info(f"Using device: {self.device}")
            
            # Initialize TTS with XTTS-v2 - using the working model name
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
            self.xtts_model = self.tts  # Set the model reference
            self.xtts_available = True  # Mark as available
            logger.info("✅ TTS model initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts = None
            self.xtts_model = None
            self.xtts_available = False
    
    def get_available_voices(self) -> List[str]:
        """Get available voice options."""
        if not self.tts:
            return []
        
        try:
            # For XTTS-v2, we can use different speaker references
            # Let's try to get the actual available speakers from the model
            if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'tts_model'):
                model = self.tts.synthesizer.tts_model
                if hasattr(model, 'speaker_manager'):
                    speaker_manager = model.speaker_manager
                    if hasattr(speaker_manager, 'speaker_names'):
                        voices = list(speaker_manager.speaker_names)
                        if voices:
                            return voices
            
            # Fallback to default voices - use indices since we know there are 58 speakers
            voices = [str(i) for i in range(58)]
            return voices
        except Exception as e:
            logger.warning(f"Could not get voices: {e}")
            return []
    
    def synthesize_speech(self, text: str, output_file: str, speaker: str = None, language: str = "en") -> Optional[str]:
        """Convert text to speech using XTTS-v2."""
        if not self.tts:
            logger.error("TTS not initialized")
            return None
        
        try:
            # Ensure output file has proper extension
            if not output_file.endswith('.wav'):
                output_file = output_file + '.wav'
            
            output_path = self.output_folder / output_file
            
            logger.info(f"Synthesizing speech: {len(text)} characters")
            logger.info(f"Speaker: {speaker}, Language: {language}")
            logger.info(f"Output: {output_path}")
            
            # Get available speakers if none provided
            if not speaker:
                available_speakers = self.get_available_voices()
                if available_speakers:
                    speaker = available_speakers[0]  # Use first available speaker
                    logger.info(f"Using default speaker: {speaker}")
                else:
                    logger.error("No speakers available")
                    return None
            
            # For XTTS-v2, we need to use voice cloning with speaker_wav
            # Get a voice sample for cloning
            voice_sample = self._get_voice_sample_for_speaker(speaker)
            
            if not voice_sample:
                logger.error("No voice sample available for voice cloning")
                return None
            
            try:
                # Use voice cloning with the reference audio
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker_wav=voice_sample,
                    language="en"
                )
                logger.info(f"✅ Speech synthesis completed with voice cloning: {output_path}")
                return str(output_path)
            except Exception as e:
                logger.error(f"Speech synthesis failed: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return None
    
    def _get_voice_sample_for_speaker(self, speaker: str) -> Optional[str]:
        """Get the appropriate voice sample for a given speaker."""
        try:
            voice_samples_dir = Path("voice_samples")
            if not voice_samples_dir.exists():
                logger.error("Voice samples directory not found")
                return None
            
            # Map speaker names to voice samples using new naming convention
            if "Male" in speaker or "male" in speaker:
                # Look for male voice sample using new naming convention
                male_samples = list(voice_samples_dir.glob("male_*.wav"))
                if male_samples:
                    return str(male_samples[0])
                # Fallback to any .wav file
                wav_files = list(voice_samples_dir.glob("*.wav"))
                if wav_files:
                    return str(wav_files[0])
            elif "Female" in speaker or "female" in speaker:
                # Look for female voice sample using new naming convention
                female_samples = list(voice_samples_dir.glob("female_*.wav"))
                if female_samples:
                    return str(female_samples[0])
                # Fallback to any .wav file
                wav_files = list(voice_samples_dir.glob("*.wav"))
                if wav_files:
                    return str(wav_files[0])
            else:
                # Default: use any available voice sample
                wav_files = list(voice_samples_dir.glob("*.wav"))
                if wav_files:
                    return str(wav_files[0])
            
            logger.error("No voice samples found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting voice sample: {e}")
            return None
    
    def _select_voice_sample_for_type(self, voice_type: str) -> Optional[str]:
        """Select the appropriate voice sample for a given voice type."""
        try:
            voice_samples_dir = Path("voice_samples")
            if not voice_samples_dir.exists():
                logger.error("Voice samples directory not found")
                return None
            
            print(f"🔍 DEBUG: Selecting voice sample for type: '{voice_type}'")
            logger.info(f"🔍 Selecting voice sample for type: '{voice_type}'")
            
            # Map voice types to voice samples - use more specific matching
            if voice_type == "male_reporter_default":
                # Look for male voice sample using new naming convention
                male_samples = list(voice_samples_dir.glob("male_*.wav"))
                if male_samples:
                    selected_sample = str(male_samples[0])
                    print(f"✅ DEBUG: Selected MALE voice sample: {selected_sample}")
                    logger.info(f"✅ Selected MALE voice sample: {selected_sample}")
                    return selected_sample
                else:
                    print(f"⚠️ DEBUG: No male voice samples found for type: {voice_type}")
                    logger.warning(f"⚠️ No male voice samples found for type: {voice_type}")
                    
            elif voice_type == "female_reporter_default":
                # Look for female voice sample using new naming convention
                female_samples = list(voice_samples_dir.glob("female_*.wav"))
                if female_samples:
                    selected_sample = str(female_samples[0])
                    print(f"✅ DEBUG: Selected FEMALE voice sample: {selected_sample}")
                    logger.info(f"✅ Selected FEMALE voice sample: {selected_sample}")
                    return selected_sample
                else:
                    print(f"⚠️ DEBUG: No female voice samples found for type: {voice_type}")
                    logger.warning(f"⚠️ No female voice samples found for type: {voice_type}")
            
            # Fallback for other voice types
            elif "male" in voice_type.lower() and "female" not in voice_type.lower():
                # Look for male voice sample (only if it doesn't contain "female")
                male_samples = list(voice_samples_dir.glob("male_*.wav"))
                if male_samples:
                    selected_sample = str(male_samples[0])
                    print(f"✅ DEBUG: Selected MALE voice sample (fallback): {selected_sample}")
                    logger.info(f"✅ Selected MALE voice sample (fallback): {selected_sample}")
                    return selected_sample
                    
            elif "female" in voice_type.lower() and "male" not in voice_type.lower():
                # Look for female voice sample (only if it doesn't contain "male")
                female_samples = list(voice_samples_dir.glob("female_*.wav"))
                if female_samples:
                    selected_sample = str(female_samples[0])
                    print(f"✅ DEBUG: Selected FEMALE voice sample (fallback): {selected_sample}")
                    logger.info(f"✅ Selected FEMALE voice sample (fallback): {selected_sample}")
                    return selected_sample
            
            # Only fallback if we couldn't find a specific voice type
            print(f"⚠️ DEBUG: No specific voice match found for '{voice_type}', using fallback")
            logger.warning(f"⚠️ No specific voice match found for '{voice_type}', using fallback")
            wav_files = list(voice_samples_dir.glob("*.wav"))
            if wav_files:
                fallback_sample = str(wav_files[0])
                print(f"⚠️ DEBUG: Using fallback voice sample: {fallback_sample}")
                logger.warning(f"⚠️ Using fallback voice sample: {fallback_sample}")
                return fallback_sample
            
            print("❌ DEBUG: No voice samples found at all")
            logger.error("❌ No voice samples found at all")
            return None
            
        except Exception as e:
            logger.error(f"Error selecting voice sample: {e}")
            return None
    
    def synthesize_script(self, script_path: str, output_prefix: str = "podcast", speaker: str = None) -> List[str]:
        """Convert a complete script file to speech."""
        if not self.tts:
            logger.error("TTS not initialized")
            return []
        
        try:
            script_path = Path(script_path)
            if not script_path.exists():
                logger.error(f"Script file not found: {script_path}")
                return []
            
            # Read script content
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # Split into speaker segments
            segments = self._parse_script_segments(script_content)
            
            generated_files = []
            for i, (speaker_name, text) in enumerate(segments):
                if text.strip():
                    output_file = f"{output_prefix}_segment_{i+1:03d}_{speaker_name}.wav"
                    result = self.synthesize_speech(text, output_file, speaker, "en")
                    if result:
                        generated_files.append(result)
            
            logger.info(f"Generated {len(generated_files)} audio segments")
            return generated_files
            
        except Exception as e:
            logger.error(f"Script synthesis failed: {e}")
            return []
    
    def _parse_script_segments(self, script_content: str, speaker_mapping: Optional[Dict[str, str]] = None) -> List[tuple]:
        """Parse script content into speaker segments."""
        segments = []
        lines = script_content.split('\n')
        current_speaker = "Unknown"
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with speaker label - prioritize Person 1/2 labels using regex
            import re
            
            # Check for Person 1/2 with various markdown formats
            person1_match = re.search(r'^\*?\*?Person\s*1\*?\*?\s*:', line)
            person2_match = re.search(r'^\*?\*?Person\s*2\*?\*?\s*:', line)
            
            if person1_match or person2_match:
                # Save previous segment
                if current_text.strip():
                    segments.append((current_speaker, current_text.strip()))
                
                # Start new segment - map Person 1/2 to voice types
                if person1_match:
                    # Use dynamic mapping if available, otherwise fallback to default
                    if speaker_mapping and "Person 1" in speaker_mapping:
                        current_speaker = speaker_mapping["Person 1"]
                    else:
                        current_speaker = "male_reporter_default"
                else:  # Person 2
                    # Use dynamic mapping if available, otherwise fallback to default
                    if speaker_mapping and "Person 2" in speaker_mapping:
                        current_speaker = speaker_mapping["Person 2"]
                    else:
                        current_speaker = "female_reporter_default"
                
                # Extract text after the label (handle any markdown)
                if person1_match:
                    current_text = re.sub(r'^\*?\*?Person\s*1\*?\*?\s*:', '', line).strip()
                else:
                    current_text = re.sub(r'^\*?\*?Person\s*2\*?\*?\s*:', '', line).strip()
            elif line.startswith("Male Reporter:") or line.startswith("Female Reporter:"):
                # Fallback for old format
                # Save previous segment
                if current_text.strip():
                    segments.append((current_speaker, current_text.strip()))
                
                # Start new segment
                current_speaker = line.split(':')[0].strip()
                current_text = line.split(':', 1)[1].strip() if ':' in line else ""
            else:
                # Continue current segment
                current_text += " " + line
        
        # Add final segment
        if current_text.strip():
            segments.append((current_speaker, current_text.strip()))
        
        return segments
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded TTS model."""
        if not self.tts:
            return {"error": "TTS not initialized"}
        
        try:
            return {
                "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "device": self.device,
                "available_voices": len(self.get_available_voices()),
                "status": "ready",
                "note": "XTTS-v2 requires voice cloning setup for multi-speaker usage"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_synthesis(self, test_text: str = "Hello, this is a test of the XTTS-v2 text-to-speech system.") -> bool:
        """Test the TTS system with a simple text."""
        try:
            # Get available speakers
            available_speakers = self.get_available_voices()
            if not available_speakers:
                logger.error("No speakers available for testing")
                return False
            
            # Use first available speaker
            speaker = available_speakers[0]
            logger.info(f"Testing with speaker: {speaker}")
            
            result = self.synthesize_speech(test_text, "test_synthesis", speaker, "en")
            if result:
                logger.info("✅ TTS test successful!")
                return True
            else:
                logger.error("❌ TTS test failed!")
                return False
        except Exception as e:
            logger.error(f"TTS test error: {e}")
            return False

    def create_continuous_podcast_audio(self, full_script: str, original_script: str, 
                                      voice_type: str, speed_factor: float = 1.0, 
                                      energy_level: str = "medium", output_dir: str = "audio_output",
                                      speaker_mapping: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Create one continuous audio file with voice switching and speed/energy control."""
        try:
            import os
            import tempfile
            from pathlib import Path
            import soundfile as sf
            import numpy as np
            import time
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Split the original script to detect speaker changes
            lines = original_script.split('\n')
            speaker_segments = []
            current_speaker = None
            current_text = []
            
            # Debug: Log the speaker mapping being used
            if speaker_mapping:
                logger.info(f"🎭 Using dynamic speaker mapping: {speaker_mapping}")
                logger.info(f"🎭 Speaker mapping keys: {list(speaker_mapping.keys())}")
                logger.info(f"🎭 Speaker mapping values: {list(speaker_mapping.values())}")
            else:
                logger.info("🎭 Using default speaker mapping (Person 1 -> male, Person 2 -> female)")
                logger.warning("🚨 NO SPEAKER MAPPING PROVIDED - using hardcoded defaults!")
            
            for line in lines:
                line = line.strip()
                # More robust speaker detection with dynamic mapping using regex
                import re
                
                # Check for Person 1 with various markdown formats
                person1_match = re.search(r'\*?\*?Person\s*1\*?\*?\s*:', line)
                if person1_match:
                    if current_speaker and current_text:
                        speaker_segments.append((current_speaker, " ".join(current_text)))
                    
                    # Use dynamic mapping if available, otherwise fallback to default
                    if speaker_mapping and "Person 1" in speaker_mapping:
                        # Map Person 1 directly to the voice file path
                        current_speaker = speaker_mapping["Person 1"]
                        logger.info(f"🎭 Person 1 mapped to: {current_speaker}")
                    else:
                        current_speaker = "male_reporter_default"  # Fallback to default
                        logger.warning(f"🚨 Person 1 using fallback: {current_speaker}")
                    
                    # Extract text after the Person 1 label (handle any markdown)
                    text_part = re.sub(r'\*?\*?Person\s*1\*?\*?\s*:', '', line).strip()
                    if text_part:
                        current_text = [text_part]
                    else:
                        current_text = []
                        
                # Check for Person 2 with various markdown formats
                elif re.search(r'\*?\*?Person\s*2\*?\*?\s*:', line):
                    if current_speaker and current_text:
                        speaker_segments.append((current_speaker, " ".join(current_text)))
                    
                    # Use dynamic mapping if available, otherwise fallback to default
                    if speaker_mapping and "Person 2" in speaker_mapping:
                        # Map Person 2 directly to the voice file path
                        current_speaker = speaker_mapping["Person 2"]
                        logger.info(f"🎭 Person 2 mapped to: {current_speaker}")
                    else:
                        current_speaker = "female_reporter_default"  # Fallback to default
                        logger.warning(f"🚨 Person 2 using fallback: {current_speaker}")
                    
                    # Extract text after the Person 2 label (handle any markdown)
                    text_part = re.sub(r'\*?\*?Person\s*2\*?\*?\s*:', '', line).strip()
                    if text_part:
                        current_text = [text_part]
                    else:
                        current_text = []
                elif "Male Reporter:" in line:  # Fallback for old format
                    if current_speaker and current_text:
                        speaker_segments.append((current_speaker, " ".join(current_text)))
                    current_speaker = "male_reporter_default"
                    # Extract text after "Male Reporter:"
                    text_part = line.split("Male Reporter:", 1)[1].strip() if "Male Reporter:" in line else ""
                    if text_part:
                        current_text = [text_part]
                    else:
                        current_text = []
                elif "Female Reporter:" in line:  # Fallback for old format
                    if current_speaker and current_text:
                        speaker_segments.append((current_speaker, " ".join(current_text)))
                    current_speaker = "female_reporter_default"
                    # Extract text after "Female Reporter:"
                    text_part = line.split("Female Reporter:", 1)[1].strip() if "Female Reporter:" in line else ""
                    if text_part:
                        current_text = [text_part]
                    else:
                        current_text = []
                elif line and current_speaker and not line.startswith("###") and not line.startswith("=="):
                    # Add text to current segment, excluding formatting lines
                    current_text.append(line)
            
            # Add the last segment
            if current_speaker and current_text:
                speaker_segments.append((current_speaker, " ".join(current_text)))
            
            logger.info(f"🎭 Detected {len(speaker_segments)} speaker segments")
            
            # Debug: Log the detected segments
            for i, (speaker, text) in enumerate(speaker_segments):
                logger.info(f"   Segment {i+1}: {speaker} - {len(text.split())} words")
                if len(text) > 100:
                    logger.info(f"      Preview: {text[:100]}...")
                else:
                    logger.info(f"      Text: {text}")
            
            # Generate audio for each segment with appropriate voice
            audio_segments = []
            
            for i, (speaker, text) in enumerate(speaker_segments):
                if not text.strip():
                    continue
                    
                logger.info(f"🎙️ Generating segment {i+1}/{len(speaker_segments)} with {speaker} voice")
                
                # Generate audio for this segment
                segment_audio = self.create_podcast_segment_macbook(
                    script_segment=text,
                    voice_type=speaker,
                    output_dir=output_dir,
                    speed_factor=speed_factor,
                    energy_level=energy_level
                )
                
                if segment_audio and os.path.exists(segment_audio):
                    # Load the audio segment
                    audio_data, sample_rate = sf.read(segment_audio)
                    audio_segments.append(audio_data)
                    
                    # Clean up temporary segment file
                    try:
                        os.remove(segment_audio)
                    except:
                        pass
                else:
                    logger.warning(f"Failed to generate audio for segment {i+1}")
            
            if not audio_segments:
                logger.error("No audio segments were generated successfully")
                return None
            
            # Concatenate all audio segments
            logger.info("🔗 Concatenating audio segments...")
            
            # Ensure all audio segments have the same data type
            audio_segments = [np.asarray(segment, dtype=np.float32) for segment in audio_segments]
            continuous_audio = np.concatenate(audio_segments)
            
            # Apply final speed adjustment if needed
            if speed_factor != 1.0:
                logger.info(f"⚡ Applying final speed adjustment: {speed_factor}x")
                # Use the same reliable method as individual segments
                if speed_factor > 1.0:
                    # Speed up by skipping samples
                    step = 1.0 / speed_factor
                    indices = np.arange(0, len(continuous_audio), step, dtype=int)
                    continuous_audio = continuous_audio[indices]
                else:
                    # Slow down by repeating samples
                    repeat_factor = int(1.0 / speed_factor)
                    continuous_audio = np.repeat(continuous_audio, repeat_factor)
                logger.info(f"✅ Final speed adjustment applied: {speed_factor}x")
            
            # Generate output filename
            timestamp = int(time.time())
            output_filename = f"continuous_podcast_{voice_type}_{timestamp}.wav"
            output_filepath = output_path / output_filename
            
            # Save the continuous audio file
            sf.write(str(output_filepath), continuous_audio, sample_rate)
            
            logger.info(f"✅ Continuous audio file generated: {output_filepath}")
            return str(output_filepath)
            
        except Exception as e:
            logger.error(f"Failed to create continuous podcast audio: {e}")
            return None

    def create_podcast_segment_macbook(self, script_segment: str, voice_type: str, output_dir: str = "audio_output", speed_factor: float = 1.0, energy_level: str = "medium") -> Optional[str]:
        """Create podcast segment using MacBook-compatible TTS with voice cloning priority."""
        try:
            # Get the appropriate voice sample
            voice_sample_path = self._select_voice_sample_for_type(voice_type)
            if not voice_sample_path:
                logger.error(f"Failed to select voice sample for type: {voice_type}")
                return None
            
            # First try XTTS-v2 for voice cloning
            if self.xtts_available and self.xtts_model:
                logger.info("🎙️ Using XTTS-v2 for voice cloning...")
                return self.create_podcast_segment_xtts_v2(
                    script_segment, voice_sample_path, output_dir, speed_factor, energy_level
                )
            
            # No fallback - XTTS-v2 is required for voice cloning
            logger.error("❌ XTTS-v2 not available - voice cloning is required")
            return None
            
        except Exception as e:
            logger.error(f"Failed to create podcast segment: {e}")
            return None
    
    def create_podcast_segment_xtts_v2(self, script_segment: str, voice_sample_path: str, output_dir: str = "audio_output", speed_factor: float = 1.0, energy_level: str = "medium") -> Optional[str]:
        """Create podcast segment using XTTS-v2 voice cloning."""
        try:
            if not self.xtts_available or not self.xtts_model:
                logger.error("XTTS-v2 not available - cannot proceed without voice cloning")
                return None
            
            logger.info(f"🎙️ Generating audio with XTTS-v2 voice cloning using: {voice_sample_path}")
            
            # Generate output filename
            import os
            import time
            timestamp = int(time.time())
            output_filename = f"xtts_cloned_{os.path.basename(voice_sample_path).split('.')[0]}_{timestamp}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate audio with voice cloning using XTTS-v2 API
            try:
                # Try the voice cloning method first
                self.xtts_model.tts_to_file(
                    text=script_segment,
                    file_path=output_path,
                    speaker_wav=voice_sample_path,
                    language="en"
                )
                logger.info(f"✅ XTTS-v2 voice cloning audio generated: {output_path}")
                
            except Exception as e:
                logger.warning(f"Voice cloning failed: {e}, trying without voice cloning...")
                try:
                    # Try without voice cloning - just basic TTS
                    # The model expects a speaker parameter, let's try different approaches
                    try:
                        # Try with speaker="en" (English speaker)
                        self.xtts_model.tts_to_file(
                            text=script_segment,
                            file_path=output_path,
                            speaker="en",
                            language="en"
                        )
                        logger.info(f"✅ XTTS-v2 basic audio generated with speaker='en': {output_path}")
                    except Exception as e2:
                        logger.warning(f"speaker='en' failed: {e2}")
                        # Try with speaker="en-US" (US English speaker)
                        self.xtts_model.tts_to_file(
                            text=script_segment,
                            file_path=output_path,
                            speaker="en-US",
                            language="en"
                        )
                        logger.info(f"✅ XTTS-v2 basic audio generated with speaker='en-US': {output_path}")
                    
                except Exception as e2:
                    logger.error(f"All XTTS-v2 methods failed: {e2}")
                    return None
            
            # Apply post-processing if needed
            if speed_factor != 1.0 or energy_level != "medium":
                self._apply_audio_post_processing(output_path, speed_factor, energy_level)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create XTTS-v2 podcast segment: {e}")
            return None

    def _apply_audio_post_processing(self, audio_file: str, speed_factor: float, energy_level: str):
        """Apply post-processing to enhance audio quality."""
        try:
            import soundfile as sf
            import numpy as np
            
            # Load audio
            audio, sr = sf.read(audio_file)
            
            # Convert to float32 for processing
            audio = audio.astype(np.float32)
            
            # Speed adjustment using simple resampling (more reliable than librosa)
            if speed_factor != 1.0:
                logger.info(f"⚡ Applying speed factor: {speed_factor}x")
                if speed_factor > 1.0:
                    # Speed up by skipping samples
                    step = 1.0 / speed_factor
                    indices = np.arange(0, len(audio), step, dtype=int)
                    audio = audio[indices]
                else:
                    # Slow down by repeating samples
                    repeat_factor = int(1.0 / speed_factor)
                    audio = np.repeat(audio, repeat_factor)
                logger.info(f"✅ Speed adjustment applied: {speed_factor}x")
            
            # Energy enhancement
            if energy_level == "high":
                # Increase volume by 30%
                audio = audio * 1.3
                
                # Apply soft clipping for energy (more punch)
                audio = np.tanh(audio * 1.2)
                logger.info("✅ Applied high energy processing")
            
            elif energy_level == "medium":
                # Moderate volume increase
                audio = audio * 1.1
                logger.info("✅ Applied medium energy processing")
            
            elif energy_level == "low":
                # Decrease volume for calm tone
                audio = audio * 0.8
                logger.info("✅ Applied low energy processing")
            
            # Ensure audio doesn't clip
            audio = np.clip(audio, -1.0, 1.0)
            
            # Save enhanced audio
            sf.write(audio_file, audio, sr)
            logger.info(f"✅ Audio enhancement completed: Speed={speed_factor}x, Energy={energy_level}")
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed, using original: {e}")
            # Continue with original audio if enhancement fails

    def initialize_macbook_tts_fallback(self):
        """Initialize MacBook-compatible TTS with voice cloning priority."""
        try:
            # First try XTTS-v2 for voice cloning
            logger.info("🔄 Attempting to initialize XTTS-v2 for voice cloning...")
            self._initialize_tts()
            
            if self.xtts_available and self.xtts_model:
                logger.info("✅ XTTS-v2 initialized successfully - voice cloning enabled!")
                return True
            
            # XTTS-v2 is required for voice cloning
            logger.error("❌ XTTS-v2 not available - voice cloning is required")
            self.macbook_tts_available = False
            self.xtts_available = False
            return False
            
        except Exception as e:
            logger.error(f"All TTS options failed: {e}")
            self.macbook_tts_available = False
            self.xtts_available = False
            return False 