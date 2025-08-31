#!/usr/bin/env python3
"""
Complete Podcast Generator
Generates a full podcast using XTTS-v2 voice cloning with multiple speakers
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time

# Import our custom modules
from voice_sample_manager import VoiceSampleManager
from xtts_integration import XTTSIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePodcastGenerator:
    """Complete podcast generation system with voice cloning."""
    
    def __init__(self):
        self.voice_manager = VoiceSampleManager()
        self.xtts = XTTSIntegration()
        self.output_dir = Path("generated_podcasts")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize XTTS
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the complete system."""
        logger.info("🚀 Initializing Complete Podcast Generator...")
        
        # Check voice samples
        voices = self.voice_manager.get_available_samples()
        if not voices:
            logger.info("Creating default voice samples...")
            self.voice_manager.create_default_samples()
            voices = self.voice_manager.get_available_samples()
        
        logger.info(f"✅ Voice samples ready: {len(voices)} available")
        
        # Initialize XTTS
        if not self.xtts.initialize_xtts():
            logger.error("Failed to initialize XTTS-v2")
            raise RuntimeError("XTTS-v2 initialization failed")
        
        logger.info("✅ System initialized successfully")
    
    def create_podcast_script(self, topic: str, duration_minutes: int = 10) -> List[Dict]:
        """Create a podcast script structure."""
        # Estimate words per minute (average speaking rate)
        words_per_minute = 150
        total_words = duration_minutes * words_per_minute
        
        # Create script structure
        script = [
            {
                "type": "intro",
                "speaker": "male_reporter_default",
                "text": f"Welcome to our podcast! Today we're discussing {topic}. I'm your host, and I'm excited to dive into this fascinating topic with you.",
                "duration_estimate": 15
            },
            {
                "type": "main_content",
                "speaker": "male_reporter_default", 
                "text": f"Let me start by giving you an overview of {topic}. This is a subject that has captured the attention of many people in recent years.",
                "duration_estimate": 20
            },
            {
                "type": "guest_intro",
                "speaker": "female_reporter_default",
                "text": "Now, let me introduce our special guest who has extensive experience in this field. Welcome to the show!",
                "duration_estimate": 12
            },
            {
                "type": "guest_speaking",
                "speaker": "female_reporter_default",
                "text": f"Thank you for having me! I'm thrilled to discuss {topic} with your audience. This is such an important and timely subject.",
                "duration_estimate": 18
            },
            {
                "type": "discussion",
                "speaker": "male_reporter_default",
                "text": "That's fascinating! Can you tell us more about the key aspects that people should understand about this topic?",
                "duration_estimate": 15
            },
            {
                "type": "guest_response",
                "speaker": "female_reporter_default",
                "text": "Absolutely! There are several key points that are crucial to understand. First, the fundamentals are essential to grasp.",
                "duration_estimate": 25
            },
            {
                "type": "conclusion",
                "speaker": "male_reporter_default",
                "text": "Thank you for sharing your insights with us today. This has been an incredibly informative discussion about {topic}.",
                "duration_estimate": 15
            },
            {
                "type": "outro",
                "speaker": "male_reporter_default",
                "text": "That wraps up today's episode. Thank you for listening, and we'll see you next time with another exciting topic!",
                "duration_estimate": 10
            }
        ]
        
        # Calculate total estimated duration
        total_seconds = sum(segment["duration_estimate"] for segment in script)
        logger.info(f"Script created: {len(script)} segments, estimated duration: {total_seconds} seconds")
        
        return script
    
    def generate_podcast(self, topic: str, duration_minutes: int = 10, 
                        output_filename: str = None) -> Dict:
        """Generate a complete podcast with voice cloning."""
        logger.info(f"🎙️ Generating podcast: {topic}")
        
        # Create script
        script = self.create_podcast_script(topic, duration_minutes)
        
        # Generate audio for each segment
        generated_files = []
        total_duration = 0
        
        for i, segment in enumerate(script):
            logger.info(f"Generating segment {i+1}/{len(script)}: {segment['type']}")
            
            output_path = self.xtts.create_podcast_segment(
                script_segment=segment['text'],
                voice_type=segment['speaker'],
                output_dir=str(self.output_dir)
            )
            
            if output_path:
                segment['audio_file'] = output_path
                generated_files.append(output_path)
                
                # Get actual duration
                import librosa
                audio, sr = librosa.load(output_path, sr=22050)
                actual_duration = len(audio) / sr
                segment['actual_duration'] = actual_duration
                total_duration += actual_duration
                
                logger.info(f"✅ Segment {i+1} generated: {actual_duration:.1f}s")
            else:
                logger.error(f"❌ Failed to generate segment {i+1}")
        
        # Create podcast metadata
        podcast_info = {
            "topic": topic,
            "target_duration_minutes": duration_minutes,
            "actual_duration_seconds": total_duration,
            "actual_duration_minutes": total_duration / 60,
            "segments": script,
            "generated_files": generated_files,
            "generation_timestamp": time.time(),
            "voice_samples_used": list(self.voice_manager.get_available_samples().keys())
        }
        
        # Save metadata
        if not output_filename:
            timestamp = int(time.time())
            output_filename = f"podcast_{topic.replace(' ', '_')}_{timestamp}"
        
        metadata_file = self.output_dir / f"{output_filename}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(podcast_info, f, indent=2)
        
        logger.info(f"🎉 Podcast generation completed!")
        logger.info(f"   Topic: {topic}")
        logger.info(f"   Duration: {total_duration/60:.1f} minutes")
        logger.info(f"   Segments: {len(generated_files)}")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   Metadata: {metadata_file}")
        
        return podcast_info
    
    def create_custom_podcast(self, script_data: List[Dict], output_filename: str = None) -> Dict:
        """Create a podcast from custom script data."""
        logger.info("🎙️ Generating custom podcast from script")
        
        # Validate script structure
        for i, segment in enumerate(script_data):
            if 'text' not in segment or 'speaker' not in segment:
                logger.error(f"Invalid segment {i}: missing text or speaker")
                return None
        
        # Generate audio
        generated_files = []
        total_duration = 0
        
        for i, segment in enumerate(script_data):
            logger.info(f"Generating custom segment {i+1}/{len(script_data)}")
            
            output_path = self.xtts.create_podcast_segment(
                script_segment=segment['text'],
                voice_type=segment['speaker'],
                output_dir=str(self.output_dir)
            )
            
            if output_path:
                segment['audio_file'] = output_path
                generated_files.append(output_path)
                
                # Get actual duration
                import librosa
                audio, sr = librosa.load(output_path, sr=22050)
                actual_duration = len(audio) / sr
                segment['actual_duration'] = actual_duration
                total_duration += actual_duration
                
                logger.info(f"✅ Custom segment {i+1} generated: {actual_duration:.1f}s")
            else:
                logger.error(f"❌ Failed to generate custom segment {i+1}")
        
        # Create podcast metadata
        podcast_info = {
            "type": "custom_podcast",
            "actual_duration_seconds": total_duration,
            "actual_duration_minutes": total_duration / 60,
            "segments": script_data,
            "generated_files": generated_files,
            "generation_timestamp": time.time(),
            "voice_samples_used": list(self.voice_manager.get_available_samples().keys())
        }
        
        # Save metadata
        if not output_filename:
            timestamp = int(time.time())
            output_filename = f"custom_podcast_{timestamp}"
        
        metadata_file = self.output_dir / f"{output_filename}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(podcast_info, f, indent=2)
        
        logger.info(f"🎉 Custom podcast generation completed!")
        logger.info(f"   Duration: {total_duration/60:.1f} minutes")
        logger.info(f"   Segments: {len(generated_files)}")
        logger.info(f"   Output directory: {self.output_dir}")
        
        return podcast_info
    
    def list_generated_podcasts(self) -> List[Dict]:
        """List all generated podcasts with metadata."""
        podcasts = []
        
        for metadata_file in self.output_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    podcast_data = json.load(f)
                    podcast_data['metadata_file'] = str(metadata_file)
                    podcasts.append(podcast_data)
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        
        return podcasts
    
    def get_system_status(self) -> Dict:
        """Get complete system status."""
        return {
            "voice_samples": len(self.voice_manager.get_available_samples()),
            "xtts_initialized": self.xtts.tts is not None,
            "output_directory": str(self.output_dir),
            "generated_podcasts": len(self.list_generated_podcasts()),
            "available_voices": list(self.voice_manager.get_available_samples().keys())
        }


def main():
    """Main demo function."""
    print("🎙️ Complete Podcast Generator Demo")
    print("=" * 60)
    
    try:
        # Initialize the system
        generator = CompletePodcastGenerator()
        
        # Show system status
        status = generator.get_system_status()
        print(f"\n📊 System Status:")
        print(f"  Voice samples: {status['voice_samples']}")
        print(f"  XTTS initialized: {'✅ Yes' if status['xtts_initialized'] else '❌ No'}")
        print(f"  Available voices: {', '.join(status['available_voices'])}")
        
        # Generate a sample podcast
        print(f"\n🎬 Generating sample podcast...")
        podcast_info = generator.generate_podcast(
            topic="Artificial Intelligence in Modern Technology",
            duration_minutes=5
        )
        
        if podcast_info:
            print(f"\n✅ Sample podcast generated successfully!")
            print(f"   Duration: {podcast_info['actual_duration_minutes']:.1f} minutes")
            print(f"   Segments: {len(podcast_info['segments'])}")
            print(f"   Output directory: {generator.output_dir}")
        
        # Show custom podcast example
        print(f"\n📝 Custom Podcast Example:")
        custom_script = [
            {
                "speaker": "male_reporter_default",
                "text": "Hello everyone! This is a custom podcast segment.",
                "type": "custom_intro"
            },
            {
                "speaker": "female_reporter_default", 
                "text": "And I'm here to show you how flexible this system can be!",
                "type": "custom_response"
            }
        ]
        
        print("Ready to generate custom podcasts!")
        print("Use: generator.create_custom_podcast(custom_script)")
        
        # Show usage examples
        print(f"\n💡 Usage Examples:")
        print(f"  1. Generate topic-based: generator.generate_podcast('Your Topic', duration_minutes)")
        print(f"  2. Custom script: generator.create_custom_podcast(script_data)")
        print(f"  3. List podcasts: generator.list_generated_podcasts()")
        print(f"  4. System status: generator.get_system_status()")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 