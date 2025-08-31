#!/usr/bin/env python3
"""
Voice Sample Manager for XTTS-v2
Provides various ways to get and manage voice samples for voice cloning
"""

import os
import requests
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class VoiceSampleManager:
    """Manages voice samples for XTTS-v2 voice cloning."""
    
    def __init__(self, samples_dir: str = "voice_samples"):
        self.samples_dir = Path(samples_dir)
        self.samples_dir.mkdir(exist_ok=True)
        self.sample_registry = {}
        self._load_existing_samples()
    
    def _load_existing_samples(self):
        """Load existing voice samples from the samples directory."""
        # Clear existing registry
        self.sample_registry.clear()
        
        for audio_file in self.samples_dir.glob("*.wav"):
            self.sample_registry[audio_file.stem] = str(audio_file)
        logger.info(f"Loaded {len(self.sample_registry)} existing voice samples")
    
    def get_available_samples(self) -> Dict[str, str]:
        """Get all available voice samples."""
        return self.sample_registry.copy()
    
    def download_sample_from_url(self, url: str, filename: str) -> Optional[str]:
        """Download a voice sample from a URL."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = self.samples_dir / f"{filename}.wav"
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.sample_registry[filename] = str(filepath)
            logger.info(f"Downloaded voice sample: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download sample from {url}: {e}")
            return None
    
    def create_sample_from_text(self, text: str, filename: str, voice_type: str = "neutral") -> Optional[str]:
        """Create a voice sample - gTTS removed, manual samples required."""
        logger.warning("gTTS removed - please provide manual voice samples for voice cloning")
        return None
    
    def get_sample_for_speaker(self, speaker_type: str) -> Optional[str]:
        """Get the best available sample for a speaker type."""
        # Try to find a matching sample
        for name, path in self.sample_registry.items():
            if speaker_type.lower() in name.lower():
                return path
        
        # Fallback to first available sample
        if self.sample_registry:
            return list(self.sample_registry.values())[0]
        
        return None
    
    def create_default_samples(self) -> Dict[str, str]:
        """Create default voice samples for common speaker types."""
        default_samples = {}
        
        # Male reporter sample
        male_text = "Hello, this is a test recording for voice cloning. I'm speaking naturally and clearly for the male reporter voice."
        male_path = self.create_sample_from_text(male_text, "male_reporter_default")
        if male_path:
            default_samples["male_reporter"] = male_path
        
        # Female reporter sample
        female_text = "Hello, this is a test recording for voice cloning. I'm speaking naturally and clearly for the female reporter voice."
        female_path = self.create_sample_from_text(female_text, "female_reporter_default")
        if female_path:
            default_samples["female_reporter"] = female_path
        
        # Neutral sample
        neutral_text = "Hello, this is a test recording for voice cloning. I'm speaking naturally and clearly for general use."
        neutral_path = self.create_sample_from_text(neutral_text, "neutral_default")
        if neutral_path:
            default_samples["neutral"] = neutral_path
        
        logger.info(f"Created {len(default_samples)} default voice samples")
        return default_samples
    
    def validate_sample(self, filepath: str) -> bool:
        """Validate that a voice sample is suitable for XTTS-v2."""
        try:
            import librosa
            
            # Load audio
            audio, sr = librosa.load(filepath, sr=22050)
            
            # Check duration (3-30 seconds is ideal)
            duration = len(audio) / sr
            if duration < 2 or duration > 60:
                logger.warning(f"Sample duration {duration:.1f}s is outside ideal range (2-60s)")
                return False
            
            # Check if it's mostly speech (not silence)
            # Simple energy-based check
            energy = librosa.feature.rms(y=audio)[0]
            if energy.mean() < 0.01:
                logger.warning("Sample appears to be mostly silence")
                return False
            
            logger.info(f"Sample validation passed: {duration:.1f}s, good energy")
            return True
            
        except Exception as e:
            logger.error(f"Sample validation failed: {e}")
            return False
    
    def get_sample_info(self, filepath: str) -> Dict[str, any]:
        """Get detailed information about a voice sample."""
        try:
            import librosa
            
            audio, sr = librosa.load(filepath, sr=22050)
            duration = len(audio) / sr
            
            # Analyze audio quality
            energy = librosa.feature.rms(y=audio)[0]
            zero_crossings = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(duration, energy.mean(), zero_crossings.mean())
            
            return {
                "filepath": filepath,
                "duration": duration,
                "sample_rate": sr,
                "channels": 1,
                "file_size": os.path.getsize(filepath),
                "valid": self.validate_sample(filepath),
                "energy_mean": float(energy.mean()),
                "energy_std": float(energy.std()),
                "zero_crossing_rate": float(zero_crossings.mean()),
                "quality_score": quality_score
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_quality_score(self, duration: float, energy: float, zero_crossing: float) -> float:
        """Calculate a quality score for voice samples."""
        # Duration score (ideal: 3-30 seconds)
        if 3 <= duration <= 30:
            duration_score = 1.0
        elif 2 <= duration <= 60:
            duration_score = 0.7
        else:
            duration_score = 0.3
        
        # Energy score (should be above threshold)
        energy_score = min(energy * 100, 1.0)
        
        # Zero crossing score (indicates speech vs noise)
        zc_score = min(zero_crossing * 10, 1.0)
        
        # Weighted average
        return (duration_score * 0.4 + energy_score * 0.4 + zc_score * 0.2)
    
    def create_custom_sample(self, text: str, filename: str, description: str = "") -> Optional[str]:
        """Create a custom voice sample with specific text."""
        return self.create_sample_from_text(text, filename)
    
    def list_online_sources(self) -> List[str]:
        """List online sources where you can find voice samples."""
        return [
            "Freesound.org - Creative Commons licensed audio",
            "Internet Archive - Public domain recordings", 
            "LibriVox - Public domain audiobooks",
            "Common Voice - Mozilla's open voice dataset",
            "YouTube (with permission) - Various voice types",
            "Podcast archives - Professional voices",
            "Audiobook samples - Clear, professional speech"
        ]

    def refresh_samples(self):
        """Force refresh the voice samples from disk."""
        logger.info("🔄 Refreshing voice samples from disk...")
        self._load_existing_samples()
        return self.get_available_samples()


def main():
    """Demo the voice sample manager."""
    print("🎙️ Voice Sample Manager Demo")
    
    manager = VoiceSampleManager()
    
    # Show existing samples
    print(f"\n📁 Existing samples: {len(manager.get_available_samples())}")
    
    # Create default samples if none exist
    if not manager.get_available_samples():
        print("\n🔄 Creating default voice samples...")
        default_samples = manager.create_default_samples()
        print(f"✅ Created {len(default_samples)} default samples")
    
    # Show all samples
    samples = manager.get_available_samples()
    print(f"\n📋 Available voice samples:")
    for name, path in samples.items():
        info = manager.get_sample_info(path)
        if "error" not in info:
            print(f"  {name}: {info['duration']:.1f}s, {'✅ Valid' if info['valid'] else '❌ Invalid'}")
    
    # Show online sources
    print(f"\n🌐 Online voice sample sources:")
    for source in manager.list_online_sources():
        print(f"  • {source}")
    
    # Show how to use with XTTS-v2
    print(f"\n💡 Usage with XTTS-v2:")
    print(f"  male_sample = manager.get_sample_for_speaker('male_reporter')")
    print(f"  female_sample = manager.get_sample_for_speaker('female_reporter')")
    
    if samples:
        male_sample = manager.get_sample_for_speaker('male_reporter')
        if male_sample:
            print(f"\n🎯 Male reporter sample: {male_sample}")
        
        female_sample = manager.get_sample_for_speaker('female_reporter')
        if female_sample:
            print(f"🎯 Female reporter sample: {female_sample}")
    
    # Show next steps
    print(f"\n🚀 Next Steps:")
    print(f"  1. Use existing samples with XTTS-v2")
    print(f"  2. Create custom samples with specific text")
    print(f"  3. Download samples from online sources")
    print(f"  4. Record your own voice samples")


if __name__ == "__main__":
    main() 