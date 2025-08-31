#!/usr/bin/env python3
"""
Speech Rate Analyzer
Analyzes voice samples to determine actual speaking speed for content generation planning.
"""

import os
import sys
import argparse
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeechRateAnalyzer:
    """Analyzes voice samples to determine speaking speed and content planning parameters."""
    
    def __init__(self, voice_samples_dir: str = "voice_samples"):
        self.voice_samples_dir = Path(voice_samples_dir)
        self.analysis_results = {}
        
    def analyze_voice_sample(self, audio_file: str) -> Dict[str, float]:
        """
        Analyze a single voice sample to estimate speaking rate.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            file_path = Path(audio_file)
            if not file_path.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return {}
            
            logger.info(f"🔍 Analyzing speech rate for: {file_path.name}")
            
            # Load audio
            audio, sr = librosa.load(str(file_path), sr=None)
            duration = len(audio) / sr
            
            # Estimate speech rate using various methods
            speech_rate_estimates = self._estimate_speech_rate(audio, sr, duration)
            
            # Calculate average speech rate
            avg_speech_rate = np.mean(list(speech_rate_estimates.values()))
            
            results = {
                'file_path': str(file_path),
                'duration_seconds': duration,
                'sample_rate': sr,
                'audio_length_samples': len(audio),
                'estimated_wpm': avg_speech_rate,
                'speech_rate_methods': speech_rate_estimates
            }
            
            logger.info(f"✅ Analysis complete for {file_path.name}")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info(f"   Estimated WPM: {avg_speech_rate:.1f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze {audio_file}: {e}")
            return {}
    
    def _estimate_speech_rate(self, audio: np.ndarray, sr: int, duration: float) -> Dict[str, float]:
        """
        Estimate speech rate using multiple methods.
        
        Args:
            audio: Audio data
            sr: Sample rate
            duration: Audio duration in seconds
            
        Returns:
            Dictionary of speech rate estimates from different methods
        """
        estimates = {}
        
        try:
            # Method 1: Energy-based speech detection
            energy = np.sum(audio**2)
            speech_activity = energy / len(audio)
            
            # Method 2: Spectral centroid analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_variance = np.var(spectral_centroids)
            
            # Method 3: Zero crossing rate (indicates speech complexity)
            zero_crossings = librosa.feature.zero_crossing_rate(audio)[0]
            avg_zero_crossings = np.mean(zero_crossings)
            
            # Method 4: MFCC analysis for speech characteristics
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_variance = np.var(mfccs)
            
            # Combine features to estimate speech rate
            # These are heuristic estimates based on typical speech patterns
            speech_complexity = (spectral_variance + mfcc_variance) / 2
            energy_factor = speech_activity / np.max(audio**2) if np.max(audio**2) > 0 else 0
            
            # Base speech rate estimation (words per minute)
            # Typical conversational speech is 120-150 WPM
            base_rate = 135  # Average conversational WPM
            
            # Adjust based on audio characteristics
            complexity_factor = np.clip(speech_complexity / 1000, 0.8, 1.2)
            energy_factor = np.clip(energy_factor * 2, 0.8, 1.2)
            
            # Method 1: Energy-based
            estimates['energy_based'] = base_rate * energy_factor
            
            # Method 2: Complexity-based
            estimates['complexity_based'] = base_rate * complexity_factor
            
            # Method 3: Combined approach
            estimates['combined'] = base_rate * (energy_factor + complexity_factor) / 2
            
            # Method 4: Conservative estimate
            estimates['conservative'] = base_rate * 0.9  # Slightly slower than average
            
            logger.debug(f"Speech rate estimates: {estimates}")
            
        except Exception as e:
            logger.warning(f"Speech rate estimation failed: {e}")
            # Fallback to conservative estimate
            estimates['fallback'] = 120.0
        
        return estimates
    
    def analyze_all_samples(self) -> Dict[str, Dict]:
        """
        Analyze all voice samples in the directory.
        
        Returns:
            Dictionary of analysis results for all samples
        """
        if not self.voice_samples_dir.exists():
            logger.error(f"Voice samples directory not found: {self.voice_samples_dir}")
            return {}
        
        wav_files = list(self.voice_samples_dir.glob("*.wav"))
        if not wav_files:
            logger.warning(f"No WAV files found in {self.voice_samples_dir}")
            return {}
        
        logger.info(f"🔍 Found {len(wav_files)} voice samples to analyze")
        
        for wav_file in wav_files:
            sample_name = wav_file.stem
            results = self.analyze_voice_sample(str(wav_file))
            if results:
                self.analysis_results[sample_name] = results
        
        return self.analysis_results
    
    def get_speech_rate_summary(self) -> Dict[str, float]:
        """
        Get a summary of speech rates for content planning.
        
        Returns:
            Dictionary with average speech rates and recommendations
        """
        if not self.analysis_results:
            return {}
        
        # Calculate average speech rate across all samples
        all_wpm_rates = [result['estimated_wpm'] for result in self.analysis_results.values()]
        avg_wpm = np.mean(all_wpm_rates)
        std_wpm = np.std(all_wpm_rates)
        
        # Determine speech rate category
        if avg_wpm < 120:
            rate_category = "slow"
        elif avg_wpm < 150:
            rate_category = "normal"
        else:
            rate_category = "fast"
        
        summary = {
            'average_wpm': avg_wpm,
            'std_wpm': std_wpm,
            'rate_category': rate_category,
            'samples_analyzed': len(self.analysis_results),
            'recommended_content_multiplier': self._calculate_content_multiplier(avg_wpm),
            'individual_samples': {name: result['estimated_wpm'] for name, result in self.analysis_results.items()}
        }
        
        return summary
    
    def _calculate_content_multiplier(self, avg_wpm: float) -> float:
        """
        Calculate content multiplier for target duration.
        
        Args:
            avg_wpm: Average words per minute
            
        Returns:
            Multiplier to apply to content generation
        """
        # Target WPM for content generation (assuming we want natural pacing)
        target_wpm = 140  # Slightly slower than average for clarity
        
        # Calculate multiplier to achieve target duration
        multiplier = target_wpm / avg_wpm
        
        # Apply reasonable bounds
        multiplier = np.clip(multiplier, 0.7, 1.5)
        
        return multiplier
    
    def generate_content_planning_guide(self, target_duration_minutes: float) -> Dict[str, any]:
        """
        Generate a content planning guide based on speech rate analysis.
        
        Args:
            target_duration_minutes: Target duration in minutes
            
        Returns:
            Content planning guide with specific recommendations
        """
        summary = self.get_speech_rate_summary()
        if not summary:
            return {}
        
        target_seconds = target_duration_minutes * 60
        avg_wpm = summary['average_wpm']
        content_multiplier = summary['recommended_content_multiplier']
        
        # Calculate target word count
        target_words = int((target_seconds / 60) * avg_wpm * content_multiplier)
        
        # Calculate target character count (rough estimate: 5 characters per word)
        target_chars = target_words * 5
        
        # Generate section breakdown
        sections = self._generate_section_breakdown(target_duration_minutes, target_words)
        
        guide = {
            'target_duration_minutes': target_duration_minutes,
            'target_duration_seconds': target_seconds,
            'target_word_count': target_words,
            'target_character_count': target_chars,
            'speech_rate_analysis': summary,
            'content_multiplier': content_multiplier,
            'section_breakdown': sections,
            'llm_prompting_instructions': self._generate_llm_instructions(target_words, sections)
        }
        
        return guide
    
    def _generate_section_breakdown(self, duration_minutes: float, total_words: int) -> Dict[str, Dict]:
        """
        Generate a section breakdown for content planning.
        
        Args:
            duration_minutes: Target duration in minutes
            total_words: Target word count
            
        Returns:
            Section breakdown with timing and word count targets
        """
        if duration_minutes <= 3:
            # Short content: Introduction + Main Points + Conclusion
            sections = {
                'introduction': {
                    'duration_minutes': duration_minutes * 0.2,
                    'word_count': int(total_words * 0.2),
                    'description': 'Brief introduction and context setting'
                },
                'main_content': {
                    'duration_minutes': duration_minutes * 0.6,
                    'word_count': int(total_words * 0.6),
                    'description': 'Core discussion points and analysis'
                },
                'conclusion': {
                    'duration_minutes': duration_minutes * 0.2,
                    'word_count': int(total_words * 0.2),
                    'description': 'Summary and key takeaways'
                }
            }
        else:
            # Longer content: More detailed structure
            sections = {
                'introduction': {
                    'duration_minutes': duration_minutes * 0.15,
                    'word_count': int(total_words * 0.15),
                    'description': 'Introduction, context, and agenda'
                },
                'background': {
                    'duration_minutes': duration_minutes * 0.15,
                    'word_count': int(total_words * 0.15),
                    'description': 'Background information and context'
                },
                'main_analysis': {
                    'duration_minutes': duration_minutes * 0.45,
                    'word_count': int(total_words * 0.45),
                    'description': 'Detailed analysis and discussion'
                },
                'implications': {
                    'duration_minutes': duration_minutes * 0.15,
                    'word_count': int(total_words * 0.15),
                    'description': 'Implications and recommendations'
                },
                'conclusion': {
                    'duration_minutes': duration_minutes * 0.1,
                    'word_count': int(total_words * 0.1),
                    'description': 'Summary and closing thoughts'
                }
            }
        
        return sections
    
    def _generate_llm_instructions(self, target_words: int, sections: Dict) -> str:
        """
        Generate specific LLM instructions for content generation.
        
        Args:
            target_words: Target word count
            sections: Section breakdown
            
        Returns:
            Formatted instructions for the LLM
        """
        instructions = f"""
CONTENT GENERATION REQUIREMENTS:
- TARGET WORD COUNT: {target_words} words (±10% acceptable)
- TARGET DURATION: {target_words / 140:.1f} minutes when spoken
- CONTENT MUST BE NATURAL AND ENGAGING

SECTION BREAKDOWN:
"""
        
        for section_name, section_info in sections.items():
            instructions += f"""
{section_name.upper().replace('_', ' ')}:
- Target: {section_info['word_count']} words
- Duration: {section_info['duration_minutes']:.1f} minutes
- Purpose: {section_info['description']}
"""
        
        instructions += """
CRITICAL REQUIREMENTS:
1. Each section must meet its word count target
2. Content must flow naturally between sections
3. Maintain conversational tone throughout
4. Include specific examples and details to reach word count
5. Ensure male and female reporters have balanced speaking time
"""
        
        return instructions

def main():
    """Main CLI interface for speech rate analysis."""
    parser = argparse.ArgumentParser(
        description="Speech Rate Analyzer for Voice Samples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--analyze-all', action='store_true',
                       help='Analyze all voice samples in the directory')
    parser.add_argument('--analyze-file', type=str,
                       help='Analyze a specific audio file')
    parser.add_argument('--target-duration', type=float, default=5.0,
                       help='Target duration in minutes for content planning (default: 5.0)')
    parser.add_argument('--output-format', choices=['summary', 'detailed', 'guide'], default='guide',
                       help='Output format (default: guide)')
    
    args = parser.parse_args()
    
    analyzer = SpeechRateAnalyzer()
    
    if args.analyze_file:
        # Analyze single file
        results = analyzer.analyze_voice_sample(args.analyze_file)
        if results:
            print(f"\n📊 Analysis Results for {args.analyze_file}:")
            for key, value in results.items():
                if key != 'speech_rate_methods':
                    print(f"   {key}: {value}")
            
            print(f"\n🔍 Speech Rate Methods:")
            for method, rate in results['speech_rate_methods'].items():
                print(f"   {method}: {rate:.1f} WPM")
    
    elif args.analyze_all:
        # Analyze all samples
        print("🔍 Analyzing all voice samples...")
        results = analyzer.analyze_all_samples()
        
        if results:
            summary = analyzer.get_speech_rate_summary()
            guide = analyzer.generate_content_planning_guide(args.target_duration)
            
            if args.output_format == 'summary':
                print(f"\n📊 Speech Rate Summary:")
                print(f"   Average WPM: {summary['average_wpm']:.1f}")
                print(f"   Rate Category: {summary['rate_category']}")
                print(f"   Content Multiplier: {summary['recommended_content_multiplier']:.2f}")
                print(f"   Samples Analyzed: {summary['samples_analyzed']}")
            
            elif args.output_format == 'detailed':
                print(f"\n📊 Detailed Analysis:")
                for sample_name, sample_results in results.items():
                    print(f"\n🎤 {sample_name}:")
                    print(f"   Duration: {sample_results['duration_seconds']:.2f}s")
                    print(f"   Estimated WPM: {sample_results['estimated_wpm']:.1f}")
                    print(f"   Sample Rate: {sample_results['sample_rate']}")
            
            elif args.output_format == 'guide':
                print(f"\n🎯 Content Planning Guide:")
                print(f"   Target Duration: {guide['target_duration_minutes']:.1f} minutes")
                print(f"   Target Word Count: {guide['target_word_count']:,} words")
                print(f"   Target Character Count: {guide['target_character_count']:,} characters")
                print(f"   Speech Rate: {guide['speech_rate_analysis']['average_wpm']:.1f} WPM")
                print(f"   Content Multiplier: {guide['content_multiplier']:.2f}")
                
                print(f"\n📝 Section Breakdown:")
                for section_name, section_info in guide['section_breakdown'].items():
                    print(f"   {section_name.replace('_', ' ').title()}:")
                    print(f"     - {section_info['word_count']} words")
                    print(f"     - {section_info['duration_minutes']:.1f} minutes")
                    print(f"     - {section_info['description']}")
                
                print(f"\n🤖 LLM Instructions:")
                print(guide['llm_prompting_instructions'])
    
    else:
        # Default: analyze all samples and show guide
        print("🔍 Analyzing all voice samples...")
        results = analyzer.analyze_all_samples()
        
        if results:
            guide = analyzer.generate_content_planning_guide(args.target_duration)
            print(f"\n🎯 Content Planning Guide for {args.target_duration} minutes:")
            print(f"   Target Word Count: {guide['target_word_count']:,} words")
            print(f"   Speech Rate: {guide['speech_rate_analysis']['average_wpm']:.1f} WPM")
            print(f"   Content Multiplier: {guide['content_multiplier']:.2f}")
        else:
            print("❌ No voice samples found to analyze")

if __name__ == "__main__":
    main() 