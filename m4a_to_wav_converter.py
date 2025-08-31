#!/usr/bin/env python3
"""
M4A to WAV Converter Plugin
Standalone tool for converting M4A audio files to WAV format for voice samples.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import shutil

class M4AToWAVConverter:
    """Standalone M4A to WAV converter for voice samples."""
    
    def __init__(self, output_dir="voice_samples"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_ffmpeg(self):
        """Install FFmpeg using Homebrew."""
        print("🔄 FFmpeg not found. Installing via Homebrew...")
        try:
            subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
            print("✅ FFmpeg installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install FFmpeg. Please install manually:")
            print("   brew install ffmpeg")
            return False
        except FileNotFoundError:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
    
    def convert_single_file(self, input_path, output_name=None):
        """Convert a single M4A file to WAV."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            return False
        
        if not input_path.suffix.lower() == '.m4a':
            print(f"❌ File is not M4A format: {input_path}")
            return False
        
        # Generate output filename
        if output_name is None:
            output_name = input_path.stem + '.wav'
        elif not output_name.endswith('.wav'):
            output_name += '.wav'
        
        output_path = self.output_dir / output_name
        
        print(f"🔄 Converting: {input_path.name} → {output_name}")
        
        try:
            # FFmpeg command for high-quality conversion
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '44100',          # 44.1kHz sample rate
                '-ac', '2',              # 2 channels (stereo)
                '-y',                    # Overwrite output file
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully converted to: {output_path}")
                return str(output_path)
            else:
                print(f"❌ Conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            return False
    
    def convert_batch(self, input_dir, pattern="*.m4a"):
        """Convert all M4A files in a directory."""
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return False
        
        m4a_files = list(input_dir.glob(pattern))
        
        if not m4a_files:
            print(f"❌ No M4A files found in: {input_dir}")
            return False
        
        print(f"🔄 Found {len(m4a_files)} M4A files to convert...")
        
        successful_conversions = []
        failed_conversions = []
        
        for m4a_file in m4a_files:
            result = self.convert_single_file(m4a_file)
            if result:
                successful_conversions.append(result)
            else:
                failed_conversions.append(str(m4a_file))
        
        # Summary
        print(f"\n📊 Conversion Summary:")
        print(f"   ✅ Successful: {len(successful_conversions)}")
        print(f"   ❌ Failed: {len(failed_conversions)}")
        
        if successful_conversions:
            print(f"   📁 Output directory: {self.output_dir}")
        
        return successful_conversions
    
    def list_voice_samples(self):
        """List all available voice samples."""
        wav_files = list(self.output_dir.glob("*.wav"))
        
        if not wav_files:
            print("📁 No voice samples found.")
            return []
        
        print(f"📁 Voice samples in {self.output_dir}:")
        for i, wav_file in enumerate(wav_files, 1):
            file_size = wav_file.stat().st_size / 1024  # KB
            print(f"   {i:2d}. {wav_file.name} ({file_size:.1f} KB)")
        
        return wav_files

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="M4A to WAV Converter for Voice Samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python m4a_to_wav_converter.py convert input.m4a
  
  # Convert single file with custom name
  python m4a_to_wav_converter.py convert input.m4a --output male_voice
  
  # Convert all M4A files in a directory
  python m4a_to_wav_converter.py batch /path/to/m4a/files
  
  # List current voice samples
  python m4a_to_wav_converter.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert single file
    convert_parser = subparsers.add_parser('convert', help='Convert single M4A file')
    convert_parser.add_argument('input_file', help='Input M4A file path')
    convert_parser.add_argument('--output', '-o', help='Output filename (without .wav extension)')
    
    # Batch convert
    batch_parser = subparsers.add_parser('batch', help='Convert all M4A files in directory')
    batch_parser.add_argument('input_dir', help='Input directory containing M4A files')
    batch_parser.add_argument('--pattern', default='*.m4a', help='File pattern to match (default: *.m4a)')
    
    # List samples
    subparsers.add_parser('list', help='List current voice samples')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize converter
    converter = M4AToWAVConverter()
    
    # Check FFmpeg
    if not converter.check_ffmpeg():
        if not converter.install_ffmpeg():
            return
    
    # Execute command
    if args.command == 'convert':
        converter.convert_single_file(args.input_file, args.output)
    
    elif args.command == 'batch':
        converter.convert_batch(args.input_dir, args.pattern)
    
    elif args.command == 'list':
        converter.list_voice_samples()

if __name__ == "__main__":
    main() 