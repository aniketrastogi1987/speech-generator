#!/usr/bin/env python3
"""
Complete memory-efficient podcast script generation system.
Integrates PDF processing, lightweight RAG, LLM generation, and TTS capabilities.
"""

import os
import sys
import gc
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from memory_efficient_pdf_processor import MemoryEfficientPDFProcessor
from lightweight_rag import LightweightRAGEngine
from prompt_manager import PromptManager
from llm_interface import LLMInterface
from xtts_integration import XTTSIntegration

# Import speech rate analyzer
try:
    from speech_rate_analyzer import SpeechRateAnalyzer
    SPEECH_RATE_AVAILABLE = True
except ImportError:
    SPEECH_RATE_AVAILABLE = False
    print("⚠️ Speech rate analyzer not available - using default content generation")

from loguru import logger

class CompleteLightweightSystem:
    """Complete memory-efficient podcast script generation system."""
    
    def __init__(self, 
                 dataset_folder: str = "dataset",
                 prompts_folder: str = "prompts",
                 output_folder: str = "output",
                 rag_db_path: str = "lightweight_rag_db"):
        self.dataset_folder = Path(dataset_folder)
        self.prompts_folder = Path(prompts_folder)
        self.output_folder = Path(output_folder)
        self.rag_db_path = Path(rag_db_path)
        
        # Initialize components
        self.pdf_processor = None
        self.rag_engine = None
        self.prompt_manager = None
        self.llm_interface = None
        self.tts_integration = None
        
        # Speech rate optimization settings
        self.max_content_iterations = 5
        self.content_length_tolerance = 0.15  # 15% tolerance for over/under target
        
        self._initialize_components()
        self._ensure_directories()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing system components...")
            
            # Initialize PDF processor
            self.pdf_processor = MemoryEfficientPDFProcessor(str(self.dataset_folder))
            logger.info("✅ PDF processor initialized")
            
            # Initialize RAG engine
            self.rag_engine = LightweightRAGEngine(str(self.rag_db_path))
            logger.info("✅ RAG engine initialized")
            
            # Initialize prompt manager
            self.prompt_manager = PromptManager(str(self.prompts_folder))
            logger.info("✅ Prompt manager initialized")
            
            # Initialize LLM interface
            self.llm_interface = LLMInterface()
            logger.info("✅ LLM interface initialized")
            
            # Initialize TTS integration
            self.tts_integration = XTTSIntegration(str(self.output_folder / "audio"))
            logger.info("✅ TTS integration initialized")
            
            # Initialize speech rate analyzer if available
            if SPEECH_RATE_AVAILABLE:
                self.speech_rate_analyzer = SpeechRateAnalyzer()
                logger.info("✅ Speech rate analyzer initialized")
            else:
                logger.warning("⚠️ Speech rate analyzer not available - using default settings")
            
            logger.info("🎉 All system components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.output_folder,
            self.output_folder / "audio",
            self.output_folder / "scripts",
            Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def process_dataset(self, pdf_chunk_size: int = 5000, text_chunk_size: int = 500, overlap: int = 100) -> bool:
        """Process all PDFs in the dataset folder and build RAG index."""
        try:
            logger.info("🔄 Starting dataset processing...")
            
            # Process PDFs - use the correct parameter name
            logger.info("📄 Processing PDF documents...")
            pdf_results = self.pdf_processor.process_all_pdfs_chunked(
                chunk_size=pdf_chunk_size  # Fixed: use chunk_size instead of pdf_chunk_size
            )
            
            if not pdf_results:
                logger.warning("No PDFs were processed successfully")
                return False
            
            logger.info(f"✅ Processed {len(pdf_results)} PDF documents")
            
            # Create text chunks from the extracted text
            logger.info("🔍 Creating text chunks...")
            all_text_chunks = []
            
            for pdf_name, pdf_text in pdf_results.items():
                # Create chunks from this PDF text
                chunks = self.pdf_processor.get_text_chunks_efficient(
                    chunk_size=text_chunk_size,
                    overlap=overlap
                )
                
                # Add filename to chunks if not already present
                for chunk in chunks:
                    if 'filename' not in chunk:
                        chunk['filename'] = pdf_name
                
                all_text_chunks.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {pdf_name}")
            
            if not all_text_chunks:
                logger.warning("No text chunks were created")
                return False
            
            logger.info(f"✅ Created {len(all_text_chunks)} total text chunks")
            
            # Build RAG index
            logger.info("🔍 Building RAG index...")
            self.rag_engine.add_documents(all_text_chunks)
            
            # Save RAG index
            self.rag_engine._save_documents()
            logger.info("✅ RAG index built and saved successfully")
            
            # Get statistics
            stats = self.rag_engine.get_collection_stats()
            logger.info(f"📊 RAG collection stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            return False
    
    def generate_script(self, prompt_name: str, length: str = "medium") -> Optional[str]:
        """Generate a podcast script using the specified prompt."""
        try:
            logger.info(f"🎭 Generating script with prompt: {prompt_name}")
            
            # Get prompt template
            prompt_template = self.prompt_manager.get_prompt(prompt_name)
            if not prompt_template:
                logger.error(f"Prompt '{prompt_name}' not found")
                return None
            
            # Get relevant context from RAG
            logger.info("🔍 Retrieving relevant context...")
            context = self.rag_engine.get_context_for_prompt(prompt_template, max_chars=2000)
            
            if not context:
                logger.warning("No relevant context found from RAG")
                context = "No specific context available. Generate a general podcast script."
            
            # Inject context into prompt
            final_prompt = self.prompt_manager.get_prompt_with_context(prompt_name, context)
            
            # Generate CONVERSATIONAL script using LLM - alternating between male and female reporters
            logger.info("🤖 Generating interactive conversation script using LLM...")
            script = self.llm_interface.generate_conversation(
                prompt=final_prompt,
                context=context,
                conversation_length=length
            )
            
            if not script:
                logger.error("LLM failed to generate conversation script")
                return None
            
            logger.info("✅ Interactive conversation script generated successfully!")
            return script
            
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            return None
    
    def save_script(self, script: str, prompt_name: str) -> Optional[str]:
        """Save the generated script to a file."""
        try:
            # Create filename
            timestamp = self._get_timestamp()
            filename = f"script_{prompt_name}_{timestamp}.txt"
            filepath = self.output_folder / "scripts" / filename
            
            # Save script
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(script)
            
            logger.info(f"💾 Script saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save script: {e}")
            return None
    
    def save_script_with_metadata(self, script: str, prompt_name: str, preferences: dict = None) -> Optional[str]:
        """Save the generated script with metadata including user preferences."""
        try:
            # Create filename
            timestamp = self._get_timestamp()
            filename = f"script_{prompt_name}_{timestamp}.txt"
            filepath = self.output_folder / "scripts" / filename
            
            # Save script
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(script)
            
            # Save metadata if preferences provided
            if preferences:
                metadata_file = filepath.with_suffix('.json')
                import json
                metadata = {
                    "script_file": filename,
                    "prompt_name": prompt_name,
                    "timestamp": timestamp,
                    "preferences": preferences,
                    "script_length": len(script),
                    "segments": len([p for p in script.split('\n\n') if p.strip()])
                }
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"💾 Script and metadata saved to: {filepath} and {metadata_file}")
            else:
                logger.info(f"💾 Script saved to: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save script with metadata: {e}")
            return None
    
    def generateAndSave(self, prompt_name: str, length: str = "medium") -> Optional[str]:
        """Generate and save a script in one operation."""
        try:
            # Generate script
            script = self.generate_script(prompt_name, length)
            if not script:
                return None
            
            # Save script
            filepath = self.save_script(script, prompt_name)
            return filepath
            
        except Exception as e:
            logger.error(f"Generate and save failed: {e}")
            return None
    
    def generate_script_with_preferences(self, prompt_name: str, preferences: dict) -> str:
        """
        Generate script with user preferences and speech rate optimization.
        
        Args:
            prompt_name: Name of the prompt to use
            preferences: User preferences including content guide
            
        Returns:
            Generated script
        """
        try:
            # Extract target duration
            target_duration = preferences.get('conversation_length', 'medium')
            duration_mapping = {
                'summary': 2.5, 
                'short': 4.0, 
                'medium': 6.5, 
                'long': 10.0, 
                'detailed': 15.0
            }
            target_minutes = duration_mapping.get(target_duration, 6.5)
            
            logger.info(f"🎭 Generating customized script with prompt: {prompt_name}")
            logger.info(f"🎯 User preferences: {preferences}")
            logger.info(f"⏱️ Target duration: {target_minutes} minutes")
            
            # Check if we already have speech rate analysis
            if 'content_guide' in preferences:
                logger.info("✅ Using existing speech rate analysis for content planning")
                content_guide = preferences['content_guide']
            else:
                # Analyze speech rate and plan content if not already done
                logger.info("🔍 Speech rate analysis not found - analyzing now...")
                content_guide = self._analyze_speech_rate_and_plan_content(target_minutes)
                if content_guide:
                    preferences['content_guide'] = content_guide
            
            # Retrieve prompt and context
            prompt = self.prompt_manager.get_prompt(prompt_name)
            if not prompt:
                raise ValueError(f"Prompt '{prompt_name}' not found")
            
            logger.info("🔍 Retrieving relevant context...")
            context = self.rag_engine.get_context_for_prompt(prompt)
            
            # Generate content with iterative improvement
            final_script = self._generate_optimized_content(
                prompt, context, preferences, content_guide
            )
            
            logger.info("✅ Interactive conversation script generated successfully with preferences!")
            return final_script
            
        except Exception as e:
            logger.error(f"Failed to generate script with preferences: {e}")
            raise
    
    def generateAndSaveWithPreferences(self, prompt_name: str, preferences: dict) -> Optional[str]:
        """Generate and save a script with user preferences for customization."""
        try:
            # Generate script with preferences
            script = self.generate_script_with_preferences(prompt_name, preferences)
            if not script:
                return None
            
            # Save script with metadata
            filepath = self.save_script_with_metadata(script, prompt_name, preferences)
            return filepath
            
        except Exception as e:
            logger.error(f"Generate and save with preferences failed: {e}")
            return None
    
    def generate_audio_from_script(self, script_path: str, output_prefix: str = "podcast") -> List[str]:
        """Generate audio from a script file using TTS."""
        try:
            logger.info(f"🎙️ Generating audio from script: {script_path}")
            
            if not self.tts_integration:
                logger.error("TTS integration not available")
                return []
            
            # Generate audio using TTS
            audio_files = self.tts_integration.synthesize_script(
                script_path=script_path,
                output_prefix=output_prefix
            )
            
            if audio_files:
                logger.info(f"✅ Generated {len(audio_files)} audio files")
                return audio_files
            else:
                logger.warning("No audio files were generated")
                return []
                
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return []
    
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt templates."""
        return self.prompt_manager.list_prompt_names()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                "system": "Complete Lightweight Podcast Generation System",
                "components": {
                    "pdf_processor": "✅ Ready" if self.pdf_processor else "❌ Not initialized",
                    "rag_engine": "✅ Ready" if self.rag_engine else "❌ Not initialized",
                    "prompt_manager": "✅ Ready" if self.prompt_manager else "❌ Not initialized",
                    "llm_interface": "✅ Ready" if self.llm_interface else "❌ Not initialized",
                    "tts_integration": "✅ Ready" if self.tts_integration else "❌ Not initialized"
                },
                "dataset": {
                    "folder": str(self.dataset_folder),
                    "exists": self.dataset_folder.exists(),
                    "pdf_count": len(list(self.dataset_folder.glob("*.pdf"))) if self.dataset_folder.exists() else 0
                },
                "rag": {
                    "database_path": str(self.rag_db_path),
                    "document_count": self.rag_engine.get_collection_stats()["total_documents"] if self.rag_engine else 0
                },
                "output": {
                    "folder": str(self.output_folder),
                    "exists": self.output_folder.exists()
                }
            }
            
            # Add TTS-specific status
            if self.tts_integration:
                tts_info = self.tts_integration.get_model_info()
                status["tts"] = tts_info
            
            # Add LLM status
            if self.llm_interface:
                try:
                    llm_info = self.llm_interface.get_model_info()
                    status["llm"] = llm_info
                except:
                    status["llm"] = {"status": "Unknown"}
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    def clear_rag_index(self) -> bool:
        """Clear the RAG index."""
        try:
            if self.rag_engine:
                self.rag_engine.clear_documents()
                logger.info("✅ RAG index cleared successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to clear RAG index: {e}")
            return False
    
    def reprocess_dataset(self, pdf_chunk_size: int = 5000, text_chunk_size: int = 500, overlap: int = 100) -> bool:
        """Reprocess the dataset (clear and rebuild RAG index)."""
        try:
            logger.info("🔄 Reprocessing dataset...")
            
            # Clear existing index
            self.clear_rag_index()
            
            # Reprocess
            success = self.process_dataset(pdf_chunk_size, text_chunk_size, overlap)
            
            if success:
                logger.info("✅ Dataset reprocessed successfully")
            else:
                logger.error("❌ Dataset reprocessing failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Dataset reprocessing failed: {e}")
            return False
    
    def test_system(self) -> bool:
        """Test all system components."""
        try:
            logger.info("🧪 Testing system components...")
            
            # Test PDF processor
            if self.pdf_processor:
                pdf_files = list(self.dataset_folder.glob("*.pdf"))
                if pdf_files:
                    logger.info(f"✅ PDF processor: Found {len(pdf_files)} PDF files")
                else:
                    logger.warning("⚠️ PDF processor: No PDF files found in dataset folder")
            
            # Test RAG engine
            if self.rag_engine:
                stats = self.rag_engine.get_collection_stats()
                logger.info(f"✅ RAG engine: {stats}")
            
            # Test prompt manager
            if self.prompt_manager:
                prompts = self.prompt_manager.list_prompt_names()
                logger.info(f"✅ Prompt manager: {len(prompts)} prompts available")
            
            # Test LLM interface
            if self.llm_interface:
                try:
                    llm_info = self.llm_interface.get_model_info()
                    logger.info(f"✅ LLM interface: {llm_info}")
                except Exception as e:
                    logger.warning(f"⚠️ LLM interface: {e}")
            
            # Test TTS integration
            if self.tts_integration:
                try:
                    tts_info = self.tts_integration.get_model_info()
                    logger.info(f"✅ TTS integration: {tts_info}")
                    
                    # Note about XTTS-v2 requirements
                    if "note" in tts_info:
                        logger.info(f"💡 TTS Note: {tts_info['note']}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ TTS integration: {e}")
            
            logger.info("🎉 System test completed!")
            return True
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            return False
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _analyze_speech_rate_and_plan_content(self, target_duration_minutes: float) -> Dict[str, Any]:
        """
        Analyze speech rate and generate content planning guide.
        
        Args:
            target_duration_minutes: Target duration in minutes
            
        Returns:
            Content planning guide with speech rate analysis
        """
        if not self.speech_rate_analyzer:
            # Fallback to default planning
            logger.warning("Speech rate analyzer not available - using default planning")
            return self._generate_default_content_plan(target_duration_minutes)
        
        try:
            logger.info(f"🔍 Analyzing speech rate for {target_duration_minutes} minute target...")
            
            # Analyze all voice samples
            analysis_results = self.speech_rate_analyzer.analyze_all_samples()
            
            if not analysis_results:
                logger.warning("No voice samples analyzed - using default planning")
                return self._generate_default_content_plan(target_duration_minutes)
            
            # Generate content planning guide
            content_guide = self.speech_rate_analyzer.generate_content_planning_guide(target_duration_minutes)
            
            logger.info(f"✅ Content planning guide generated:")
            logger.info(f"   Target words: {content_guide['target_word_count']:,}")
            logger.info(f"   Speech rate: {content_guide['speech_rate_analysis']['average_wpm']:.1f} WPM")
            logger.info(f"   Content multiplier: {content_guide['content_multiplier']:.2f}")
            
            return content_guide
            
        except Exception as e:
            logger.error(f"Speech rate analysis failed: {e}")
            return self._generate_default_content_plan(target_duration_minutes)

    def _generate_optimized_content(self, prompt: str, context: str, preferences: dict, 
                                  content_guide: dict) -> str:
        """
        Generate content with intelligent iterative optimization to meet length requirements.
        
        Args:
            prompt: Base prompt
            context: Retrieved context
            preferences: User preferences
            content_guide: Content planning guide
            
        Returns:
            Optimized content meeting length requirements
        """
        target_words = content_guide.get('target_word_count', 500)
        llm_instructions = content_guide.get('llm_prompting_instructions', '')
        
        logger.info(f"🤖 Generating optimized content with target: {target_words:,} words")
        
        # Track all iterations and their results
        iterations = []
        best_result = None
        best_score = float('inf')  # Lower is better (closer to target)
        
        # Initial content generation
        logger.info("🔄 Generating initial content...")
        current_script = self.llm_interface.generate_conversation_with_preferences(
            prompt, context, preferences
        )
        
        # Validate initial content
        validation = self._validate_content_length(current_script, target_words)
        current_score = abs(validation['word_percentage'] - 100)  # Distance from 100%
        
        iterations.append({
            'iteration': 0,
            'script': current_script,
            'validation': validation,
            'score': current_score
        })
        
        logger.info(f"📊 Initial content validation: {validation['word_count']} words ({validation['word_percentage']:.1f}% of target)")
        
        # Check if initial content is acceptable
        if validation['is_acceptable']:
            logger.info("✅ Content meets length requirements on first attempt!")
            return current_script
        
        # Update best result
        if current_score < best_score:
            best_score = current_score
            best_result = current_script
            logger.info(f"🏆 New best result: {validation['word_count']} words (score: {current_score:.1f})")
        
        # Iterative improvement with guardrails
        iteration = 1
        while iteration <= self.max_content_iterations:
            logger.info(f"🔄 Iteration {iteration}/{self.max_content_iterations}: Improving content length...")
            
            # Generate improvement prompt with specific guidance
            improvement_prompt = self._create_smart_improvement_prompt(
                current_script, validation, target_words, llm_instructions, iterations
            )
            
            # Generate improved content
            improved_script = self.llm_interface.generate_text(improvement_prompt)
            
            # Validate improved content
            validation = self._validate_content_length(improved_script, target_words)
            current_score = abs(validation['word_percentage'] - 100)
            
            # Track this iteration
            iterations.append({
                'iteration': iteration,
                'script': improved_script,
                'validation': validation,
                'score': current_score
            })
            
            logger.info(f"📊 Iteration {iteration} validation: {validation['word_count']} words ({validation['word_percentage']:.1f}% of target, score: {current_score:.1f})")
            
            # Check if this iteration is the best so far
            if current_score < best_score:
                best_score = current_score
                best_result = improved_script
                logger.info(f"🏆 New best result: {validation['word_count']} words (score: {current_score:.1f})")
            
            # Check if we've reached acceptable length
            if validation['is_acceptable']:
                logger.info(f"✅ Content meets length requirements after {iteration} iterations!")
                break
            
            # Check if we're getting worse (overshooting)
            if validation['is_too_long'] and iteration > 1:
                logger.info(f"⚠️ Content is overshooting target. Stopping to avoid further degradation.")
                break
            
            # Update current script for next iteration
            current_script = improved_script
            iteration += 1
        
        # Final analysis and result selection
        logger.info(f"🎯 Final analysis of {len(iterations)} iterations:")
        for iter_data in iterations:
            status = "✅ ACCEPTABLE" if iter_data['validation']['is_acceptable'] else "⚠️ OUTSIDE RANGE"
            logger.info(f"   Iteration {iter_data['iteration']}: {iter_data['validation']['word_count']} words "
                       f"({iter_data['validation']['word_percentage']:.1f}% of target, score: {iter_data['score']:.1f}) {status}")
        
        # Return the best result (closest to target)
        if best_result:
            logger.info(f"🏆 Returning best result: {len(best_result.split())} words (score: {best_score:.1f})")
            return best_result
        else:
            logger.warning("⚠️ No valid results found, returning last iteration")
            return iterations[-1]['script']
    
    def _create_smart_improvement_prompt(self, current_script: str, validation: dict, 
                                       target_words: int, base_instructions: str, 
                                       previous_iterations: list) -> str:
        """Create an intelligent prompt for improving content length."""
        current_words = validation['word_count']
        target_percentage = validation['word_percentage']
        
        # Calculate what we need
        if target_percentage < 100:
            # Too short - need more words
            needed_words = target_words - current_words
            action = "EXPAND"
            guidance = f"ADD approximately {needed_words} words to reach the target"
        else:
            # Too long - need fewer words
            excess_words = current_words - target_words
            action = "REDUCE"
            guidance = f"REMOVE approximately {excess_words} words to reach the target"
        
        # Analyze previous iterations for learning
        iteration_analysis = ""
        if len(previous_iterations) > 1:
            prev_iter = previous_iterations[-2]  # Previous iteration
            prev_words = prev_iter['validation']['word_count']
            prev_percentage = prev_iter['validation']['word_percentage']
            
            if prev_percentage < target_percentage and target_percentage > 100:
                iteration_analysis = f"\nLEARNING: Previous iteration had {prev_words} words ({prev_percentage:.1f}%). "
                iteration_analysis += "You overshot the target. Be more conservative this time."
            elif prev_percentage < target_percentage and target_percentage < 100:
                iteration_analysis = f"\nLEARNING: Previous iteration had {prev_words} words ({prev_percentage:.1f}%). "
                iteration_analysis += "You're getting closer. Add content more precisely."
        
        improvement_prompt = f"""
INTELLIGENT CONTENT LENGTH ADJUSTMENT - ITERATION {len(previous_iterations)}

CURRENT STATUS:
- Current length: {current_words} words
- Target length: {target_words} words
- Current percentage: {target_percentage:.1f}% of target
- Action needed: {action}

{guidance}

{base_instructions}

CURRENT CONTENT:
{current_script}

{iteration_analysis}

IMPROVEMENT REQUIREMENTS:
1. {action} the content to reach {target_words} words (±10% acceptable: {int(target_words * 0.9)} to {int(target_words * 1.1)} words)
2. Be PRECISE - don't overshoot or undershoot significantly
3. MAINTAIN the conversational flow and natural dialogue
4. ENSURE male and female reporters have balanced speaking time
5. Focus on QUALITY over quantity - don't add filler

STRATEGY FOR {action.upper()}:
"""
        
        if action == "EXPAND":
            improvement_prompt += f"""
- Add approximately {needed_words} words
- Expand existing points with more details and examples
- Include additional case studies or statistics
- Add more interactive dialogue between reporters
- Include transitional phrases and connecting thoughts
- Be specific and meaningful - avoid repetition
"""
        else:  # REDUCE
            improvement_prompt += f"""
- Remove approximately {excess_words} words
- Consolidate similar points
- Remove redundant examples
- Streamline dialogue while keeping key insights
- Focus on essential information
- Maintain the core message and structure
"""
        
        improvement_prompt += f"""

TARGET: Generate content that is CLOSE to {target_words} words, not significantly over or under.
Be precise and thoughtful in your adjustments.

Generate the IMPROVED version now:
"""
        
        return improvement_prompt

    def _validate_content_length(self, content: str, target_words: int) -> Dict[str, Any]:
        """
        Validate if generated content meets length requirements.
        
        Args:
            content: Generated content
            target_words: Target word count
            
        Returns:
            Validation results
        """
        # Count words (simple whitespace-based approach)
        word_count = len(content.split())
        char_count = len(content)
        
        # Calculate percentage of target
        word_percentage = (word_count / target_words) * 100 if target_words > 0 else 0
        
        # Determine if content meets requirements
        min_acceptable = target_words * (1 - self.content_length_tolerance)
        max_acceptable = target_words * (1 + self.content_length_tolerance)
        
        is_acceptable = min_acceptable <= word_count <= max_acceptable
        is_too_short = word_count < min_acceptable
        is_too_long = word_count > max_acceptable
        
        validation_result = {
            'word_count': word_count,
            'char_count': char_count,
            'target_words': target_words,
            'word_percentage': word_percentage,
            'is_acceptable': is_acceptable,
            'is_too_short': is_too_short,
            'is_too_long': is_too_long,
            'min_acceptable': min_acceptable,
            'max_acceptable': max_acceptable,
            'feedback': self._generate_content_feedback(word_count, target_words, word_percentage)
        }
        
        return validation_result
    
    def _generate_content_feedback(self, actual_words: int, target_words: int, percentage: float) -> str:
        """Generate feedback about content length."""
        if percentage < 85:
            return f"Content is {100-percentage:.1f}% too short. Need approximately {target_words - actual_words} more words."
        elif percentage > 115:
            return f"Content is {percentage-100:.1f}% too long. Consider reducing by approximately {actual_words - target_words} words."
        else:
            return f"Content length is within acceptable range ({percentage:.1f}% of target)."


def main():
    """Main function for testing the system."""
    logger.info("🚀 Starting Complete Lightweight System...")
    
    try:
        # Initialize system
        system = CompleteLightweightSystem()
        
        # Test system
        system.test_system()
        
        logger.info("✅ System test completed successfully!")
        
    except Exception as e:
        logger.error(f"System test failed: {e}")


if __name__ == "__main__":
    main() 