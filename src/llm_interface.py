"""
LLM Interface Module

This module handles communication with the local Ollama instance
running Qwen 2.5 14B for text generation.
"""

import os
import json
from typing import Dict, Any, Optional, List
import ollama
from loguru import logger


class LLMInterface:
    """Interface for communicating with Ollama LLM."""
    
    def __init__(self, model_name: str = "qwen2.5:14b-instruct", base_url: str = "http://localhost:11434"):
        """Initialize the LLM interface.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: URL of the Ollama server
        """
        self.model_name = model_name
        self.base_url = base_url
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Ollama client."""
        try:
            # Set the base URL for Ollama
            if hasattr(ollama, 'set_host'):
                ollama.set_host(self.base_url)
            self.client = ollama
            
            # Test connection by listing models
            try:
                models = self.client.list()
                logger.info(f"Connected to Ollama at {self.base_url}")
                
                if 'models' in models and models['models']:
                    # Handle both dict and object models
                    model_names = []
                    for model in models['models']:
                        if hasattr(model, 'model'):
                            # Ollama object model
                            model_names.append(model.model)
                        elif isinstance(model, dict) and 'name' in model:
                            # Dictionary model
                            model_names.append(model['name'])
                        elif isinstance(model, dict) and 'model' in model:
                            # Dictionary with 'model' key
                            model_names.append(model['model'])
                        else:
                            # Fallback
                            model_names.append(str(model))
                    
                    logger.info(f"Available models: {model_names}")
                    
                    # Check if our target model is available
                    if self.model_name not in model_names:
                        logger.warning(f"Model {self.model_name} not found. Available: {model_names}")
                        # Look for similar models (e.g., qwen2.5:14b-instruct)
                        similar_models = [m for m in model_names if 'qwen' in m.lower()]
                        if similar_models:
                            self.model_name = similar_models[0]
                            logger.info(f"Using similar model: {self.model_name}")
                        elif model_names:
                            self.model_name = model_names[0]  # Use first available model
                            logger.info(f"Using available model: {self.model_name}")
                else:
                    logger.warning("No models found in Ollama")
                    
            except Exception as e:
                logger.warning(f"Could not list models (Ollama may not be running): {e}")
                # Continue with default model name
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    def generate_text(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate text using the LLM.
        
        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
            
        Returns:
            Generated text
        """
        try:
            logger.info(f"Generating text with model {self.model_name}")
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            # Generate response using Ollama
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            
            generated_text = response['response']
            logger.info(f"Generated {len(generated_text)} characters")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return f"Error generating text: {str(e)}"
    
    def generate_conversation(self, prompt: str, context: str = "", 
                            conversation_length: str = "medium") -> str:
        """Generate a conversation script between two reporters.
        
        Args:
            prompt: Base prompt for the conversation
            context: Relevant context information
            conversation_length: Desired length ("short", "medium", "long")
            
        Returns:
            Generated conversation script
        """
        try:
            # Build the full prompt
            full_prompt = self._build_conversation_prompt(prompt, context, conversation_length)
            
            # Generate the conversation
            response = self.generate_text(
                prompt=full_prompt,
                max_tokens=5000 if conversation_length == "long" else 3000 if conversation_length == "medium" else 2000,
                temperature=0.8
            )
            
            # Format the response to ensure proper speaker labels
            formatted_response = self._format_conversation_response(response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to generate conversation: {e}")
            return f"Error generating conversation: {str(e)}"
    
    def generate_conversation_with_preferences(self, prompt: str, context: str = "", preferences: dict = None) -> str:
        """
        Generate interactive conversation script with user preferences and speech rate optimization.
        
        Args:
            prompt: Base prompt text
            context: Retrieved context information
            preferences: User preferences for content generation
            
        Returns:
            Generated conversation script
        """
        try:
            # Build enhanced prompt with preferences and speech rate optimization
            enhanced_prompt = self._build_conversation_prompt_with_preferences(prompt, context, preferences)
            
            # Generate text using the enhanced prompt
            response = self.generate_text(enhanced_prompt)
            
            # Format the response based on voice type
            formatted_response = self._format_conversation_response_with_preferences(response, preferences)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to generate conversation with preferences: {e}")
            raise
    
    def _build_conversation_prompt(self, prompt: str, context: str, conversation_length: str) -> str:
        """Build the full prompt for conversation generation.
        
        Args:
            prompt: Base prompt
            context: Context information
            conversation_length: Desired length
            
        Returns:
            Complete prompt string
        """
        # Add context if provided
        if context:
            prompt_with_context = f"{prompt}\n\nContext Information:\n{context}"
        else:
            prompt_with_context = prompt
        
        # Enhanced conversation format instructions for cybersecurity/audit topics
        format_instructions = f"""
        
        Create a natural, engaging conversation between two speakers discussing this topic.
        
        CONVERSATION REQUIREMENTS:
        - Person 1: Use "Person 1:" as the speaker label
        - Person 2: Use "Person 2:" as the speaker label
        - ALTERNATE speakers naturally - don't let one person dominate
        - Each speaker should have 3-5 sentences per turn
        - Create natural dialogue flow with questions, responses, and follow-ups
        - Make it sound like a real podcast conversation between colleagues
        
        CONVERSATION STRUCTURE:
        - Start with Person 1 introducing the topic
        - Person 2 responds with insights or questions
        - Continue alternating with natural back-and-forth
        - Include moments where they build on each other's points
        - End with both speakers summarizing key takeaways
        
        LENGTH: Make this conversation {conversation_length} in length.
        
        STYLE: Professional but conversational, like two cybersecurity experts discussing findings.
        
        FORMAT: Use clear speaker labels and natural dialogue flow.
        """
        
        return prompt_with_context + format_instructions
    
    def _format_conversation_response(self, response: str) -> str:
        """Format the LLM response to ensure proper conversation structure.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Formatted conversation
        """
        # Ensure the response starts with a speaker label
        if not response.strip().startswith(("Male Reporter:", "Female Reporter:")):
            response = "Male Reporter: " + response
        
        # Clean up any formatting issues
        lines = response.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Ensure speaker labels are properly formatted
                if line.startswith("Male Reporter") or line.startswith("Female Reporter"):
                    formatted_lines.append(line)
                else:
                    # If line doesn't start with speaker label, it's probably continuation
                    if formatted_lines:
                        formatted_lines[-1] += " " + line
                    else:
                        formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _build_conversation_prompt_with_preferences(self, prompt: str, context: str, preferences: dict) -> str:
        """
        Build enhanced prompt with user preferences and speech rate optimization.
        
        Args:
            prompt: Base prompt text
            context: Retrieved context information
            preferences: User preferences
            
        Returns:
            Enhanced prompt for LLM
        """
        # Extract preferences
        speed_factor = preferences.get('speed_factor', 1.0)
        energy_level = preferences.get('energy_level', 'medium')
        conversation_length = preferences.get('conversation_length', 'medium')
        voice_type = preferences.get('voice_type', 'both')
        starting_speaker = preferences.get('starting_speaker', 'Person 1')
        speaker_assignments = preferences.get('speaker_assignments', {})
        
        # Get speech rate optimization if available
        speech_rate_instructions = ""
        content_planning = ""
        
        if 'content_guide' in preferences:
            content_guide = preferences['content_guide']
            speech_rate_instructions = content_guide.get('llm_prompting_instructions', '')
            
                    # Add specific content planning instructions
        target_words = content_guide.get('target_word_count', 0)
        target_duration = content_guide.get('target_duration_minutes', 0)
        speech_rate = content_guide.get('speech_rate_analysis', {}).get('average_wpm', 0)
        
        if target_words and target_duration and speech_rate:
            content_planning = f"""
CRITICAL CONTENT PLANNING REQUIREMENTS:
- TARGET WORD COUNT: {target_words:,} words (±10% acceptable: {int(target_words * 0.9):,} to {int(target_words * 1.1):,} words)
- TARGET DURATION: {target_duration:.1f} minutes when spoken
- MEASURED SPEECH RATE: {speech_rate:.1f} words per minute
- CONTENT MULTIPLIER: {content_guide.get('content_multiplier', 1.0):.2f}x

SECTION BREAKDOWN TARGETS (STRICT REQUIREMENTS):
"""
        
        # Add dynamic speaker mapping instructions
        speaker_mapping = ""
        if speaker_assignments and voice_type == "both":
            speaker_mapping = f"""
SPEAKER MAPPING REQUIREMENTS:
- {starting_speaker} will start the conversation
- Use alternating dialogue between Person 1 and Person 2
- Person 1 represents: {starting_speaker}
- Person 2 represents: {[s for s in speaker_assignments.keys() if s != starting_speaker][0] if len(speaker_assignments) > 1 else 'Person 2'}
- Ensure natural conversation flow with the starting speaker introducing the topic
"""
        
        # Add section breakdown with strict word counts
        for section_name, section_info in content_guide.get('section_breakdown', {}).items():
            content_planning += f"- {section_name.replace('_', ' ').title()}: {section_info['word_count']} words ({section_info['duration_minutes']:.1f} min)\n"
        
        content_planning += f"""
CRITICAL: You MUST generate content that is CLOSE to {target_words:,} words total.
- Too short: Content will be too brief for the target duration
- Too long: Content will exceed the time limit
- Target range: {int(target_words * 0.9):,} to {int(target_words * 1.1):,} words
- Be precise and thoughtful in your content generation
"""
        
        # Build the enhanced prompt
        enhanced_prompt = f"""
{prompt}

CONTEXT INFORMATION:
{context}

USER PREFERENCES:
- Speed: {self._get_speed_description(speed_factor)}
- Energy: {self._get_energy_description(energy_level)}
- Duration: {self._get_duration_description(conversation_length)}
- Voice: {self._get_voice_description(voice_type)}

{content_planning}

{speech_rate_instructions}

CRITICAL FORMAT REQUIREMENTS:
1. Generate a NATURAL, ENGAGING conversation between speakers
2. Use alternating speaker labels: "Person 1:" and "Person 2:"
3. Each speaker should have meaningful contributions
4. Maintain conversational flow and natural dialogue
5. Include specific details, examples, and insights
6. Ensure balanced speaking time between Person 1 and Person 2
7. MEET THE SPECIFIED WORD COUNT TARGETS for each section
8. COUNT YOUR WORDS CAREFULLY - this is critical for timing

EXAMPLE FOR 'BOTH' VOICE:
Person 1: Welcome to today's discussion. Let me start by setting the context...
Person 2: That's a great point. I'd like to add that...
Person 1: Excellent observation. Another aspect to consider is...
Person 2: Absolutely. And we shouldn't forget about...

GENERATION INSTRUCTIONS:
- Create content that feels like a natural conversation
- Include transitions between topics
- Add specific examples and case studies
- Ensure each speaker contributes meaningfully
- Maintain professional but engaging tone
- Focus on providing valuable insights and analysis
- STRICTLY ADHERE to word count targets for proper timing
- Be mindful of the total word count - don't overshoot or undershoot significantly

WORD COUNT VALIDATION:
Before submitting your response, estimate the total word count.
- Target: {content_guide.get('target_word_count', 'unknown') if 'content_guide' in preferences else 'unknown'} words
- Acceptable range: ±10% of target
- If you're significantly off target, adjust your content accordingly

Generate the conversation script now, being mindful of the word count requirements:
"""
        
        return enhanced_prompt
    
    def _format_conversation_response_with_preferences(self, response: str, preferences: dict) -> str:
        """Format the LLM response to ensure proper conversation structure with user preferences.
        
        Args:
            response: Raw LLM response
            preferences: User preferences for formatting
            
        Returns:
            Formatted conversation
        """
        voice_type = preferences.get('voice_type', 'both')
        
        if voice_type == 'both':
            # Force alternating dialogue for 'both' voice type
            return self._force_alternating_dialogue(response, preferences)
        else:
            # Single voice formatting
            return self._format_single_voice(response, voice_type)
    
    def _force_alternating_dialogue(self, response: str, preferences: dict = None) -> str:
        """Force the response to alternate between Person 1 and Person 2."""
        # Split into sentences
        sentences = response.replace('\n', ' ').split('. ')
        if not sentences:
            return response
        
        # Determine starting speaker from preferences
        starting_speaker = "Person 1"
        if preferences and 'starting_speaker' in preferences:
            # Map the user's starting speaker choice to Person 1
            starting_speaker = "Person 1"
        
        # Ensure we start with the specified starting speaker
        formatted_lines = []
        current_speaker = starting_speaker
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Clean up any existing speaker labels in the sentence
            sentence = sentence.replace('Person 1:', '').replace('Person 2:', '').replace('Male Reporter:', '').replace('Female Reporter:', '').strip()
            
            # Add speaker label
            formatted_line = f"{current_speaker}: {sentence}"
            if not sentence.endswith('.'):
                formatted_line += '.'
            formatted_lines.append(formatted_line)
            
            # Alternate speaker
            current_speaker = "Person 2" if current_speaker == "Person 1" else "Person 1"
        
        return '\n\n'.join(formatted_lines)
    
    def _format_single_voice(self, response: str, voice_type: str) -> str:
        """Format response for single voice type."""
        speaker = "Person 1"  # Single voice always uses Person 1
        
        # Ensure the response starts with the speaker label
        if not response.strip().startswith(speaker):
            response = f"{speaker}: {response}"
        
        # Clean up any formatting issues
        lines = response.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Ensure speaker labels are properly formatted
                if line.startswith(speaker):
                    formatted_lines.append(line)
                else:
                    # If line doesn't start with speaker label, it's probably continuation
                    if formatted_lines:
                        formatted_lines[-1] += " " + line
                    else:
                        formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _get_speed_description(self, speed: float) -> str:
        """Get description for speed preference."""
        if speed == 0.5:
            return "very slow, detailed analysis"
        elif speed == 1.0:
            return "normal pace"
        elif speed == 1.5:
            return "fast, quick insights"
        elif speed == 2.0:
            return "very fast, high-energy"
        return "normal pace"
    
    def _get_energy_description(self, energy: str) -> str:
        """Get description for energy preference."""
        descriptions = {
            "low": "calm, serious, and analytical tone",
            "medium": "balanced, professional, and engaging",
            "high": "exciting, dynamic, and passionate"
        }
        return descriptions.get(energy, "balanced energy")
    
    def _get_duration_description(self, duration: str) -> str:
        """Get description for duration preference."""
        descriptions = {
            "short": "2-3 minutes with key points only",
            "medium": "5-7 minutes with balanced coverage",
            "long": "10-15 minutes with comprehensive discussion"
        }
        return descriptions.get(duration, "balanced coverage")
    
    def _get_voice_description(self, voice: str) -> str:
        """Get description for voice preference."""
        descriptions = {
            "male": "single male reporter presenting the information",
            "female": "single female reporter presenting the information",
            "both": "interactive dialogue between male and female reporters"
        }
        return descriptions.get(voice, "interactive dialogue")
    
    def test_connection(self) -> bool:
        """Test the connection to Ollama.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            models = self.client.list()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        try:
            models = self.client.list()
            for model in models['models']:
                # Get model name from either object or dict
                model_name = None
                if hasattr(model, 'model'):
                    model_name = model.model
                elif isinstance(model, dict):
                    model_name = model.get('name') or model.get('model')
                else:
                    model_name = str(model)
                
                if model_name == self.model_name:
                    # Extract info from either object or dict
                    if hasattr(model, 'size'):
                        return {
                            'name': model_name,
                            'size': getattr(model, 'size', 'Unknown'),
                            'modified_at': getattr(model, 'modified_at', 'Unknown'),
                            'digest': getattr(model, 'digest', 'Unknown')
                        }
                    elif isinstance(model, dict):
                        return {
                            'name': model_name,
                            'size': model.get('size', 'Unknown'),
                            'modified_at': model.get('modified_at', 'Unknown'),
                            'digest': model.get('digest', 'Unknown')
                        }
                    else:
                        return {
                            'name': model_name,
                            'info': str(model)
                        }
            return {'error': 'Model not found'}
        except Exception as e:
            return {'error': str(e)} 

    def clean_script_for_audio(self, script: str, save_to_file: bool = True) -> tuple[str, str]:
        """
        Clean script for audio generation while preserving speaker labels.
        
        Args:
            script: Original script with speaker labels and formatting
            save_to_file: Whether to save cleaned script to file
            
        Returns:
            Tuple of (cleaned_script, filename) where filename is empty string if not saved
        """
        try:
            # Remove time annotations and other bracketed content
            import re
            
            # Remove time annotations like [Time: 0.8 minutes], [Time: 30s], etc.
            script = re.sub(r'\[Time:\s*\d+(?:\.\d+)?\s*(?:minutes?|mins?|s|seconds?)\]', '', script, flags=re.IGNORECASE)
            
            # Remove section headers like **Introduction:**, **Background:**, etc.
            script = re.sub(r'\*\*([^*]+):\*\*', '', script)
            
            # Remove any other bracketed content
            script = re.sub(r'\[[^\]]*\]', '', script)
            
            # Remove markdown formatting but preserve speaker labels
            script = re.sub(r'\*\*(.*?)\*\*', r'\1', script)  # Remove bold formatting
            script = re.sub(r'\*(.*?)\*', r'\1', script)      # Remove italic formatting
            
            # Remove section dividers like "---"
            script = re.sub(r'-{3,}', '', script)
            
            # Remove extra whitespace and normalize
            script = re.sub(r'\n\s*\n', '\n\n', script)  # Remove excessive blank lines
            script = re.sub(r' +', ' ', script)          # Normalize spaces
            
            # Clean up the end of the script
            script = re.sub(r'Total Duration:\s*.*?minutes?.*', '', script, flags=re.IGNORECASE)
            script = re.sub(r'By expanding.*', '', script)  # Remove meta-commentary
            script = re.sub(r'This expanded version.*', '', script)  # Remove meta-commentary
            
            # Ensure Person 1/2 labels are properly formatted for audio generation
            # Convert "**Person 1:**" to "Person 1:" and "**Person 2:**" to "Person 2:"
            script = re.sub(r'\*\*Person\s*1\*\*:\s*', 'Person 1: ', script)
            script = re.sub(r'\*\*Person\s*2\*\*:\s*', 'Person 2: ', script)
            
            # Also handle any remaining Person 1/2 labels without markdown
            script = re.sub(r'Person\s*1:\s*', 'Person 1: ', script)
            script = re.sub(r'Person\s*2:\s*', 'Person 2: ', script)
            
            # Remove any trailing whitespace
            script = script.strip()
            
            # Save cleaned script to file if requested
            filename = ""
            if save_to_file:
                try:
                    import os
                    from datetime import datetime
                    
                    # Create cleanup-scripts directory if it doesn't exist
                    cleanup_dir = "output/cleanup-scripts"
                    os.makedirs(cleanup_dir, exist_ok=True)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"script_cleaned_{timestamp}.txt"
                    filepath = os.path.join(cleanup_dir, filename)
                    
                    # Save cleaned script to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(script)
                    
                    logger.info(f"✅ Cleaned script saved to: {filepath}")
                    
                except Exception as e:
                    logger.error(f"Failed to save cleaned script to file: {e}")
                    filename = ""
            
            return script, filename
            
        except Exception as e:
            logger.error(f"Failed to clean script for audio: {e}")
            return script, "" 