"""
Prompt Manager Module

This module handles loading and managing prompt templates for the podcast
script generation system.
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


class PromptManager:
    """Manages prompt templates and their loading."""
    
    def __init__(self, prompts_folder: str = "prompts"):
        """Initialize the prompt manager.
        
        Args:
            prompts_folder: Path to the folder containing prompt files
        """
        self.prompts_folder = Path(prompts_folder)
        self.prompts = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompt files from the prompts folder."""
        if not self.prompts_folder.exists():
            logger.warning(f"Prompts folder {self.prompts_folder} does not exist")
            return
        
        # Load all .txt files in the prompts folder
        prompt_files = list(self.prompts_folder.glob("*.txt"))
        logger.info(f"Found {len(prompt_files)} prompt files")
        
        for prompt_file in prompt_files:
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    prompt_name = prompt_file.stem  # Remove .txt extension
                    self.prompts[prompt_name] = content
                    logger.info(f"Loaded prompt: {prompt_name}")
            except Exception as e:
                logger.error(f"Failed to load prompt {prompt_file}: {e}")
    
    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """Get a specific prompt by name.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            
        Returns:
            Prompt content or None if not found
        """
        return self.prompts.get(prompt_name)
    
    def get_all_prompts(self) -> Dict[str, str]:
        """Get all loaded prompts.
        
        Returns:
            Dictionary of all prompts
        """
        return self.prompts.copy()
    
    def list_prompt_names(self) -> List[str]:
        """Get list of available prompt names.
        
        Returns:
            List of prompt names
        """
        return list(self.prompts.keys())
    
    def reload_prompts(self):
        """Reload all prompts from the prompts folder."""
        self.prompts.clear()
        self._load_prompts()
        logger.info("Reloaded all prompts")
    
    def add_prompt(self, name: str, content: str):
        """Add a new prompt programmatically.
        
        Args:
            name: Name of the prompt
            content: Content of the prompt
        """
        self.prompts[name] = content
        logger.info(f"Added new prompt: {name}")
    
    def remove_prompt(self, name: str) -> bool:
        """Remove a prompt.
        
        Args:
            name: Name of the prompt to remove
            
        Returns:
            True if prompt was removed, False if not found
        """
        if name in self.prompts:
            del self.prompts[name]
            logger.info(f"Removed prompt: {name}")
            return True
        return False
    
    def get_prompt_with_context(self, prompt_name: str, context: str) -> str:
        """Get a prompt with injected context.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            context: Context to inject into the prompt
            
        Returns:
            Prompt with injected context
        """
        prompt = self.get_prompt(prompt_name)
        if not prompt:
            logger.error(f"Prompt '{prompt_name}' not found")
            return ""
        
        # Simple context injection - replace {context} placeholder
        if "{context}" in prompt:
            return prompt.replace("{context}", context)
        else:
            # If no placeholder, append context at the end
            return f"{prompt}\n\nContext Information:\n{context}"
    
    def validate_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt exists and is valid.
        
        Args:
            prompt_name: Name of the prompt to validate
            
        Returns:
            True if prompt is valid, False otherwise
        """
        prompt = self.get_prompt(prompt_name)
        if not prompt:
            return False
        
        # Basic validation - check if prompt has minimum content
        return len(prompt.strip()) > 10 