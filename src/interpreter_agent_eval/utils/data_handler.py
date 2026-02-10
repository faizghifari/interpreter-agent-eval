"""Data handling utilities for the evaluation framework."""

import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path


class DataHandler:
    """Utility class for handling evaluation data."""
    
    @staticmethod
    def load_conversation_data(filepath: str) -> Dict[str, Any]:
        """Load conversation data from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Loaded conversation data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_conversation_data(data: Dict[str, Any], filepath: str) -> None:
        """Save conversation data to a JSON file.
        
        Args:
            data: Conversation data to save
            filepath: Path to save the file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_to_csv(
        conversation_log: List[Dict[str, Any]],
        filepath: str,
        fields: Optional[List[str]] = None
    ) -> None:
        """Export conversation log to CSV.
        
        Args:
            conversation_log: List of conversation turns
            filepath: Path to save CSV file
            fields: Optional list of fields to include (uses all if not specified)
        """
        if not conversation_log:
            return
        
        # Determine fields
        if fields is None:
            fields = list(conversation_log[0].keys())
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for turn in conversation_log:
                row = {k: turn.get(k, '') for k in fields}
                writer.writerow(row)
    
    @staticmethod
    def load_translation_brief(filepath: str) -> str:
        """Load translation brief from a file.
        
        Args:
            filepath: Path to the brief file
            
        Returns:
            Translation brief content
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_user_context(filepath: str) -> str:
        """Load user context from a file.
        
        Args:
            filepath: Path to the context file
            
        Returns:
            User context content
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def aggregate_results(result_files: List[str]) -> Dict[str, Any]:
        """Aggregate results from multiple evaluation runs.
        
        Args:
            result_files: List of paths to result JSON files
            
        Returns:
            Aggregated results
        """
        all_results = []
        for filepath in result_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_results.append(json.load(f))
        
        # Calculate aggregate metrics
        total_turns = sum(r.get('metrics', {}).get('total_turns', 0) for r in all_results)
        total_translation_time = sum(
            r.get('metrics', {}).get('average_translation_time', 0) *
            r.get('metrics', {}).get('total_turns', 0)
            for r in all_results
        )
        avg_translation_time = total_translation_time / total_turns if total_turns > 0 else 0
        
        return {
            "num_evaluations": len(all_results),
            "total_turns": total_turns,
            "average_translation_time": avg_translation_time,
            "evaluations": all_results
        }
