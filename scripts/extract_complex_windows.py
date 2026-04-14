import argparse
import csv
import json
import re
import heapq
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Language-pair-specific heuristic dictionaries (currently: Indonesian → Korean)
# To adapt for a different language pair, replace the marker sets below with
# equivalents for the source language (discourse/nuance markers) and target
# language (social address forms, expressive verb endings, etc.).
# ---------------------------------------------------------------------------
ID_DISCOURSE_MARKERS = {"dong", "deh", "sih", "kok", "nih", "lah", "kan", "ya", "lho", "kek"}
ID_NUANCE_MARKERS = {"tolong", "coba", "biarin", "terserah", "gimana", "masa", "mending"}
KO_SOCIAL_MARKERS = {"요", "습니다", "님", "씨", "선배", "형", "누나", "언니", "오빠", "아저씨", "아줌마"}
KO_EXPRESSIVE_ENDINGS = {"잖아", "거든", "네요", "데요", "죠", "지요"}

def is_sfx_only(text: str) -> bool:
    return False

def is_single_speaker(text: str) -> bool:
    turns = len(re.findall(r"(?:(?<=\s)|^)-", text.strip()))
    return turns <= 1

def passes_hard_alignment_filter(src: str, tgt: str, metadata: str) -> bool:
    # Temporarily disable metadata exact-match requirement since it might be overly restrictive
    # if metadata and metadata != "exact_1to1_text_match":
    #     return False
        
    src = src.strip()
    tgt = tgt.strip()
    
    if not src or not tgt:
        return False
    if is_sfx_only(src) or is_sfx_only(tgt):
        return False
    if not is_single_speaker(src) or not is_single_speaker(tgt):
        return False
        
    src_tokens = len(re.findall(r"\w+", src))
    tgt_tokens = len(re.findall(r"\w+", tgt))
    
    if src_tokens < 2 or tgt_tokens < 2:
        return False
        
    # Drop extreme length mismatches to avoid completely misaligned garbage
    length_ratio = max(src_tokens, tgt_tokens) / max(1, min(src_tokens, tgt_tokens))
    if length_ratio > 3.0:
        return False
        
    return True

def calculate_translation_complexity(src: str, tgt: str) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []
    
    src_lower = src.lower()
    
    # Indonesian Markers (exact substring match to handle attached clitics safely without proper tokenization)
    id_disc = [m for m in ID_DISCOURSE_MARKERS if f" {m}" in f" {src_lower}"]
    if id_disc:
        score += 1.0 * len(id_disc)
        reasons.append(f"id_discourse({','.join(id_disc)})")
        
    id_nuance = [m for m in ID_NUANCE_MARKERS if f" {m}" in f" {src_lower}"]
    if id_nuance:
        score += 1.0 * len(id_nuance)
        reasons.append(f"id_nuance({','.join(id_nuance)})")
        
    # Korean Markers
    ko_social = [m for m in KO_SOCIAL_MARKERS if m in tgt]
    if ko_social:
        score += 1.0 * len(ko_social)
        reasons.append(f"ko_social({','.join(ko_social)})")
        
    ko_endings = [m for m in KO_EXPRESSIVE_ENDINGS if m in tgt]
    if ko_endings:
        score += 1.0 * len(ko_endings)
        reasons.append(f"ko_ending({','.join(ko_endings)})")
        
    # Structural Mismatches (implies adaptation)
    src_q = "?" in src or "؟" in src
    tgt_q = "?" in tgt or "？" in tgt
    if src_q != tgt_q:
        score += 0.5
        reasons.append("question_mismatch")
        
    src_exc = "!" in src
    tgt_exc = "!" in tgt
    if src_exc != tgt_exc:
        score += 0.5
        reasons.append("exclamation_mismatch")
        
    # Moderate length asymmetry (requires adaptation, not 1:1 literal)
    src_tokens = len(re.findall(r"\w+", src))
    tgt_tokens = len(re.findall(r"\w+", tgt))
    ratio = max(src_tokens, tgt_tokens) / max(1, min(src_tokens, tgt_tokens))
    if 1.5 < ratio <= 3.0:
        score += 0.5
        reasons.append("moderate_length_asymmetry")
        
    return score, reasons

def process_files(input_files: List[Path], n_prev: int, n_after: int, top_n: int) -> List[Dict[str, Any]]:
    # Min-heap to keep track of top N elements. 
    # Heap stores tuples of (score, tiebreaker_id, record_dict)
    top_windows = []
    global_id = 0
    total_qualified = 0
    
    for file_path in input_files:
        print(f"Processing {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        print(f"  Parsed {len(rows)} rows from TSV.")
            
        for i in range(len(rows)):
            target = rows[i]
            src_text = target.get('source_text', '')
            tgt_text = target.get('target_text', '')
            metadata = target.get('metadata_match_type', '')
            
            if src_text is None: src_text = ""
            if tgt_text is None: tgt_text = ""
            if metadata is None: metadata = ""
            
            src_text = src_text.strip()
            tgt_text = tgt_text.strip()
            metadata = metadata.strip()
            
            if not passes_hard_alignment_filter(src_text, tgt_text, metadata):
                continue
                
            complexity_score, reasons = calculate_translation_complexity(src_text, tgt_text)
            
            # Only consider segments with SOME complexity (score > 0)
            if complexity_score <= 0.0:
                continue
                
            total_qualified += 1
                
            start_idx = max(0, i - n_prev)
            end_idx = min(len(rows), i + n_after + 1)
            
            prev_context = rows[start_idx:i]
            after_context = rows[i+1:end_idx]
            
            record = {
                "segment_id": target.get("segment_id", str(global_id)),
                "film_key": target.get("film_key", "unknown"),
                "file_source": file_path.name,
                "complexity_score": complexity_score,
                "complexity_reasons": reasons,
                "target_source": src_text,
                "target_translation": tgt_text,
                "prev_context_source": [r.get('source_text', '').strip() for r in prev_context if r.get('source_text')],
                "prev_context_translation": [r.get('target_text', '').strip() for r in prev_context if r.get('target_text')],
                "after_context_source": [r.get('source_text', '').strip() for r in after_context if r.get('source_text')],
                "after_context_translation": [r.get('target_text', '').strip() for r in after_context if r.get('target_text')],
            }
            
            if len(top_windows) < top_n:
                heapq.heappush(top_windows, (complexity_score, global_id, record))
            else:
                # If score is greater than the smallest score in the heap, push & pop
                if complexity_score > top_windows[0][0]:
                    heapq.heappushpop(top_windows, (complexity_score, global_id, record))
            
            global_id += 1

    # Extract records and sort them descending by score
    results = [item[2] for item in top_windows]
    results.sort(key=lambda x: x["complexity_score"], reverse=True)
    
    print(f"Total qualified windows (score > 0): {total_qualified}")
    
    return results
def main():
    parser = argparse.ArgumentParser(description="Extract top complex translation windows")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing TSV files")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--n-prev", type=int, default=15, help="Previous context size")
    parser.add_argument("--n-after", type=int, default=3, help="After context size")
    parser.add_argument("--top-n", type=int, default=1000, help="Number of top candidates to keep")
    parser.add_argument("--file-limit", type=int, default=None, help="Limit number of TSV files to process (for testing)")
    
    args = parser.parse_args()
    
    input_files = sorted(list(args.input_dir.glob("*.tsv")))
    if args.file_limit:
        input_files = input_files[:args.file_limit]
        
    print(f"Found {len(input_files)} TSV files. Extracting top {args.top_n} complex windows...")
    
    results = process_files(input_files, args.n_prev, args.n_after, args.top_n)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"Successfully saved {len(results)} windows to {args.output}")

if __name__ == "__main__":
    main()
