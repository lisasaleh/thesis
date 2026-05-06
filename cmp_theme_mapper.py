import json
import csv
from pathlib import Path
from datetime import datetime
import argparse
import re
import sys

from llm_utils import LocalLLM

class ThemeMapper:
    def __init__(self, themes_csv, cmp_csv, model_name="/scratch-shared/lsaleh/models/Qwen2.5-32B-Instruct", 
                 output_dir="mapping_results"):
        """
        Initialize the theme mapper
        
        Args:
            themes_csv: Path to themes.csv
            cmp_csv: Path to CMP codebook CSV
            model_name: Model path (local)
            output_dir: Directory to store results
        """
        self.themes_csv = Path(themes_csv)
        self.cmp_csv = Path(cmp_csv)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Metadata for tracking
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prompt_version = "v1_hierarchical_inclusive"
        
        # Load data
        self.themes = self._load_themes()
        self.cmp_codes = self._load_cmp_codes()
        
        # Initialize LLM
        print(f"Loading model: {model_name}")
        self.llm = LocalLLM(model_name)
        print("Model ready")
    
    def _load_themes(self):
        """Load and structure themes"""
        themes_by_main = {}
        all_themes = []
        
        with open(self.themes_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_themes.append(row)
                main = row['main_theme']
                if main not in themes_by_main:
                    themes_by_main[main] = []
                themes_by_main[main].append(row)
        
        print(f"Loaded {len(all_themes)} themes")
        print(f"Unique main themes: {len(themes_by_main)}")
        
        # Precompute subtheme strings for each main theme (for Pass 2 efficiency)
        subthemes_by_main = {}
        for main_theme, themes_list in themes_by_main.items():
            subtheme_lines = [
                f"{main_theme} | {theme['sub_theme']} (ID: {theme['id']})"
                for theme in themes_list
            ]
            subthemes_by_main[main_theme] = "\n".join(subtheme_lines)
        
        return {
            'all': all_themes,
            'by_main': themes_by_main,
            'main_list': sorted(themes_by_main.keys()),
            'subthemes_by_main': subthemes_by_main  # Precomputed
        }
    
    def _load_cmp_codes(self):
        """Load CMP codes"""
        codes = []
        with open(self.cmp_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                codes.append(row)
        
        print(f"Loaded {len(codes)} CMP codes")
        return codes
    
    def _call_model(self, prompt, temperature=0.0, max_tokens=500):
        """Call model using llm_utils"""
        return self.llm.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
    
    def _extract_json(self, response):
        """Extract JSON from model response"""
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If no JSON found, try parsing entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    
    def _has_subthemes(self, main_themes):
        """Check if any of the main themes have subthemes"""
        for main_theme in main_themes:
            if main_theme in self.themes['by_main'] and len(self.themes['by_main'][main_theme]) > 0:
                return True
        return False
    
    def _generate_main_theme_prompt(self, cmp_code):
        """Generate main theme classification prompt"""
        main_themes_list = ", ".join(self.themes['main_list'])
        
        prompt = f"""You are an expert policy analyst. Classify this CMP policy code to all plausibly relevant Dutch policy main themes.

INSTRUCTIONS:
- Select ALL themes that could reasonably apply
- You must select at least one main theme unless the CMP description is completely unrelated to public policy
- When in doubt, INCLUDE rather than exclude
- Do not exclude themes because of potential overlap
- This is a recall-oriented classification (aim for completeness)
- Only select from the provided theme list
- Return ONLY valid JSON, no markdown or prose

CMP Code: {cmp_code['code']}
CMP Title: {cmp_code.get('title', 'N/A')}
CMP Description: {cmp_code.get('description_md', '')}

Available main themes:
{main_themes_list}

Respond with ONLY this JSON structure (no markdown, no extra text):
{{
    "cmp_code": "{cmp_code['code']}",
    "selected_main_themes": ["Theme1", "Theme2"],
    "reasoning": "Brief explanation of selection"
}}"""
        return prompt
    
    def classify_main_themes(self, cmp_code):
        """Step 1: Classify to main themes"""
        prompt = self._generate_main_theme_prompt(cmp_code)
        response = self._call_model(prompt, temperature=0.0)
        result = self._extract_json(response)
        
        return {
            'cmp_code': cmp_code['code'],
            'raw_response': response,
            'parsed_response': result,
            'timestamp': datetime.now().isoformat(),
            'prompt_version': self.prompt_version,
            'model': self.model_name,
            'stage': 'main_themes'
        }
    
    def _generate_subtheme_prompts_batch(self, cmp_codes_with_themes):
        """Generate batch of subtheme prompts. Skip codes with no available subthemes."""
        prompts = []
        code_info = []
        no_subthemes_codes = []  # Codes with no subthemes available (no Pass 2 needed)
        
        for cmp_code, selected_main_themes in cmp_codes_with_themes:
            # Check if any selected main themes have subthemes
            if not self._has_subthemes(selected_main_themes):
                no_subthemes_codes.append(cmp_code['code'])
                continue
            
            # Build subtheme list from precomputed strings (no looping needed!)
            subtheme_lines = [
                self.themes['subthemes_by_main'][main_theme]
                for main_theme in selected_main_themes
                if main_theme in self.themes['subthemes_by_main']
            ]
            subthemes_str = "\n".join(subtheme_lines)
            
            prompt = f"""You are an expert policy analyst. Classify this CMP policy code to all relevant Dutch policy sub-themes WITHIN the selected main themes.

INSTRUCTIONS:
- Select ALL sub-themes that substantively apply
- When in doubt, INCLUDE rather than exclude
- Only select from sub-themes listed below
- This is a recall-oriented classification
- Return ONLY valid JSON, no markdown or prose

CMP Code: {cmp_code['code']}
CMP Title: {cmp_code.get('title', 'N/A')}
CMP Description: {cmp_code.get('description_md', '')}

Available sub-themes (in selected main themes):
{subthemes_str}

Respond with ONLY this JSON structure (no markdown, no extra text):
{{
    "cmp_code": "{cmp_code['code']}",
    "selected_subthemes": [
        {{"main_theme": "Theme1", "sub_theme": "SubTheme1", "id": "theme_XXXX"}},
        {{"main_theme": "Theme2", "sub_theme": "SubTheme2", "id": "theme_YYYY"}}
    ],
    "reasoning": "Brief explanation"
}}"""
            prompts.append(prompt)
            code_info.append(cmp_code['code'])
        
        return prompts, code_info, no_subthemes_codes
    
    def run_mapping(self, start_idx=0, end_idx=None, batch_size=16):
        """Run complete two-pass classification pipeline with batching"""
        results = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'prompt_version': self.prompt_version,
            'mappings': []
        }
        
        codes_to_process = self.cmp_codes[start_idx:end_idx]
        total = len(codes_to_process)
        
        print(f"\nProcessing {total} CMP codes (batch_size={batch_size})")
        print("=" * 60)
        
        # PASS 1: Batch main theme classification
        print("\n[PASS 1] Classifying main themes in batches...")
        main_results_dict = {}  # cmp_code -> main_result
        cmp_codes_with_main_themes = []  # For pass 2
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_codes = codes_to_process[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            
            print(f"  Batch {batch_num}: Processing codes {batch_start+1}-{batch_end}/{total}")
            
            # Generate prompts for batch
            prompts = [self._generate_main_theme_prompt(code) for code in batch_codes]
            
            # Batch inference
            responses = self.llm.batch_generate(prompts, max_new_tokens=500, temperature=0.0)
            
            # Parse responses
            for code, response in zip(batch_codes, responses):
                result = self._extract_json(response)
                main_result = {
                    'cmp_code': code['code'],
                    'raw_response': response,
                    'parsed_response': result,
                    'timestamp': datetime.now().isoformat(),
                    'prompt_version': self.prompt_version,
                    'model': self.model_name,
                    'stage': 'main_themes'
                }
                main_results_dict[code['code']] = main_result
                
                # Track codes with valid main themes for pass 2
                if result and result.get('selected_main_themes'):
                    cmp_codes_with_main_themes.append((code, result['selected_main_themes']))
        
        print(f"Pass 1 complete: {len(cmp_codes_with_main_themes)}/{total} codes with main themes")
        
        # PASS 2: Batch subtheme classification
        print("\n[PASS 2] Classifying sub-themes in batches...")
        
        for batch_start in range(0, len(cmp_codes_with_main_themes), batch_size):
            batch_end = min(batch_start + batch_size, len(cmp_codes_with_main_themes))
            batch_items = cmp_codes_with_main_themes[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            
            print(f"  Batch {batch_num}: Processing {batch_start+1}-{batch_end}/{len(cmp_codes_with_main_themes)}")
            
            # Generate prompts for batch
            prompts, code_ids, no_subthemes_codes = self._generate_subtheme_prompts_batch(batch_items)
            
            # Handle codes with no subthemes (skip LLM call)
            for code_id in no_subthemes_codes:
                main_result = main_results_dict[code_id]
                main_themes = main_result['parsed_response'].get('selected_main_themes', [])
                
                cmp_code_obj = next((c for c, _ in batch_items if c['code'] == code_id), {})
                cmp_title = cmp_code_obj.get('title', '')
                cmp_desc = cmp_code_obj.get('description_md', '')
                mapping = {
                    'cmp_code': code_id,
                    'cmp_title': cmp_title,
                    'cmp_description': cmp_desc,
                    'main_themes_stage': main_result,
                    'subthemes_stage': None,  # No subthemes to classify
                    'final_mapping': {
                        'main_themes': main_themes,
                        'subthemes': []  # Empty - no subthemes available for these main themes
                    }
                }
                results['mappings'].append(mapping)
                print(f"    {code_id}: No subthemes available (main themes: {', '.join(main_themes)})")
            
            # Batch inference only if we have prompts
            if not prompts:
                continue
            
            responses = self.llm.batch_generate(prompts, max_new_tokens=500, temperature=0.0)
            
            # Parse responses
            for code_id, response in zip(code_ids, responses):
                result = self._extract_json(response)
                sub_result = {
                    'cmp_code': code_id,
                    'raw_response': response,
                    'parsed_response': result,
                    'timestamp': datetime.now().isoformat(),
                    'prompt_version': self.prompt_version,
                    'model': self.model_name,
                    'stage': 'subthemes'
                }
                main_result = main_results_dict[code_id]
                main_themes = main_result['parsed_response'].get('selected_main_themes', [])
                
                # Combine results
                cmp_code_obj = next((c for c, _ in batch_items if c['code'] == code_id), {})
                mapping = {
                    'cmp_code': code_id,
                    'cmp_title': cmp_code_obj.get('title', ''),
                    'cmp_description': cmp_code_obj.get('description_md', ''),
                    'main_themes_stage': main_result,
                    'subthemes_stage': sub_result,
                    'final_mapping': {
                        'main_themes': main_themes,
                        'subthemes': result.get('selected_subthemes', []) if result else []
                    }
                }
                results['mappings'].append(mapping)
        
        print(f"Pass 2 complete: {len(results['mappings'])} codes with subthemes mapped")
        
        # Ensure ALL codes appear in output (explicit null mappings for unmatched codes)
        processed_codes = {m['cmp_code'] for m in results['mappings']}
        unmatched_count = 0
        
        for cmp_code in codes_to_process:
            if cmp_code['code'] not in processed_codes:
                # Code did not map to any main theme
                main_result = main_results_dict.get(cmp_code['code'])
                mapping = {
                    'cmp_code': cmp_code['code'],
                    'cmp_title': cmp_code.get('title', ''),
                    'cmp_description': cmp_code.get('description_md', ''),
                    'main_themes_stage': main_result,
                    'subthemes_stage': None,
                    'final_mapping': {
                        'main_themes': [],
                        'subthemes': []
                    }
                }
                results['mappings'].append(mapping)
                unmatched_count += 1
        
        print(f"Added {unmatched_count} unmatched codes (no main themes)")
        print(f"Total codes in output: {len(results['mappings'])} (expected: {total})")
        print("=" * 60)
        
        return results
    
    def _save_results(self, results, filename=None):
        """Save results to JSONL file"""
        if filename is None:
            filename = f"cmp_theme_mappings_{self.run_id}.jsonl"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write metadata
            f.write(json.dumps({
                'type': 'metadata',
                'run_id': results['run_id'],
                'timestamp': results['timestamp'],
                'model': results['model'],
                'prompt_version': results['prompt_version'],
                'total_mappings': len(results['mappings'])
            }) + '\n')
            
            # Write each mapping
            for mapping in results['mappings']:
                f.write(json.dumps(mapping) + '\n')
        
        print(f"\n Results saved to: {output_path}")
        return output_path
    
    def export_summary_csv(self, results, filename=None):
        """Export detailed CSV with one row per theme mapping"""
        if filename is None:
            filename = f"cmp_theme_summary_{self.run_id}.csv"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'main_theme', 'sub_theme', 'id', 'cmp_code', 'cmp_title', 'cmp_description'
            ])
            
            for mapping in results['mappings']:
                cmp_code = mapping['cmp_code']
                cmp_title = mapping['cmp_title']
                cmp_desc = mapping.get('cmp_description', '')
                
                subthemes = mapping['final_mapping']['subthemes']
                
                if subthemes:
                    # Write one row per subtheme
                    for st in subthemes:
                        writer.writerow([
                            st.get('main_theme', ''),
                            st.get('sub_theme', ''),
                            st.get('id', ''),
                            cmp_code,
                            cmp_title,
                            cmp_desc
                        ])
                else:
                    # No subthemes: write main themes if they exist
                    main_themes = mapping['final_mapping']['main_themes']
                    if main_themes:
                        for main_theme in main_themes:
                            writer.writerow([
                                main_theme,
                                '',  # empty sub_theme
                                '',  # empty id
                                cmp_code,
                                cmp_title,
                                cmp_desc
                            ])
                    else:
                        # Unmapped: one row with empty themes
                        writer.writerow([
                            '',
                            '',
                            '',
                            cmp_code,
                            cmp_title,
                            cmp_desc
                        ])
        
        print(f"Summary exported to: {output_path}")
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMP to Dutch Theme Hierarchical Mapper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--themes',
        type=str,
        default='themes.csv',
        help='Path to themes.csv'
    )
    
    parser.add_argument(
        '--cmp',
        type=str,
        default='CMP_codebook.csv',
        help='Path to CMP codebook CSV'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='/scratch-shared/lsaleh/models/Qwen2.5-32B-Instruct',
        help='Model path (local only)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='mapping_results',
        help='Output directory'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        help='End index'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for inference (default: 16)'
    )
    
    args = parser.parse_args()
    
    print(f"CMP Theme Mapper - Batch Processing")
    print(f"=" * 60)
    print(f"Themes: {args.themes}")
    print(f"CMP Codebook: {args.cmp}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    
    mapper = ThemeMapper(
        themes_csv=args.themes,
        cmp_csv=args.cmp,
        model_name=args.model,
        output_dir=args.output
    )
    
    results = mapper.run_mapping(start_idx=args.start, end_idx=args.end, batch_size=args.batch_size)
    
    # Save detailed results
    mapper._save_results(results)
    
    # Export summary CSV
    mapper.export_summary_csv(results)
    
    print(f"\nMapping complete!")
