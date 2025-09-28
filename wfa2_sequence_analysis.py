#!/usr/bin/env python3
"""
WFA2 Sequence Analysis Tool
Consolidated script for pairwise and categorical sequence alignment analysis
"""

import glob
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import typer


def extract_sequences(fasta_file):
    """Extract sequences from FASTA file"""
    sequences = {}
    current_name = None
    current_seq = ""
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = current_seq
                current_name = line[1:]
                current_seq = ""
            elif line and not line.startswith('#'):
                current_seq += line
    
    if current_name:
        sequences[current_name] = current_seq
    
    return sequences

def categorize_sequences(sequences):
    """Categorize sequences by type"""
    categories = defaultdict(list)
    
    for name, seq in sequences.items():
        if "left_flank" in name:
            categories["left_flank"].append((name, seq))
        elif "right_flank" in name:
            categories["right_flank"].append((name, seq))
        elif "repeat" in name:
            categories["repeat"].append((name, seq))
        else:
            categories["other"].append((name, seq))
    
    return categories

def sanitize_filename(name):
    """Convert sequence name to a valid filename"""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = sanitized.replace(' ', '_')
    return sanitized

def create_pairwise_files(sequences, output_dir):
    """Create files for each pairwise comparison"""
    names = list(sequences.keys())
    pair_count = 0

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            seq1, seq2 = sequences[name1], sequences[name2]

            # Create SEQ format file
            pair_file = os.path.join(output_dir, f"pair_{pair_count:03d}.txt")
            with open(pair_file, 'w') as f:
                f.write(f">{seq1}\n")
                f.write(f"<{seq2}\n")

            # Create mapping file
            mapping_file = os.path.join(output_dir, f"pair_{pair_count:03d}_mapping.txt")
            with open(mapping_file, 'w') as f:
                f.write(f"{name1}\t{name2}\n")

            pair_count += 1

    return pair_count

def create_categorical_pairs(categories, output_dir):
    """Create pairwise files for each category"""
    pair_count = 0
    
    for category_name, sequences in categories.items():
        if len(sequences) < 2:
            print(f"Category '{category_name}' has only {len(sequences)} sequence(s), skipping...")
            continue
            
        print(f"Processing category '{category_name}' with {len(sequences)} sequences...")
        
        # Create all pairwise combinations within this category
        for i in range(len(sequences)):
            for j in range(i+1, len(sequences)):
                name1, seq1 = sequences[i]
                name2, seq2 = sequences[j]
                
                # Create descriptive filename based on sequence names
                sanitized_name1 = sanitize_filename(name1)
                sanitized_name2 = sanitize_filename(name2)
                pair_filename = f"cat_{category_name}_{sanitized_name1}_vs_{sanitized_name2}.txt"
                
                # Create SEQ format file
                pair_file = os.path.join(output_dir, pair_filename)
                with open(pair_file, 'w') as f:
                    f.write(f">{seq1}\n")
                    f.write(f"<{seq2}\n")
                
                # Create mapping file with same base name
                mapping_filename = f"cat_{category_name}_{sanitized_name1}_vs_{sanitized_name2}_mapping.txt"
                mapping_file = os.path.join(output_dir, mapping_filename)
                with open(mapping_file, 'w') as f:
                    f.write(f"{name1}\t{name2}\n")
                
                pair_count += 1
    
    return pair_count

def calculate_similarity_from_fasta(fasta_file):
    """Calculate similarity from aligned FASTA sequences"""
    try:
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
        
        # Find the two aligned sequences
        seq1 = ""
        seq2 = ""
        for line in lines:
            if line.startswith('>'):
                continue
            elif seq1 == "":
                seq1 = line.strip()
            elif seq2 == "":
                seq2 = line.strip()
                break
        
        if seq1 and seq2:
            # Calculate similarity
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
            total_positions = len(seq1)
            similarity = matches / total_positions if total_positions > 0 else 0
            return similarity, matches, total_positions
    
    except Exception as e:
        print(f"Error processing {fasta_file}: {e}")
    
    return None, None, None

def fix_fasta_names(results_dir, prefix="cat"):
    """Fix FASTA file names to use real sequence IDs"""
    print(f"Fixing FASTA names for {prefix} comparisons...")
    
    # Find all aligned FASTA files
    fasta_files = glob.glob(os.path.join(results_dir, f"{prefix}_*_aligned.fasta"))
    
    for fasta_file in fasta_files:
        # Extract the base name to find the corresponding mapping file
        base_name = os.path.basename(fasta_file).replace("_aligned.fasta", "")
        mapping_file = os.path.join(results_dir, f"{base_name}_mapping.txt")
        
        if os.path.exists(mapping_file):
            # Read the mapping
            with open(mapping_file, 'r') as f:
                line = f.readline().strip()
                names = line.split('\t')
                if len(names) == 2:
                    seq1_name, seq2_name = names[0], names[1]
                    
                    # Read the FASTA file
                    with open(fasta_file, 'r') as f:
                        content = f.read()
                    
                    # Replace the generic names with real names
                    content = content.replace('>pair_0_query', f'>{seq1_name}')
                    content = content.replace('>pair_0_target', f'>{seq2_name}')
                    
                    # Write back the fixed content
                    with open(fasta_file, 'w') as f:
                        f.write(content)
                    
                    print(f"Fixed {base_name}: {seq1_name} vs {seq2_name}")

def analyze_results(results_dir, analysis_type="categorical"):
    """Analyze alignment results and generate summary"""
    print(f"Analyzing {analysis_type} alignment results...")
    
    # Initialize results storage
    if analysis_type == "categorical":
        categories = {
            "left_flank": [],
            "right_flank": [],
            "repeat": []
        }
        
        # Find all aligned FASTA files
        fasta_files = glob.glob(os.path.join(results_dir, "cat_*_aligned.fasta"))
        
        for fasta_file in fasta_files:
            # Extract the base name to find the corresponding mapping file
            base_name = os.path.basename(fasta_file).replace("_aligned.fasta", "")
            mapping_file = os.path.join(results_dir, f"{base_name}_mapping.txt")
            
            # Determine category from filename
            if "left_flank" in base_name:
                category = "left_flank"
            elif "right_flank" in base_name:
                category = "right_flank"
            elif "repeat" in base_name:
                category = "repeat"
            else:
                continue
            
            if os.path.exists(mapping_file):
                # Read the mapping
                with open(mapping_file, 'r') as f:
                    line = f.readline().strip()
                    names = line.split('\t')
                    if len(names) == 2:
                        seq1_name, seq2_name = names[0], names[1]
                        
                        # Calculate similarity
                        similarity, matches, total = calculate_similarity_from_fasta(fasta_file)
                        if similarity is not None:
                            categories[category].append({
                                'seq1': seq1_name,
                                'seq2': seq2_name,
                                'similarity': similarity,
                                'matches': matches,
                                'total': total
                            })
                            print(f"  {seq1_name} vs {seq2_name}: {similarity:.3f} ({matches}/{total})")
    else:
        # Pairwise analysis
        categories = {"pairwise": []}
        fasta_files = glob.glob(os.path.join(results_dir, "aligned_*.fasta"))
        
        for fasta_file in fasta_files:
            # Extract pair number
            pair_num = os.path.basename(fasta_file).replace("aligned_", "").replace(".fasta", "")
            mapping_file = os.path.join(results_dir, f"pair_{pair_num}_mapping.txt")
            
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    line = f.readline().strip()
                    names = line.split('\t')
                    if len(names) == 2:
                        seq1_name, seq2_name = names[0], names[1]
                        
                        similarity, matches, total = calculate_similarity_from_fasta(fasta_file)
                        if similarity is not None:
                            categories["pairwise"].append({
                                'seq1': seq1_name,
                                'seq2': seq2_name,
                                'similarity': similarity,
                                'matches': matches,
                                'total': total
                            })
    
    # Create summary
    summary_file = os.path.join(results_dir, f"{analysis_type}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"{analysis_type.title()} Alignment Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for category, results in categories.items():
            if results:
                f.write(f"{category.upper()} COMPARISONS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total comparisons: {len(results)}\n")
                
                similarities = [r['similarity'] for r in results]
                avg_similarity = np.mean(similarities)
                min_similarity = np.min(similarities)
                max_similarity = np.max(similarities)
                
                f.write(f"Average similarity: {avg_similarity:.3f}\n")
                f.write(f"Range: {min_similarity:.3f} - {max_similarity:.3f}\n\n")
                
                # Find most similar and most different
                most_similar = max(results, key=lambda x: x['similarity'])
                most_different = min(results, key=lambda x: x['similarity'])
                
                f.write(f"Most similar pair: {most_similar['seq1']} vs {most_similar['seq2']} ({most_similar['similarity']:.3f})\n")
                f.write(f"Most different pair: {most_different['seq1']} vs {most_different['seq2']} ({most_different['similarity']:.3f})\n\n")
                
                # List all comparisons
                f.write("All comparisons:\n")
                for result in sorted(results, key=lambda x: x['similarity'], reverse=True):
                    f.write(f"  {result['seq1']} vs {result['seq2']}: {result['similarity']:.3f}\n")
                f.write("\n")
    
    # Create CSV summary
    all_results = []
    for category, results in categories.items():
        for result in results:
            all_results.append({
                'category': category,
                'seq1': result['seq1'],
                'seq2': result['seq2'],
                'similarity': result['similarity'],
                'matches': result['matches'],
                'total': result['total']
            })
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(results_dir, f"{analysis_type}_similarities.csv"), index=False)
        
        print(f"\nResults saved to {results_dir}/")
        print(f"- {analysis_type}_summary.txt: Summary statistics")
        print(f"- {analysis_type}_similarities.csv: Detailed results")
        
        # Print summary
        print(f"\nSUMMARY:")
        for category, results in categories.items():
            if results:
                similarities = [r['similarity'] for r in results]
                print(f"  {category}: {len(results)} comparisons, avg similarity: {np.mean(similarities):.3f}")

def organize_output_files(results_dir, organized_dir):
    """Organize output files into folders by type"""
    os.makedirs(organized_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        'aligned_fastas': 'aligned_fastas',
        'detailed_reports': 'detailed_reports', 
        'mapping_files': 'mapping_files',
        'logs': 'logs',
        'summary_files': 'summary_files',
        'input_files': 'input_files'
    }
    
    for subdir in subdirs.values():
        os.makedirs(os.path.join(organized_dir, subdir), exist_ok=True)
    
    print("Organizing output files...")
    
    # Move aligned FASTA files
    fasta_files = glob.glob(os.path.join(results_dir, "*_aligned.fasta"))
    for fasta_file in fasta_files:
        if "mapping" not in fasta_file:  # Skip the mapping files
            dest = os.path.join(organized_dir, 'aligned_fastas', os.path.basename(fasta_file))
            shutil.move(fasta_file, dest)
            print(f"Moved: {os.path.basename(fasta_file)} → aligned_fastas/")
    
    # Move detailed reports
    detailed_files = glob.glob(os.path.join(results_dir, "*_detailed.txt"))
    for detailed_file in detailed_files:
        dest = os.path.join(organized_dir, 'detailed_reports', os.path.basename(detailed_file))
        shutil.move(detailed_file, dest)
        print(f"Moved: {os.path.basename(detailed_file)} → detailed_reports/")
    
    # Move mapping files
    mapping_files = glob.glob(os.path.join(results_dir, "*_mapping.txt"))
    for mapping_file in mapping_files:
        dest = os.path.join(organized_dir, 'mapping_files', os.path.basename(mapping_file))
        shutil.move(mapping_file, dest)
        print(f"Moved: {os.path.basename(mapping_file)} → mapping_files/")
    
    # Move log files
    log_files = glob.glob(os.path.join(results_dir, "*_log.txt"))
    for log_file in log_files:
        dest = os.path.join(organized_dir, 'logs', os.path.basename(log_file))
        shutil.move(log_file, dest)
        print(f"Moved: {os.path.basename(log_file)} → logs/")
    
    # Move summary files
    summary_files = glob.glob(os.path.join(results_dir, "*summary*.txt"))
    summary_files.extend(glob.glob(os.path.join(results_dir, "*similarities*.csv")))
    for summary_file in summary_files:
        dest = os.path.join(organized_dir, 'summary_files', os.path.basename(summary_file))
        shutil.move(summary_file, dest)
        print(f"Moved: {os.path.basename(summary_file)} → summary_files/")
    
    # Move input files
    input_files = glob.glob(os.path.join(results_dir, "*.txt"))
    input_files = [f for f in input_files if not any(suffix in f for suffix in ['_aligned', '_detailed', '_mapping', '_log', 'summary'])]
    for input_file in input_files:
        dest = os.path.join(organized_dir, 'input_files', os.path.basename(input_file))
        shutil.move(input_file, dest)
        print(f"Moved: {os.path.basename(input_file)} → input_files/")
    
    # Remove empty directories
    try:
        os.rmdir(results_dir)
        print(f"Removed empty directory: {results_dir}")
    except OSError:
        print(f"Directory {results_dir} not empty or already removed")
    
    print(f"\nOrganization complete! Results in: {organized_dir}/")

# Create Typer app
app = typer.Typer(help="WFA2 Sequence Analysis Tool - Fast pairwise sequence alignment")

# Alignment mode enum
from enum import Enum


class AlignmentMode(str, Enum):
    default = "default"
    global_mode = "global"
    ends_free = "ends-free"

@app.command()
def align(
    seq1: str = typer.Argument(..., help="First sequence to align"),
    seq2: str = typer.Argument(..., help="Second sequence to align"),
    mode: AlignmentMode = typer.Option(AlignmentMode.default, "--mode", "-m", help="Alignment mode"),
    global_align: bool = typer.Option(False, "--global", help="Use global alignment"),
    ends_free: bool = typer.Option(False, "--ends-free", help="Use ends-free alignment")
):
    """Quick alignment of two sequences without needing a FASTA file"""
    
    # Handle alignment mode conflicts
    if global_align and ends_free:
        typer.echo("Error: Cannot use both --global and --ends-free options", err=True)
        raise typer.Exit(1)
    
    # Determine alignment parameters
    if global_align or mode == AlignmentMode.global_mode:
        alignment_flag = "--wfa-span global"
        alignment_type = "global"
    elif ends_free or mode == AlignmentMode.ends_free:
        alignment_flag = "--wfa-span ends-free,0,0,0,0"
        alignment_type = "ends-free"
    else:
        alignment_flag = ""
        alignment_type = "default"
    
    # Create temporary input file
    temp_input = "temp_align_input.txt"
    with open(temp_input, 'w') as f:
        f.write(f">{seq1}\n")
        f.write(f"<{seq2}\n")
    
    # Run alignment
    temp_output_fasta = "temp_align_output.fasta"
    temp_output_detailed = "temp_align_output_detailed.txt"
    
    # Get the directory where this script is located (WFA2-lib root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    align_benchmark_path = os.path.join(script_dir, "bin", "align_benchmark")
    
    cmd = f"{align_benchmark_path} --algorithm edit-wfa {alignment_flag} --input {temp_input} --output-fasta {temp_output_fasta} --output-full {temp_output_detailed}"
    
    typer.echo(f"Running {alignment_type} alignment...")
    typer.echo(f"Sequence 1: {seq1}")
    typer.echo(f"Sequence 2: {seq2}")
    typer.echo()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Show aligned sequences
            if os.path.exists(temp_output_fasta):
                typer.echo("=== ALIGNED SEQUENCES ===")
                with open(temp_output_fasta, 'r') as f:
                    content = f.read()
                    typer.echo(content)
            
            # Show detailed results
            if os.path.exists(temp_output_detailed):
                typer.echo("=== ALIGNMENT DETAILS ===")
                with open(temp_output_detailed, 'r') as f:
                    line = f.readline().strip()
                    parts = line.split('\t')
                    if len(parts) >= 6:
                        seq1_len, seq2_len, edit_dist = parts[0], parts[1], parts[2]
                        cigar = parts[6] if len(parts) > 6 else "N/A"
                        similarity = 1.0 - (int(edit_dist) / max(int(seq1_len), int(seq2_len)))
                        typer.echo(f"Sequence 1 length: {seq1_len}")
                        typer.echo(f"Sequence 2 length: {seq2_len}")
                        typer.echo(f"Edit distance: {edit_dist}")
                        typer.echo(f"Similarity: {similarity:.4f}")
                        typer.echo(f"CIGAR: {cigar}")
            
            # Calculate and show similarity from FASTA
            try:
                fasta_result = calculate_similarity_from_fasta(temp_output_fasta)
                if fasta_result[0] is not None:
                    similarity_score, matches, total_positions = fasta_result
                    typer.echo(f"FASTA-based similarity: {similarity_score:.4f} ({matches}/{total_positions} matches)")
            except Exception as e:
                typer.echo(f"Note: Could not calculate FASTA similarity: {e}")
            
        else:
            typer.echo(f"Error running alignment: {result.stderr}", err=True)
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    
    # Clean up temporary files
    for temp_file in [temp_input, temp_output_fasta, temp_output_detailed]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.command()
def pairwise(
    input_fasta: str = typer.Argument(..., help="Input FASTA file"),
    global_align: bool = typer.Option(False, "--global", help="Use global alignment"),
    ends_free: bool = typer.Option(False, "--ends-free", help="Use ends-free alignment")
):
    """Run pairwise analysis on all sequences in a FASTA file"""
    
    # Handle alignment mode conflicts
    if global_align and ends_free:
        typer.echo("Error: Cannot use both --global and --ends-free options", err=True)
        raise typer.Exit(1)
    
    output_dir = "pairwise_results"
    
    typer.echo(f"Running pairwise analysis on {input_fasta}...")
    
    # Extract sequences
    sequences = extract_sequences(input_fasta)
    typer.echo(f"Found {len(sequences)} sequences")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pairwise files
    pair_count = create_pairwise_files(sequences, output_dir)
    typer.echo(f"Created {pair_count} pairwise comparison files")
    
    # Determine alignment parameters
    if global_align:
        alignment_flag = "--wfa-span global"
        alignment_type = "global"
    elif ends_free:
        alignment_flag = "--wfa-span ends-free,0,0,0,0"
        alignment_type = "ends-free"
    else:
        alignment_flag = ""
        alignment_type = "default"
    
    # Get the directory where this script is located (WFA2-lib root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    align_benchmark_path = os.path.join(script_dir, "bin", "align_benchmark")
    
    typer.echo(f"Please run {alignment_type} alignments using the WFA2 tool:")
    typer.echo(f"for file in {output_dir}/pair_*.txt; do")
    typer.echo(f"  {align_benchmark_path} --algorithm edit-wfa {alignment_flag} --input \"$file\" --output-fasta \"${{file%.txt}}_aligned.fasta\" --output-full \"${{file%.txt}}_detailed.txt\" > \"${{file%.txt}}_log.txt\" 2>&1")
    typer.echo("done")
    
    # Fix FASTA names
    fix_fasta_names(output_dir, "pair")
    
    # Analyze results
    typer.echo("Analyzing pairwise alignment results...")

@app.command()
def categorical(
    input_fasta: str = typer.Argument(..., help="Input FASTA file"),
    global_align: bool = typer.Option(False, "--global", help="Use global alignment"),
    ends_free: bool = typer.Option(False, "--ends-free", help="Use ends-free alignment")
):
    """Run categorical analysis on sequences grouped by type"""
    
    # Handle alignment mode conflicts
    if global_align and ends_free:
        typer.echo("Error: Cannot use both --global and --ends-free options", err=True)
        raise typer.Exit(1)
    
    output_dir = "categorical_results"
    
    typer.echo(f"Running categorical analysis on {input_fasta}...")
    
    # Extract and categorize sequences
    sequences = extract_sequences(input_fasta)
    categories = categorize_sequences(sequences)
    
    typer.echo(f"Found {len(sequences)} sequences:")
    for category, seqs in categories.items():
        typer.echo(f"  {category}: {len(seqs)} sequences")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create categorical files
    pair_count = create_categorical_pairs(categories, output_dir)
    typer.echo(f"Created {pair_count} categorical comparison files")
    
    # Determine alignment parameters
    if global_align:
        alignment_flag = "--wfa-span global"
        alignment_type = "global"
    elif ends_free:
        alignment_flag = "--wfa-span ends-free,0,0,0,0"
        alignment_type = "ends-free"
    else:
        alignment_flag = ""
        alignment_type = "default"
    
    # Get the directory where this script is located (WFA2-lib root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    align_benchmark_path = os.path.join(script_dir, "bin", "align_benchmark")
    
    typer.echo(f"Please run {alignment_type} alignments using the WFA2 tool:")
    typer.echo(f"for file in {output_dir}/cat_*.txt; do")
    typer.echo(f"  {align_benchmark_path} --algorithm edit-wfa {alignment_flag} --input \"$file\" --output-fasta \"${{file%.txt}}_aligned.fasta\" --output-full \"${{file%.txt}}_detailed.txt\" > \"${{file%.txt}}_log.txt\" 2>&1")
    typer.echo("done")
    
    # Fix FASTA names
    fix_fasta_names(output_dir, "cat")
    
    # Analyze results
    typer.echo("Analyzing categorical alignment results...")

@app.command()
def organize(
    results_dir: str = typer.Argument(..., help="Results directory to organize")
):
    """Organize output files into folders by type"""
    organize_output_files(results_dir)

@app.command()
def analyze(
    results_dir: str = typer.Argument(..., help="Results directory to analyze")
):
    """Analyze existing alignment results"""
    if "categorical" in results_dir.lower():
        analyze_results(results_dir, analysis_type="categorical")
    else:
        analyze_results(results_dir, analysis_type="pairwise")

@app.command()
def cleanup():
    """Remove intermediate files and directories"""
    patterns_to_remove = [
        "pair_*.txt", "pair_*.fasta", "pair_*_detailed.txt", "pair_*_log.txt",
        "cat_*.txt", "cat_*.fasta", "cat_*_detailed.txt", "cat_*_log.txt",
        "pairwise_results/", "categorical_results/", "temp_*.txt", "temp_*.fasta"
    ]
    
    removed_count = 0
    for pattern in patterns_to_remove:
        if "/" in pattern:
            # Directory
            if os.path.exists(pattern.rstrip("/")):
                shutil.rmtree(pattern.rstrip("/"))
                typer.echo(f"Removed directory: {pattern}")
                removed_count += 1
        else:
            # Files
            for file_path in glob.glob(pattern):
                os.remove(file_path)
                typer.echo(f"Removed file: {file_path}")
                removed_count += 1
    
    if removed_count == 0:
        typer.echo("No intermediate files found to clean up")
    else:
        typer.echo(f"Cleaned up {removed_count} files/directories")

# Entry point for the CLI
def main():
    """Entry point for the wfa command"""
    app()

if __name__ == "__main__":
    main()
