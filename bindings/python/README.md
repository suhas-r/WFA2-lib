# PyWFA2 - Python Bindings for WFA2

High-performance Python bindings for the [WFA2 sequence alignment library](https://github.com/smarco/WFA2-lib).

## Overview

PyWFA2 provides efficient Python access to the WaveFont Alignment Algorithm (WFA2), enabling fast and accurate sequence alignment with minimal overhead. These bindings use Cython for optimal performance, making them suitable for both interactive use and high-throughput applications.

## Features

- **High Performance**: Cython-based bindings with minimal Python overhead
- **Multiple Distance Metrics**: Edit distance, gap-linear, gap-affine, and dual-gap affine
- **Flexible Alignment Modes**: Global, semi-global, and ends-free alignment
- **Memory Efficient**: Multiple memory usage strategies (high, medium, low, ultralow)
- **Pythonic API**: Clean, intuitive interface with type hints
- **Reusable Objects**: Efficient aligner reuse for multiple alignments

## Installation

### Prerequisites

1. **Build WFA2 library** (required before installing Python bindings):
   ```bash
   cd /path/to/WFA2-lib
   make lib_wfa
   ```

2. **Install Python dependencies**:
   ```bash
   pip install cython numpy
   ```

### Install PyWFA2

From the WFA2-lib directory:
```bash
cd bindings/python
pip install -e .
```

## Quick Start

### Basic Usage

```python
import pywfa2

# Quick global alignment
result = pywfa2.align_global("ATCGATCG", "ATCGATCG")
print(f"Score: {result.score}")
print(f"Status: {result.status_message}")

# Quick semi-global alignment (pattern fully aligned, text free ends)
result = pywfa2.align_semi_global("ATCG", "XXATCGXX")
print(f"Score: {result.score}")
```

### Advanced Usage with Custom Parameters

```python
from pywfa2 import WFA2Aligner, DistanceMetric, AlignmentScope, MemoryMode

# Create a reusable aligner with gap-affine scoring
aligner = WFA2Aligner(
    distance_metric=DistanceMetric.GAP_AFFINE,
    alignment_scope=AlignmentScope.FULL_ALIGNMENT,
    memory_mode=MemoryMode.HIGH,
    match_score=0,
    mismatch_penalty=1,
    gap_open_penalty=0,
    gap_extend_penalty=1
)

# Perform multiple alignments (efficient - reuses internal structures)
sequences = [
    ("ATCGATCGATCGATCG", "ATCGAAATCGATCG"),
    ("TCCAGAGGCAGTGGG", "CTCAGAGGCAGTGGG"),
    # ... more sequence pairs
]

results = []
for seq1, seq2 in sequences:
    result = aligner.align_semi_global(seq1, seq2)
    results.append(result)
    print(f"Score: {result.score}, Success: {result.is_success}")
```

### Integration with Your Configuration

Match your exact WFA Builder settings:

```python
# Your WFA Builder configuration: match=0, mismatch=1, gap_open=0, gap_extend=1
aligner = WFA2Aligner(
    distance_metric=DistanceMetric.GAP_AFFINE,
    alignment_scope=AlignmentScope.FULL_ALIGNMENT,
    match_score=0,
    mismatch_penalty=1, 
    gap_open_penalty=0,
    gap_extend_penalty=1
)

# Semi-global alignment: pattern fully aligned, text free ends
result = aligner.align_semi_global(pattern, text)
```

## API Reference

### Classes

#### `WFA2Aligner`

Main aligner class for performing sequence alignments.

**Constructor Parameters:**
- `distance_metric`: Distance metric (`DistanceMetric.EDIT`, `GAP_AFFINE`, etc.)
- `alignment_scope`: Compute score only or full alignment
- `memory_mode`: Memory usage strategy
- `match_score`: Score for matches (default: 0)
- `mismatch_penalty`: Penalty for mismatches (default: 1)
- `gap_open_penalty`: Penalty for opening gaps (default: 0)
- `gap_extend_penalty`: Penalty for extending gaps (default: 1)

**Methods:**
- `align_global(pattern, text)`: Global (end-to-end) alignment
- `align_semi_global(pattern, text)`: Semi-global alignment
- `align_ends_free(pattern, text, ...)`: Custom ends-free alignment

#### `AlignmentResult`

Result object containing alignment information.

**Properties:**
- `score`: Alignment score
- `status`: Status code
- `is_success`: Whether alignment succeeded
- `status_message`: Human-readable status
- `cigar`: CIGAR string (if full alignment computed)

### Convenience Functions

- `align_global(pattern, text, **kwargs)`: Quick global alignment
- `align_semi_global(pattern, text, **kwargs)`: Quick semi-global alignment
- `align_sequences(pattern, text, mode, **kwargs)`: General alignment function

### Enums

- `DistanceMetric`: `EDIT`, `GAP_LINEAR`, `GAP_AFFINE`, `GAP_AFFINE_2P`
- `AlignmentScope`: `SCORE_ONLY`, `FULL_ALIGNMENT`
- `AlignmentMode`: `GLOBAL`, `SEMI_GLOBAL`, `ENDS_FREE`
- `MemoryMode`: `HIGH`, `MEDIUM`, `LOW`, `ULTRALOW`

## Performance Comparison

PyWFA2 provides significant performance improvements over subprocess-based approaches:

```python
import time
import pywfa2

# Your sequences
seq1 = "ATCGATCG" * 100  # 800bp
seq2 = "ATCGAACG" * 100  # 800bp

# Time the Python bindings
aligner = pywfa2.WFA2Aligner(distance_metric=pywfa2.DistanceMetric.GAP_AFFINE)

start = time.time()
for _ in range(1000):
    result = aligner.align_global(seq1, seq2)
python_time = time.time() - start

print(f"1000 alignments in {python_time:.3f}s ({1000/python_time:.1f} alignments/sec)")
```

Expected performance: **10-100x faster** than subprocess calls, depending on sequence length and system.

## Examples

See the `examples/` directory for more comprehensive examples:

- `basic_alignment.py`: Simple alignment examples
- `batch_processing.py`: Processing multiple sequences efficiently
- `custom_scoring.py`: Using different scoring schemes
- `memory_optimization.py`: Memory-efficient alignment strategies

## Troubleshooting

### Build Issues

1. **"WFA2 library not found"**: Make sure you've built the library with `make lib_wfa`
2. **Compilation errors**: Ensure you have a C compiler and development headers installed
3. **Missing dependencies**: Install `cython` and `numpy` before building

### Runtime Issues

1. **Import errors**: Make sure the WFA2 library is in your library path
2. **Memory errors**: Try using a lower memory mode (`MemoryMode.LOW` or `ULTRALOW`)
3. **Performance issues**: Reuse aligner objects instead of creating new ones for each alignment

## Contributing

Contributions are welcome! Please see the main WFA2 repository for contribution guidelines.

## License

PyWFA2 is distributed under the same MIT license as the WFA2 library.

## Citation

If you use PyWFA2 in your research, please cite the original WFA2 paper:

> Marco-Sola, S., Moure, J.C., Moreto, M. et al. Fast gap-affine pairwise alignment using the wavefront algorithm. Bioinformatics 37, 456–463 (2021).
