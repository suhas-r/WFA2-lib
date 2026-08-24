"""
Type stubs for pywfa2 - WFA2 Python bindings.

This file provides type hints and IDE support for the compiled Cython module.
"""

from enum import Enum
from typing import Optional, Union

__version__: str
__author__: str

class AlignmentScope(Enum):
    """Alignment computation scope"""
    SCORE_ONLY: str
    FULL_ALIGNMENT: str

class AlignmentMode(Enum):
    """Alignment mode"""
    GLOBAL: str
    SEMI_GLOBAL: str
    ENDS_FREE: str

class DistanceMetric(Enum):
    """Distance metric for alignment"""
    EDIT: str
    GAP_LINEAR: str
    GAP_AFFINE: str
    GAP_AFFINE_2P: str

class Heuristic(Enum):
    """Wavefront pruning heuristic. NONE keeps WFA exact."""
    NONE: str
    WFADAPTIVE: str

class MemoryMode(Enum):
    """Memory usage strategy"""
    HIGH: str
    MEDIUM: str
    LOW: str
    ULTRALOW: str

class AlignmentResult:
    """Result of a sequence alignment operation."""
    score: int
    status: int
    cigar: Optional[str]
    pattern_aligned: Optional[str]
    text_aligned: Optional[str]
    
    def __init__(
        self,
        score: int,
        status: int,
        cigar: Optional[str] = None,
        pattern_aligned: Optional[str] = None,
        text_aligned: Optional[str] = None
    ) -> None: ...
    
    @property
    def is_success(self) -> bool:
        """Check if alignment was successful"""
        ...
    
    @property
    def status_message(self) -> str:
        """Get human-readable status message"""
        ...

class WFA2Aligner:
    """
    High-performance sequence aligner using the WaveFont Algorithm (WFA2).
    
    This class provides a Pythonic interface to the WFA2 library, offering
    fast and accurate sequence alignment with various distance metrics.
    """
    
    def __init__(
        self,
        distance_metric: Union[DistanceMetric, str] = DistanceMetric.EDIT,
        alignment_scope: Union[AlignmentScope, str] = AlignmentScope.FULL_ALIGNMENT,
        memory_mode: Union[MemoryMode, str] = MemoryMode.HIGH,
        match_score: int = 0,
        mismatch_penalty: int = 1,
        gap_open_penalty: int = 0,
        gap_extend_penalty: int = 1,
        max_alignment_steps: int = -1,
        max_memory_mb: int = -1,
        heuristic: Union[Heuristic, str] = Heuristic.NONE,
        heuristic_min_wavefront_length: int = 10,
        heuristic_max_distance_threshold: int = 50,
        heuristic_steps_between_cutoffs: int = 1
    ) -> None:
        """
        Initialize WFA2 aligner.
        
        Args:
            distance_metric: Distance metric to use (edit, gap_linear, gap_affine, gap_affine_2p)
            alignment_scope: Whether to compute score only or full alignment
            memory_mode: Memory usage strategy (high, med, low, ultralow)
            match_score: Score for matches (default: 0)
            mismatch_penalty: Penalty for mismatches (default: 1)
            gap_open_penalty: Penalty for opening gaps (default: 0)
            gap_extend_penalty: Penalty for extending gaps (default: 1)
            max_alignment_steps: Maximum alignment steps (-1 for unlimited)
            max_memory_mb: Maximum memory in MB (-1 for unlimited)
            heuristic: Pruning strategy (default: none, which keeps WFA exact)
            heuristic_min_wavefront_length: wf-adaptive parameter
            heuristic_max_distance_threshold: wf-adaptive parameter
            heuristic_steps_between_cutoffs: wf-adaptive parameter
        """
        ...
    
    def align_global(
        self,
        *,
        pattern: str,
        text: str
    ) -> AlignmentResult:
        """
        Perform global (end-to-end) alignment.
        
        Both sequences must align completely from start to end.
        
        Args:
            pattern: First sequence (keyword-only)
            text: Second sequence (keyword-only)
            
        Returns:
            AlignmentResult with score, status, CIGAR string, and input sequences
            
        Example:
            >>> aligner = WFA2Aligner()
            >>> result = aligner.align_global(pattern="ACGT", text="ACCT")
            >>> print(result.score, result.cigar)
        """
        ...
    
    def align_semi_global(
        self,
        *,
        pattern: str,
        text: str,
        pattern_begin_free: int = 0,
        pattern_end_free: int = 0,
        text_begin_free: Optional[int] = None,
        text_end_free: Optional[int] = None
    ) -> AlignmentResult:
        """
        Perform semi-global alignment (ends-free alignment).
        
        ⚠️  IMPORTANT: Semi-global alignment is ASYMMETRIC!
        - The 'pattern' must align completely (query sequence)
        - The 'text' can have unpenalized free ends (target sequence)
        - Swapping pattern↔text will give different results!
        
        Args:
            pattern: Query sequence - must fully align (keyword-only)
            text: Target sequence - can have free ends (keyword-only)
            pattern_begin_free: Free bases at start of pattern (default: 0, keyword-only)
            pattern_end_free: Free bases at end of pattern (default: 0, keyword-only)
            text_begin_free: Free bases at start of text (default: len(text), keyword-only)
            text_end_free: Free bases at end of text (default: len(text), keyword-only)
            
        Returns:
            AlignmentResult with alignment details
            
        Example:
            # Find where short_read aligns in long_genome (typical use case)
            >>> aligner = WFA2Aligner()
            >>> result = aligner.align_semi_global(pattern=short_read, text=long_genome)
        """
        ...
    
    def align_ends_free(
        self,
        *,
        pattern: str,
        text: str,
        pattern_begin_free: int = 0,
        pattern_end_free: int = 0,
        text_begin_free: int = 0,
        text_end_free: int = 0
    ) -> AlignmentResult:
        """
        Perform ends-free alignment with custom free end parameters.
        
        Allows fine-grained control over which sequence ends are penalized.
        
        Args:
            pattern: First sequence (keyword-only)
            text: Second sequence (keyword-only)
            pattern_begin_free: Free bases at start of pattern (default: 0, keyword-only)
            pattern_end_free: Free bases at end of pattern (default: 0, keyword-only)
            text_begin_free: Free bases at start of text (default: 0, keyword-only)
            text_end_free: Free bases at end of text (default: 0, keyword-only)
            
        Returns:
            AlignmentResult with alignment details
            
        Example:
            # Custom alignment where both sequences can have free ends
            >>> aligner = WFA2Aligner()
            >>> result = aligner.align_ends_free(
            ...     pattern=seq1, text=seq2,
            ...     pattern_end_free=10, text_end_free=10
            ... )
        """
        ...
