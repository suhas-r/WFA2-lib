# Python bindings for WFA2 library
# High-performance sequence alignment using the WaveFont Algorithm

__version__ = "2.3.0"
__author__ = "WFA2 Python Bindings"

import cython
from libc.stdlib cimport malloc, free
from libc.string cimport strlen
cimport wfa2_lib

# Import Python modules
from enum import Enum
from typing import Optional, Tuple, Union

# Python enums for user-friendly API
class AlignmentScope(Enum):
    """Alignment computation scope"""
    SCORE_ONLY = "score"
    FULL_ALIGNMENT = "alignment"

class AlignmentMode(Enum):
    """Alignment mode"""
    GLOBAL = "global"
    SEMI_GLOBAL = "semi_global" 
    ENDS_FREE = "ends_free"

class DistanceMetric(Enum):
    """Distance metric for alignment"""
    EDIT = "edit"
    GAP_LINEAR = "gap_linear"
    GAP_AFFINE = "gap_affine"
    GAP_AFFINE_2P = "gap_affine_2p"

class Heuristic(Enum):
    """Wavefront pruning heuristic.

    WFA2-lib's `wavefront_aligner_attr_default` enables wf-adaptive pruning
    (wavefront_attributes.c:72). Pruning can discard the diagonal carrying the optimal
    alignment, so NONE is the only setting that preserves WFA's exactness guarantee, and
    it is this binding's default. WFADAPTIVE restores the library's attribute default.
    """
    NONE = "none"
    WFADAPTIVE = "wfadaptive"

class MemoryMode(Enum):
    """Memory usage strategy"""
    HIGH = "high"
    MEDIUM = "med"
    LOW = "low"
    ULTRALOW = "ultralow"

class AlignmentResult:
    """Result of sequence alignment"""
    def __init__(self, score: int, status: int, cigar: Optional[str] = None,
                 pattern_aligned: Optional[str] = None, text_aligned: Optional[str] = None):
        self.score = score
        self.status = status
        self.cigar = cigar
        self.pattern_aligned = pattern_aligned
        self.text_aligned = text_aligned
        
    @property
    def is_success(self) -> bool:
        """Check if alignment was successful"""
        return self.status >= 0
        
    @property
    def status_message(self) -> str:
        """Get human-readable status message"""
        if self.status == wfa2_lib.WF_STATUS_ALG_COMPLETED:
            return "Alignment completed successfully"
        elif self.status == wfa2_lib.WF_STATUS_ALG_PARTIAL:
            return "Partial alignment found"
        elif self.status == wfa2_lib.WF_STATUS_MAX_STEPS_REACHED:
            return "Maximum steps reached"
        elif self.status == wfa2_lib.WF_STATUS_OOM:
            return "Out of memory"
        else:
            return f"Unknown status: {self.status}"

cdef class WFA2Aligner:
    """
    High-performance sequence aligner using the WaveFont Algorithm (WFA2).
    
    This class provides a Pythonic interface to the WFA2 library, offering
    fast and accurate sequence alignment with various distance metrics.
    """
    
    cdef wfa2_lib.wavefront_aligner_t* _aligner
    cdef wfa2_lib.wavefront_aligner_attr_t _attributes
    cdef object _distance_metric
    cdef object _alignment_scope
    cdef object _memory_mode
    cdef object _heuristic
    
    def __init__(self,
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
                 heuristic_steps_between_cutoffs: int = 1):
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
            heuristic: Wavefront pruning strategy. Defaults to NONE, which keeps WFA's
                exactness guarantee. Pass WFADAPTIVE for WFA2-lib's own attribute default,
                which is faster but can return a suboptimal alignment.
            heuristic_min_wavefront_length: wf-adaptive parameter (ignored when NONE)
            heuristic_max_distance_threshold: wf-adaptive parameter (ignored when NONE)
            heuristic_steps_between_cutoffs: wf-adaptive parameter (ignored when NONE)
        """

        # Convert string enums to enum objects if needed
        if isinstance(distance_metric, str):
            distance_metric = DistanceMetric(distance_metric)
        if isinstance(heuristic, str):
            heuristic = Heuristic(heuristic)
        if isinstance(alignment_scope, str):
            alignment_scope = AlignmentScope(alignment_scope)
        if isinstance(memory_mode, str):
            memory_mode = MemoryMode(memory_mode)
            
        self._distance_metric = distance_metric
        self._alignment_scope = alignment_scope
        self._memory_mode = memory_mode
        self._heuristic = heuristic

        # Initialize attributes with defaults
        self._attributes = wfa2_lib.wavefront_aligner_attr_default
        
        # Set distance metric
        if distance_metric == DistanceMetric.EDIT:
            self._attributes.distance_metric = wfa2_lib.edit
        elif distance_metric == DistanceMetric.GAP_LINEAR:
            self._attributes.distance_metric = wfa2_lib.gap_linear
        elif distance_metric == DistanceMetric.GAP_AFFINE:
            self._attributes.distance_metric = wfa2_lib.gap_affine
        elif distance_metric == DistanceMetric.GAP_AFFINE_2P:
            self._attributes.distance_metric = wfa2_lib.gap_affine_2p
            
        # Set alignment scope
        if alignment_scope == AlignmentScope.SCORE_ONLY:
            self._attributes.alignment_scope = wfa2_lib.compute_score
        else:
            self._attributes.alignment_scope = wfa2_lib.compute_alignment
            
        # Set memory mode
        if memory_mode == MemoryMode.HIGH:
            self._attributes.memory_mode = wfa2_lib.wavefront_memory_high
        elif memory_mode == MemoryMode.MEDIUM:
            self._attributes.memory_mode = wfa2_lib.wavefront_memory_med
        elif memory_mode == MemoryMode.LOW:
            self._attributes.memory_mode = wfa2_lib.wavefront_memory_low
        elif memory_mode == MemoryMode.ULTRALOW:
            self._attributes.memory_mode = wfa2_lib.wavefront_memory_ultralow
            
        # Set penalties
        if distance_metric in (DistanceMetric.GAP_LINEAR, DistanceMetric.GAP_AFFINE, DistanceMetric.GAP_AFFINE_2P):
            self._attributes.affine_penalties.match = match_score
            self._attributes.affine_penalties.mismatch = mismatch_penalty
            self._attributes.affine_penalties.gap_opening = gap_open_penalty
            self._attributes.affine_penalties.gap_extension = gap_extend_penalty
            
        # Create the aligner first
        self._aligner = wfa2_lib.wavefront_aligner_new(&self._attributes)
        if self._aligner == NULL:
            raise MemoryError("Failed to create WFA2 aligner")

        # Apply the heuristic after construction. wavefront_aligner_new copies the
        # attribute defaults, which enable wf-adaptive pruning; the setters are the
        # only way to reach wf_aligner->heuristic and its bialigner copy.
        if heuristic == Heuristic.NONE:
            wfa2_lib.wavefront_aligner_set_heuristic_none(self._aligner)
        else:
            wfa2_lib.wavefront_aligner_set_heuristic_wfadaptive(
                self._aligner, heuristic_min_wavefront_length,
                heuristic_max_distance_threshold, heuristic_steps_between_cutoffs)


        # Set limits using setter functions
        if max_alignment_steps > 0:
            wfa2_lib.wavefront_aligner_set_max_alignment_steps(self._aligner, max_alignment_steps)
        if max_memory_mb > 0:
            memory_bytes = max_memory_mb * 1024 * 1024
            wfa2_lib.wavefront_aligner_set_max_memory(self._aligner, memory_bytes, memory_bytes * 2)
    
    def __dealloc__(self):
        """Clean up the aligner when object is destroyed"""
        if self._aligner != NULL:
            wfa2_lib.wavefront_aligner_delete(self._aligner)
            
    def align_global(self, *, pattern: str, text: str) -> AlignmentResult:
        """
        Perform global (end-to-end) alignment of two sequences.
        
        Args:
            pattern: First sequence (query) - keyword-only
            text: Second sequence (target) - keyword-only
            
        Returns:
            AlignmentResult object with score, status, and alignment details
        """
        wfa2_lib.wavefront_aligner_set_alignment_end_to_end(self._aligner)
        return self._align_sequences(pattern, text)
    
    def align_semi_global(self, *, pattern: str, text: str,
                         pattern_begin_free: int = 0, pattern_end_free: int = 0,
                         text_begin_free: Optional[int] = None, 
                         text_end_free: Optional[int] = None) -> AlignmentResult:
        """
        Perform semi-global alignment (ends-free alignment).
        
        ⚠️  IMPORTANT: Semi-global alignment is ASYMMETRIC!
        - The 'pattern' must align completely (query sequence)
        - The 'text' can have unpenalized free ends (target sequence)
        - Swapping pattern↔text will give different results!
        
        Args:
            pattern: First sequence (query) - must fully align - keyword-only
            text: Second sequence (target) - can have free ends - keyword-only
            pattern_begin_free: Free bases at start of pattern (keyword-only)
            pattern_end_free: Free bases at end of pattern (keyword-only)
            text_begin_free: Free bases at start of text (keyword-only, default: full text length)
            text_end_free: Free bases at end of text (keyword-only, default: full text length)
            
        Returns:
            AlignmentResult object with score, status, and alignment details
            
        Example:
            # Find where short_read aligns in long_genome (typical use case)
            result = aligner.align_semi_global(pattern=short_read, text=long_genome)
        """
        # Default to full text length for semi-global (pattern fully aligned, text free ends)
        if text_begin_free is None:
            text_begin_free = len(text)
        if text_end_free is None:
            text_end_free = len(text)
            
        wfa2_lib.wavefront_aligner_set_alignment_free_ends(
            self._aligner, pattern_begin_free, pattern_end_free, 
            text_begin_free, text_end_free)
        return self._align_sequences(pattern, text)
    
    def align_ends_free(self, *, pattern: str, text: str,
                       pattern_begin_free: int = 0, pattern_end_free: int = 0,
                       text_begin_free: int = 0, text_end_free: int = 0) -> AlignmentResult:
        """
        Perform ends-free alignment with custom free end parameters.
        
        Args:
            pattern: First sequence (query) - keyword-only
            text: Second sequence (target) - keyword-only
            pattern_begin_free: Free bases at start of pattern (keyword-only)
            pattern_end_free: Free bases at end of pattern (keyword-only)
            text_begin_free: Free bases at start of text (keyword-only)
            text_end_free: Free bases at end of text (keyword-only)
            
        Returns:
            AlignmentResult object with score, status, and alignment details
        """
        wfa2_lib.wavefront_aligner_set_alignment_free_ends(
            self._aligner, pattern_begin_free, pattern_end_free,
            text_begin_free, text_end_free)
        return self._align_sequences(pattern, text)
    
    def _align_sequences(self, str pattern, str text):
        """Internal method to perform the actual alignment"""
        # Convert Python strings to C strings
        cdef bytes pattern_bytes = pattern.encode('utf-8')
        cdef bytes text_bytes = text.encode('utf-8')
        cdef const char* pattern_c = pattern_bytes
        cdef const char* text_c = text_bytes
        cdef int pattern_len = len(pattern)
        cdef int text_len = len(text)
        
        # Perform alignment - the function returns the status
        cdef int status = wfa2_lib.wavefront_align(
            self._aligner, pattern_c, pattern_len, text_c, text_len)
            
        # Get the score directly from the CIGAR structure (WFA2 computes it for us!)
        cdef int score = 0
        if status >= 0 and self._aligner.cigar != NULL:
            score = self._aligner.cigar.score
        else:
            # If alignment failed, return error code
            score = status
        
        # Get CIGAR and aligned sequences if full alignment was requested
        cdef str cigar_str = None
        cdef str pattern_aligned = None
        cdef str text_aligned = None
        
        if self._alignment_scope == AlignmentScope.FULL_ALIGNMENT and status >= 0 and self._aligner.cigar != NULL:
            # Get CIGAR string using the operations buffer directly
            cigar_len = self._aligner.cigar.end_offset - self._aligner.cigar.begin_offset
            if cigar_len > 0:
                # The operations buffer contains the CIGAR (from begin_offset to end_offset)
                cigar_bytes = self._aligner.cigar.operations[
                    self._aligner.cigar.begin_offset:self._aligner.cigar.end_offset
                ]
                cigar_str = cigar_bytes.decode('ascii')
            
        return AlignmentResult(score, status, cigar_str, pattern_aligned, text_aligned)
    
    @property 
    def distance_metric(self) -> DistanceMetric:
        """Get the distance metric being used"""
        return self._distance_metric
    
    @property
    def alignment_scope(self) -> AlignmentScope:
        """Get the alignment scope (score only or full alignment)"""
        return self._alignment_scope
        
    @property
    def memory_mode(self) -> MemoryMode:
        """Get the memory usage mode"""
        return self._memory_mode

    @property
    def heuristic(self) -> Heuristic:
        """Get the wavefront pruning heuristic in effect"""
        return self._heuristic

# Convenience functions for quick alignment
def align_global(pattern: str, text: str, **kwargs) -> AlignmentResult:
    """
    Quick global alignment of two sequences.
    
    Args:
        pattern: First sequence
        text: Second sequence
        **kwargs: Additional arguments passed to WFA2Aligner constructor
        
    Returns:
        AlignmentResult object
    """
    aligner = WFA2Aligner(**kwargs)
    return aligner.align_global(pattern, text)

def align_semi_global(pattern: str, text: str, **kwargs) -> AlignmentResult:
    """
    Quick semi-global alignment of two sequences.
    
    Args:
        pattern: First sequence (fully aligned)
        text: Second sequence (free ends)
        **kwargs: Additional arguments passed to WFA2Aligner constructor
        
    Returns:
        AlignmentResult object
    """
    aligner = WFA2Aligner(**kwargs)
    return aligner.align_semi_global(pattern, text)

def align_sequences(pattern: str, text: str, 
                   mode: Union[AlignmentMode, str] = AlignmentMode.GLOBAL,
                   **kwargs) -> AlignmentResult:
    """
    General sequence alignment function.
    
    Args:
        pattern: First sequence
        text: Second sequence
        mode: Alignment mode (global, semi_global, ends_free)
        **kwargs: Additional arguments
        
    Returns:
        AlignmentResult object
    """
    if isinstance(mode, str):
        mode = AlignmentMode(mode)
        
    aligner = WFA2Aligner(**kwargs)
    
    if mode == AlignmentMode.GLOBAL:
        return aligner.align_global(pattern, text)
    elif mode == AlignmentMode.SEMI_GLOBAL:
        return aligner.align_semi_global(pattern, text)
    elif mode == AlignmentMode.ENDS_FREE:
        return aligner.align_ends_free(pattern, text)
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")
