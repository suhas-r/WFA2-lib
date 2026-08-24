# Cython declarations for WFA2 library
# This file declares the C API for use in Cython

from libc.stdint cimport int8_t, uint8_t, int32_t, uint32_t, uint64_t

cdef extern from "wavefront/wfa.h":
    # Constants
    cdef int WF_STATUS_ALG_COMPLETED
    cdef int WF_STATUS_ALG_PARTIAL
    cdef int WF_STATUS_MAX_STEPS_REACHED
    cdef int WF_STATUS_OOM
    
    # Enums
    ctypedef enum alignment_scope_t:
        compute_score
        compute_alignment
        
    ctypedef enum alignment_span_t:
        alignment_end2end
        alignment_endsfree
        
    ctypedef enum wavefront_memory_t:
        wavefront_memory_high
        wavefront_memory_med
        wavefront_memory_low
        wavefront_memory_ultralow
        
    ctypedef enum distance_metric_t:
        edit
        gap_linear
        gap_affine
        gap_affine_2p
        
    # Penalties structures
    ctypedef struct linear_penalties_t:
        int match
        int mismatch
        int indel
        
    ctypedef struct affine_penalties_t:
        int match
        int mismatch
        int gap_opening
        int gap_extension
        
    ctypedef struct affine2p_penalties_t:
        int match
        int mismatch
        int gap_opening1
        int gap_extension1
        int gap_opening2
        int gap_extension2
        
    # Alignment form structure
    ctypedef struct alignment_form_t:
        alignment_span_t span
        int pattern_begin_free
        int pattern_end_free
        int text_begin_free
        int text_end_free
        bint extension
        
    # System parameters structure
    ctypedef struct alignment_system_t:
        uint64_t max_memory_compact
        uint64_t max_memory_resident
        uint64_t max_memory_abort
        int verbose
        bint check_alignment_correct
        int max_num_threads
        int min_offsets_per_thread
        
    # CIGAR structure
    ctypedef struct cigar_t:
        char* operations
        int max_operations
        int begin_offset
        int end_offset
        int score
        int end_v
        int end_h
        
    # Penalties structure (must be before wavefront_aligner_t)
    ctypedef struct wavefront_penalties_t:
        distance_metric_t distance_metric
        int match
        int mismatch
        int gap_opening1
        int gap_extension1
        int gap_opening2
        int gap_extension2
        linear_penalties_t linear_penalties
        affine_penalties_t affine_penalties
        affine2p_penalties_t affine2p_penalties
        
    # Main aligner attributes structure
    ctypedef struct wavefront_aligner_attr_t:
        distance_metric_t distance_metric
        alignment_scope_t alignment_scope
        alignment_form_t alignment_form
        linear_penalties_t linear_penalties
        affine_penalties_t affine_penalties
        affine2p_penalties_t affine2p_penalties
        wavefront_memory_t memory_mode
        alignment_system_t system
        
    # Main aligner structure
    ctypedef struct wavefront_aligner_t:
        cigar_t* cigar
        alignment_scope_t alignment_scope
        wavefront_penalties_t penalties
        
    # Core functions
    wavefront_aligner_attr_t wavefront_aligner_attr_default
    
    wavefront_aligner_t* wavefront_aligner_new(wavefront_aligner_attr_t* attributes)
    void wavefront_aligner_delete(wavefront_aligner_t* wf_aligner)
    
    void wavefront_aligner_set_alignment_end_to_end(wavefront_aligner_t* wf_aligner)
    void wavefront_aligner_set_alignment_free_ends(
        wavefront_aligner_t* wf_aligner,
        int pattern_begin_free,
        int pattern_end_free, 
        int text_begin_free,
        int text_end_free)
        
    int wavefront_align(
        wavefront_aligner_t* wf_aligner,
        const char* pattern,
        int pattern_length,
        const char* text,
        int text_length)
        
    # Heuristic configuration functions
    void wavefront_aligner_set_heuristic_none(wavefront_aligner_t* wf_aligner)

    void wavefront_aligner_set_heuristic_wfadaptive(
        wavefront_aligner_t* wf_aligner,
        int min_wavefront_length,
        int max_distance_threshold,
        int steps_between_cutoffs)

    # System configuration functions
    void wavefront_aligner_set_max_alignment_steps(
        wavefront_aligner_t* wf_aligner,
        int max_alignment_steps)
        
    void wavefront_aligner_set_max_memory(
        wavefront_aligner_t* wf_aligner,
        uint64_t max_memory_resident,
        uint64_t max_memory_abort)
        
    
    # CIGAR functions
    char* cigar_sprint_pretty(
        cigar_t* cigar,
        const char* pattern,
        int pattern_length,
        const char* text,
        int text_length,
        char* cigar_buffer)
        
cdef extern from "alignment/cigar.h":
    # CIGAR operations
    ctypedef enum cigar_operation_t:
        cigar_match
        cigar_mismatch
        cigar_ins
        cigar_del
        
    # CIGAR string functions
    int cigar_sprint(char* buffer, const cigar_t* cigar, bint print_matches)
    
    # Score functions for different metrics
    int cigar_score_edit(const cigar_t* cigar)
    int cigar_score_gap_linear(const cigar_t* cigar, const linear_penalties_t* penalties)
    int cigar_score_gap_affine(const cigar_t* cigar, const affine_penalties_t* penalties)
    int cigar_score_gap_affine2p(const cigar_t* cigar, const affine2p_penalties_t* penalties)
