"""
Configuration domain models.

Validated configuration objects for the pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for which tasks to run."""
    
    do_parsing: bool = False
    do_mass_calculating: bool = False
    do_post_processing: bool = False
    do_histogram_creation: bool = False
    
    def any_enabled(self) -> bool:
        """Check if any task is enabled."""
        return any([
            self.do_parsing,
            self.do_mass_calculating,
            self.do_post_processing,
            self.do_histogram_creation
        ])


@dataclass(frozen=True)
class ParsingConfig:
    """Configuration for parsing task."""
    
    # Paths
    output_path: str
    file_urls_path: str
    jobs_logs_path: str
    
    # Processing
    release_years: tuple[str, ...] = field(default_factory=tuple)
    specific_record_ids: Optional[tuple[int, ...]] = None
    parse_mc: bool = False
    
    # Performance
    threads: int = 8
    chunk_yield_threshold_bytes: int = 5_000_000_000  # 5GB default
    env_threshold_memory_mb: int = 20_000  # 20GB default
    
    # Behavior
    create_dirs: bool = False
    show_progress_bar: bool = True
    count_retries_failed_files: int = 3
    fetching_metadata_timeout: int = 60
    
    # Data selection
    possible_data_tree_names: tuple[str, ...] = ("CollectionTree",)
    max_files_to_process: Optional[int] = None  # Limit files (for testing)
    enable_jet_tagging: bool = False
    jet_btagging_thresholds: Optional[dict] = None

    # Optional selection (from YAML): applied after reading each file, before chunking
    particle_counts: Optional[dict] = None
    kinematic_cuts: Optional[dict] = None

    # Physics objects persisted and considered during event selection.
    # Excluded objects are ignored rather than vetoing an event.
    objects_to_store: tuple[str, ...] = (
        "Electrons", "Muons", "Jets", "BJets", "Photons", "Taus"
    )
    # ATLAS-style overlap removal, applied after
    # kinematic cuts. False disables it entirely; True or a dict of cut
    # overrides enables it (see event_selection.DEFAULT_OVERLAP_REMOVAL_CUTS).
    overlap_removal: Optional[dict] = None
    enable_overlap_removal: bool = False

    def __post_init__(self):
        """Validate parsing configuration."""
        if self.threads <= 0:
            raise ValueError(f"threads must be positive, got {self.threads}")
        if self.chunk_yield_threshold_bytes <= 0:
            raise ValueError(f"chunk_yield_threshold_bytes must be positive, got {self.chunk_yield_threshold_bytes}")
        if self.env_threshold_memory_mb <= 0:
            raise ValueError(f"env_threshold_memory_mb must be positive, got {self.env_threshold_memory_mb}")
        if self.count_retries_failed_files < 0:
            raise ValueError(f"count_retries_failed_files must be non-negative, got {self.count_retries_failed_files}")
        if not self.output_path:
            raise ValueError("output_path cannot be empty")
        if not self.file_urls_path:
            raise ValueError("file_urls_path cannot be empty")
        if not self.jobs_logs_path:
            raise ValueError("jobs_logs_path cannot be empty")
        if not isinstance(self.enable_jet_tagging, bool):
            raise ValueError("enable_jet_tagging must be a boolean")
        if self.enable_jet_tagging and not self.jet_btagging_thresholds:
            # TODO add algorithm-specific validation
            raise ValueError("jet_btagging_thresholds must be specified when enable_jet_tagging is set")
        if not self.objects_to_store:
            raise ValueError("objects_to_store cannot be empty")
        if self.particle_counts:
            # Keep this local rather than importing the parsing service into
            # the domain layer.  YAML accepts lower-case plural names.
            yaml_names = {
                "electrons": "Electrons", "muons": "Muons", "jets": "Jets",
                "bjets": "BJets", "photons": "Photons", "taus": "Taus",
            }
            excluded = sorted(
                str(key) for key in self.particle_counts
                if yaml_names.get(str(key).lower(), key) not in self.objects_to_store
            )
            if excluded:
                raise ValueError(
                    "particle_counts may only contain objects listed in "
                    f"objects_to_calculate; excluded keys: {excluded}"
                )


@dataclass(frozen=True)
class MassCalculationConfig:
    """Configuration for mass calculation task."""
    
    # Paths
    input_dir: str
    output_dir: str
    
    # Processing
    field_to_slice_by: str = "pt"  # Which kinematic field to rank/slice particles by
    use_multiprocessing: bool = True
    parallel_processes: int = 4
    fs_chunk_threshold_bytes: int = 500_000_000  # 500MB - chunk threshold for final-state files
    
    # Combinatorics configuration
    objects_to_calculate: tuple[str, ...] = (
        "Electrons", "Muons", "Jets", "BJets", "Photons", "Taus"
    )
    min_particles_in_combination: int = 1
    max_particles_in_combination: int = 4
    min_count_particle_in_combination: int = 2
    max_count_particle_in_combination: int = 4
    max_total_particles_in_combination: int = 4
    min_events_per_fs: int = 100  # Minimum events for a final state to be kept
    include_subleading: bool = True        # include e1, j1, etc. by default
    max_subleading_index: int = 1          # highest rank index to consider

    # Per-event mass resolution sigma_m (stored as the _sigm sibling) from a
    # public object-resolution table; None table = the packaged default.
    mass_resolution_enabled: bool = False
    mass_resolution_table: Optional[str] = None
    mass_resolution_apply_data_mc_scale: bool = True
    
    def __post_init__(self):
        """Validate mass calculation configuration."""
        if not self.input_dir:
            raise ValueError("input_dir cannot be empty")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        if self.parallel_processes <= 0:
            raise ValueError(f"parallel_processes must be positive, got {self.parallel_processes}")
        if self.min_particles_in_combination < 1:
            raise ValueError(f"min_particles_in_combination must be >= 1, got {self.min_particles_in_combination}")
        if self.max_particles_in_combination < self.min_particles_in_combination:
            raise ValueError(
                f"max_particles_in_combination ({self.max_particles_in_combination}) must be >= "
                f"min_particles_in_combination ({self.min_particles_in_combination})"
            )
        if self.min_count_particle_in_combination < 1:
            raise ValueError(f"min_count_particle_in_combination must be >= 1, got {self.min_count_particle_in_combination}")
        if self.max_count_particle_in_combination < self.min_count_particle_in_combination:
            raise ValueError(
                f"max_count_particle_in_combination ({self.max_count_particle_in_combination}) must be >= "
                f"min_count_particle_in_combination ({self.min_count_particle_in_combination})"
            )
        if self.fs_chunk_threshold_bytes <= 0:
            raise ValueError(f"fs_chunk_threshold_bytes must be positive, got {self.fs_chunk_threshold_bytes}")
        if not self.objects_to_calculate:
            raise ValueError("objects_to_calculate cannot be empty")
        if self.min_events_per_fs < 0:
            raise ValueError(f"min_events_per_fs must be non-negative, got {self.min_events_per_fs}")


@dataclass(frozen=True)
class PostProcessingConfig:
    """Configuration for post-processing task."""
    
    # Paths
    input_dir: str
    output_dir: str
    
    # Processing parameters
    peak_detection_bin_width_gev: float = 10.0

    z_peak_cutoff: float = 110.0
    max_mass_cutoff: float = 10_000.0
    # Drop masses below the rightmost-highest peak of each spectrum.
    enable_peak_removal: bool = True

    def __post_init__(self):
        """Validate post-processing configuration."""
        if not self.input_dir:
            raise ValueError("input_dir cannot be empty")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        if self.peak_detection_bin_width_gev <= 0:
            raise ValueError(f"peak_detection_bin_width_gev must be positive, got {self.peak_detection_bin_width_gev}")
        if self.z_peak_cutoff < 0:
            raise ValueError(f"z_peak_cutoff must be non-negative, got {self.z_peak_cutoff}")
        if self.max_mass_cutoff < 0:
            raise ValueError(f"max_mass_cutoff must be non-negative, got {self.max_mass_cutoff}")
        if not isinstance(self.enable_peak_removal, bool):
            raise ValueError("enable_peak_removal must be a boolean")


@dataclass(frozen=True)
class HistogramCreationConfig:
    """Configuration for histogram creation task."""
    
    # Paths
    input_dir: str
    output_dir: str
    
    # Processing parameters (one width or a list of widths)
    bin_width_gev: Union[float, Tuple[float, ...]] = 10.0
    single_output_file: bool = False
    output_filename: Optional[str] = None
    
    # Outlier and naming
    exclude_outliers: bool = True  # Whether to exclude outlier histograms
    use_bumpnet_naming: bool = False  # When true, use mass_<combo>_cat_<final_state> naming
    apply_peak_removal_at_histogram_level: bool = False
    # Slice each final histogram to its first..last non-empty bin (after hadd in batch runs)
    trim_empty_bins: bool = False
    
    # Pre-postproc histograms
    also_save_pre_postproc: bool = False
    pre_postproc_filename: Optional[str] = None

    # Inputs of the resolution rebinning: RESO_ profile of the per-event
    # sigma_m (None = not filled) and PROC_<group>_ histograms per process
    # group ({group: [DSID, ...]}; MC only).
    resolution_profile_bin_width_gev: Optional[float] = None
    process_groups: Optional[dict] = None

    def __post_init__(self):
        """Validate histogram creation configuration."""
        if not self.input_dir:
            raise ValueError("input_dir cannot be empty")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        widths = self.bin_width_gev if isinstance(self.bin_width_gev, tuple) else (self.bin_width_gev,)
        if not widths or any(w <= 0 for w in widths):
            raise ValueError(f"bin_width_gev must be positive, got {self.bin_width_gev}")
        if self.resolution_profile_bin_width_gev is not None and self.resolution_profile_bin_width_gev <= 0:
            raise ValueError("resolution_profile_bin_width_gev must be positive")
        if self.single_output_file and not self.output_filename:
            raise ValueError("output_filename required when single_output_file=True")


@dataclass(frozen=True)
class RebinConfig:
    """
    Post-merge resolution rebinning (services/pipelines/rebin_pipeline.py).

    ``fit`` derives the binning from the RESO_ profiles and saves it under
    ``binning_dir``; ``cached`` reuses a saved binning (e.g. data with the
    binning of the MC run). ``binning_dir`` None = <run_dir>/rebin.
    """

    enabled: bool = False
    mode: str = "fit"
    base_bin_width_gev: float = 1.0
    binning_dir: Optional[str] = None
    output_filename: str = "rebinned.root"
    rho: float = 1.0
    gpr_alpha_floor: float = 1.0
    min_profile_entries: float = 5.0
    z_veto_edge_gev: float = 106.18
    top_roi_start_gev: float = 175.0
    ztautau_cutoff_gev: float = 80.0
    min_tail_events: float = 1.0   # binning ends where < this many expected bkg events lie above

    def __post_init__(self):
        if self.mode not in ("fit", "cached"):
            raise ValueError(f"rebin_config.mode must be 'fit' or 'cached', got {self.mode!r}")
        if self.base_bin_width_gev <= 0 or self.rho <= 0:
            raise ValueError("rebin_config.base_bin_width_gev and rho must be positive")
        if self.enabled and self.mode == "cached" and not self.binning_dir:
            raise ValueError("rebin_config.mode 'cached' needs binning_dir (the MC run's rebin dir)")


@dataclass(frozen=True)
class MCWeightingConfig:
    """
    Configuration for Monte-Carlo event weighting.

    Holds the analysis-level constants that are NOT part of the per-dataset
    metadata: the target integrated luminosity the simulated samples are
    normalized to (which can vary by MC campaign). The physics inputs (cross section, k-factor, filter efficiency, sum of
    weights) come from the dataset metadata, not here.
    """

    enabled: bool = False
    # Target luminosity in fb^-1. When enabled, exactly one of this and
    # luminosity_by_campaign must be set.
    target_luminosity_fb: Optional[float] = None
    # Per-campaign luminosity, e.g. {"mc20a": 36.2, "mc20d": 44.3, "mc20e": 58.5}.
    # Mutually exclusive with target_luminosity_fb.
    luminosity_by_campaign: Optional[dict] = None
    # Multiply each event by its nominal pileup weight (PHYSLITE
    # PileupWeight_NOSYS). The normalization (sumOfWeights) is unchanged.
    use_pileup_weight: bool = False

    def __post_init__(self):
        """Validate MC weighting configuration."""
        if self.luminosity_by_campaign:
            if self.target_luminosity_fb is not None:
                raise ValueError(
                    "target_luminosity_fb and luminosity_by_campaign are mutually exclusive; set only one"
                )
        elif self.enabled and self.target_luminosity_fb is None:
            raise ValueError(
                "MC weighting is enabled but neither target_luminosity_fb nor "
                "luminosity_by_campaign is set"
            )
        if self.target_luminosity_fb is not None and self.target_luminosity_fb < 0:
            raise ValueError(
                f"target_luminosity_fb must be non-negative, got {self.target_luminosity_fb}"
            )
        if self.luminosity_by_campaign:
            for campaign, lumi in self.luminosity_by_campaign.items():
                if lumi < 0:
                    raise ValueError(
                        f"luminosity for campaign '{campaign}' must be non-negative, got {lumi}"
                    )

    def get_luminosity(self, campaign: Optional[str] = None) -> float:
        """
        Resolve the target luminosity for a given MC campaign.

        Uses ``luminosity_by_campaign`` when configured, else ``target_luminosity_fb``.

        Raises:
            ValueError: when ``luminosity_by_campaign`` is configured and has no
                entry for ``campaign``.
        """
        if self.luminosity_by_campaign:
            if campaign not in self.luminosity_by_campaign:
                raise ValueError(f"luminosity_by_campaign has no entry for campaign {campaign!r}")
            return self.luminosity_by_campaign[campaign]
        return self.target_luminosity_fb


@dataclass(frozen=True)
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    Immutable configuration object validated at creation.
    """
    
    # Task configuration
    tasks: TaskConfig
    
    # Stage configurations
    parsing_config: Optional[ParsingConfig] = None
    mass_calculation_config: Optional[MassCalculationConfig] = None
    post_processing_config: Optional[PostProcessingConfig] = None
    histogram_creation_config: Optional[HistogramCreationConfig] = None

    # Cross-cutting configuration
    trigger_config: Optional[dict] = None
    mc_weighting_config: Optional[MCWeightingConfig] = None
    rebin_config: Optional[RebinConfig] = None
    
    # Run metadata
    run_name: str = "pipeline_run"
    batch_job_index: Optional[int] = None
    total_batch_jobs: Optional[int] = None
    
    def __post_init__(self):
        """Validate pipeline configuration."""
        if not self.tasks.any_enabled():
            raise ValueError("At least one task must be enabled")
        
        if self.tasks.do_parsing and not self.parsing_config:
            raise ValueError("parsing_config required when do_parsing=True")
        
        if self.tasks.do_mass_calculating and not self.mass_calculation_config:
            raise ValueError("mass_calculation_config required when do_mass_calculating=True")
        
        if self.tasks.do_post_processing and not self.post_processing_config:
            raise ValueError("post_processing_config required when do_post_processing=True")
        
        if self.tasks.do_histogram_creation and not self.histogram_creation_config:
            raise ValueError("histogram_creation_config required when do_histogram_creation=True")
        
        if self.batch_job_index is not None:
            if self.batch_job_index < 1:
                raise ValueError(f"batch_job_index must be >= 1 (1-indexed), got {self.batch_job_index}")
            if self.total_batch_jobs is None:
                raise ValueError("total_batch_jobs required when batch_job_index is set")
            if self.total_batch_jobs < 1:
                raise ValueError(f"total_batch_jobs must be >= 1, got {self.total_batch_jobs}")
            if self.batch_job_index > self.total_batch_jobs:
                raise ValueError(
                    f"batch_job_index ({self.batch_job_index}) must be <= "
                    f"total_batch_jobs ({self.total_batch_jobs})"
                )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PipelineConfig':
        """
        Create PipelineConfig from a dictionary (e.g., loaded from YAML).
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            Validated PipelineConfig instance
        """
        # Parse task config
        tasks_dict = config_dict.get("tasks", {})
        tasks = TaskConfig(
            do_parsing=tasks_dict.get("do_parsing", False),
            do_mass_calculating=tasks_dict.get("do_mass_calculating", False),
            do_post_processing=tasks_dict.get("do_post_processing", False),
            do_histogram_creation=tasks_dict.get("do_histogram_creation", False),
        )
        
        # Resolve this once because parsing must use the same allow-list as
        # invariant-mass calculation, even in parsing-only runs.
        mass_dict = config_dict.get("mass_calculation_task_config", {})
        objects_raw = mass_dict.get("objects_to_calculate")
        objects = tuple(objects_raw) if objects_raw else (
            "Electrons", "Muons", "Jets", "BJets", "Photons", "Taus"
        )

        # Parse parsing config if parsing is enabled
        parsing_config = None
        if tasks.do_parsing:
            parsing_dict = config_dict.get("parsing_task_config", {})
            
            # Handle release_years (can be None, list, or empty)
            release_years_raw = parsing_dict.get("release_years")
            release_years = tuple(release_years_raw) if release_years_raw else ()
            
            parsing_config = ParsingConfig(
                output_path=parsing_dict["output_path"],
                file_urls_path=parsing_dict["file_urls_path"],
                jobs_logs_path=parsing_dict["jobs_logs_path"],
                release_years=release_years,
                specific_record_ids=tuple(parsing_dict["specific_record_ids"]) if parsing_dict.get("specific_record_ids") else None,
                parse_mc=parsing_dict.get("parse_mc", False),
                threads=parsing_dict.get("threads", 8),
                chunk_yield_threshold_bytes=parsing_dict.get("chunk_yield_threshold_bytes", 5_000_000_000),
                env_threshold_memory_mb=parsing_dict.get("env_threshold_memory_mb", 20_000),
                create_dirs=parsing_dict.get("create_dirs", False),
                show_progress_bar=parsing_dict.get("show_progress_bar", True),
                count_retries_failed_files=parsing_dict.get("count_retries_failed_files", 3),
                fetching_metadata_timeout=parsing_dict.get("fetching_metadata_timeout", 60),
                possible_data_tree_names=tuple(parsing_dict.get("possible_data_tree_names", ["CollectionTree"])),
                max_files_to_process=parsing_dict.get("max_files_to_process"),
                enable_jet_tagging=parsing_dict.get("enable_jet_tagging", False),
                jet_btagging_thresholds=parsing_dict.get("jet_btagging_thresholds", None),
                particle_counts=parsing_dict.get("particle_counts"),
                kinematic_cuts=parsing_dict.get("kinematic_cuts"),
                objects_to_store=objects,
                enable_overlap_removal=parsing_dict.get("enable_overlap_removal", False),
                overlap_removal=parsing_dict.get("overlap_removal"),
            )
        
        # Parse mass calculation config if enabled
        mass_calculation_config = None
        if tasks.do_mass_calculating or tasks.do_post_processing:
            mass_calculation_config = MassCalculationConfig(
                input_dir=mass_dict["input_dir"],
                output_dir=mass_dict["output_dir"],
                field_to_slice_by=mass_dict.get("field_to_slice_by", "pt"),
                use_multiprocessing=mass_dict.get("use_multiprocessing", True),
                parallel_processes=mass_dict.get("parallel_processes", 4),
                fs_chunk_threshold_bytes=mass_dict.get("fs_chunk_threshold_bytes", 500_000_000),
                objects_to_calculate=objects,
                min_particles_in_combination=mass_dict.get("min_particles_in_combination", 1),
                max_particles_in_combination=mass_dict.get("max_particles_in_combination", 4),
                min_count_particle_in_combination=mass_dict.get("min_count_particle_in_combination", 2),
                max_count_particle_in_combination=mass_dict.get("max_count_particle_in_combination", 4),
                max_total_particles_in_combination=mass_dict.get("max_total_particles_in_combination", 4),
                min_events_per_fs=mass_dict.get("min_events_per_fs", 100),
                include_subleading=mass_dict.get("include_subleading", True),
                max_subleading_index=mass_dict.get("max_subleading_index", 1),
                mass_resolution_enabled=bool((mass_dict.get("mass_resolution") or {}).get("enabled", False)),
                mass_resolution_table=(mass_dict.get("mass_resolution") or {}).get("table"),
                mass_resolution_apply_data_mc_scale=bool(
                    (mass_dict.get("mass_resolution") or {}).get("apply_data_mc_scale", True)
                ),
            )
        
        # Parse post-processing config if enabled
        post_processing_config = None
        if tasks.do_post_processing:
            post_dict = config_dict.get("post_processing_task_config", {})
            post_processing_config = PostProcessingConfig(
                input_dir=post_dict["input_dir"],
                output_dir=post_dict["output_dir"],
                peak_detection_bin_width_gev=post_dict.get("peak_detection_bin_width_gev", 10.0),
                z_peak_cutoff=post_dict.get("z_peak_cutoff", 110.0),
                max_mass_cutoff=post_dict.get("max_mass_cutoff", 10_000.0),
                enable_peak_removal=post_dict.get("enable_peak_removal", True),
            )
        
        # Parse histogram creation config if enabled
        histogram_creation_config = None
        if tasks.do_histogram_creation:
            hist_dict = config_dict.get("histogram_creation_task_config", {})
            bin_width = hist_dict.get("bin_width_gev", 10.0)
            histogram_creation_config = HistogramCreationConfig(
                input_dir=hist_dict["input_dir"],
                output_dir=hist_dict["output_dir"],
                bin_width_gev=tuple(bin_width) if isinstance(bin_width, (list, tuple)) else bin_width,
                single_output_file=hist_dict.get("single_output_file", False),
                output_filename=hist_dict.get("output_filename"),
                exclude_outliers=hist_dict.get("exclude_outliers", True),
                use_bumpnet_naming=hist_dict.get("use_bumpnet_naming", False),
                apply_peak_removal_at_histogram_level=hist_dict.get(
                    "apply_peak_removal_at_histogram_level", False
                ),
                trim_empty_bins=hist_dict.get("trim_empty_bins", False),
                also_save_pre_postproc=hist_dict.get("also_save_pre_postproc", False),
                pre_postproc_filename=hist_dict.get("pre_postproc_filename"),
                resolution_profile_bin_width_gev=hist_dict.get("resolution_profile_bin_width_gev"),
                process_groups=hist_dict.get("process_groups"),
            )
        
        # Parse MC weighting config (optional cross-cutting section)
        mc_weighting_config = None
        mc_dict = config_dict.get("mc_weighting_config")
        if mc_dict:
            mc_weighting_config = MCWeightingConfig(
                enabled=mc_dict.get("enabled", False),
                target_luminosity_fb=mc_dict.get("target_luminosity_fb"),
                luminosity_by_campaign=mc_dict.get("luminosity_by_campaign"),
                use_pileup_weight=bool(mc_dict.get("use_pileup_weight", False)),
            )
            # MC weighting normalizes simulated samples only.
            if mc_weighting_config.enabled and not (config_dict.get("parsing_task_config") or {}).get("parse_mc", False):
                raise ValueError("mc_weighting_config.enabled requires parsing_task_config.parse_mc: true")

        rebin_config = None
        rebin_dict = config_dict.get("rebin_config")
        if rebin_dict:
            rebin_config = RebinConfig(**rebin_dict)
        # The binning is decided by the background run alone (mode "fit"). A run
        # that reuses it (mode "cached": signal, data) only fills those bins, so
        # its own events must not be cut by the main/outlier split, which depends
        # on where gaps fall in that run's own spectrum.
        if (rebin_config and rebin_config.enabled and rebin_config.mode == "cached"
                and histogram_creation_config and histogram_creation_config.exclude_outliers):
            import logging
            from dataclasses import replace
            histogram_creation_config = replace(histogram_creation_config, exclude_outliers=False)
            logging.getLogger(__name__).info(
                "rebin_config.mode=cached: exclude_outliers set to false; this run follows the "
                "background binning and keeps all its events")

        # Parse run metadata
        run_metadata = config_dict.get("run_metadata", {})
        
        return cls(
            tasks=tasks,
            parsing_config=parsing_config,
            mass_calculation_config=mass_calculation_config,
            post_processing_config=post_processing_config,
            histogram_creation_config=histogram_creation_config,
            trigger_config=config_dict.get("trigger_config"),
            mc_weighting_config=mc_weighting_config,
            rebin_config=rebin_config,
            run_name=run_metadata.get("run_name", "pipeline_run"),
            batch_job_index=run_metadata.get("batch_job_index"),
            total_batch_jobs=run_metadata.get("total_batch_jobs"),
        )