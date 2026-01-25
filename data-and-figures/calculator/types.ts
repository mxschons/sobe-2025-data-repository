/**
 * AUTO-GENERATED FILE - DO NOT EDIT
 * Generated from TSV files by scripts/generate-types.ts
 * Run 'npm run generate:types' to regenerate
 */

/**
 * Shared project-level parameters
 */
export interface SharedParameters {
  /** Physical brain volume to be mapped (mm³) */
  biologicalVolume: number;
  /** Number of neurons in the given volume (count) */
  neuronCount: number;
  /** Budget and schedule contingency fraction (fraction) */
  riskBufferFirst: number;
  /** Budget and schedule contingency fraction (fraction) */
  riskBufferMarginal: number;
  /** Years from now when initial purchases are made (years) */
  yearsUntilStart: number;
  /** Time span for overall project (years) */
  projectDuration: number;
  /** Upfront investment in microscopes ($) */
  microscopeBudget: number;
  /** Number of GPUs dedicated during processing (count) */
  maxParallelGpus: number;
  /** Cleanroom, isolation, HVAC, racks ($/year) */
  facilityCostPerYear: number;
  /** Average salary with all overheads ($/year) */
  avgStaffSalary: number;
  /** Staff for project oversight (FTE) */
  projectMgmtStaff: number;
  /** Beyond sample and microscope operations (FTE) */
  technicalStaff: number;
  /** Operations, assistants, PR, etc (FTE) */
  miscStaff: number;
  /** Shipping, permits, biosafety, IP, pubs ($) */
  otherCostsPerConnectome: number;
  /** Pipelines, data mgmt, viewers, model training ($) */
  dataScienceCost: number;
  /** Scopes + facility + core SW + initial training ($) */
  capitalBase: number;
  /** Maximum FP16 FLOP/s for GPU (TFLOPs/s) */
  peakFlopsGpu: number;
  /** Continuous GPU usage across time (fraction) */
  gpuUtilization: number;
  /** Costs for access to 1h of GPU ($/hour) */
  gpuCostPerHour: number;
  /** Cost drop compute per year (fraction) */
  costDropComputePerYear: number;
  /** Cost drop storage per year (fraction) */
  costDropStoragePerYear: number;
  /** Storage for working files ($/PB-month) */
  activeStorageCostPbMonth: number;
  /** Storage for backup files ($/PB-month) */
  archiveStorageCostPbMonth: number;
  /** Assuming 8-bits per voxel (bytes) */
  bytesPerVoxel: number;
  /** Raw to lossless ratio (×) */
  losslessCompression: number;
  /** Raw to lossy ratio (×) */
  lossyCompression: number;
  /** Labels + meshes + skeletons + graphs (fraction) */
  labelOverhead: number;
  /** Hot copies lossless online first (count) */
  replicasActiveFirst: number;
  /** Cold copies retained first (count) */
  replicasArchiveFirst: number;
  /** Hot copies lossless online marginal (count) */
  replicasActiveMarginal: number;
  /** Cold copies retained marginal (count) */
  replicasArchiveMarginal: number;
  /** How long active copies are kept (years) */
  activeRetentionYears: number;
  /** How long archive copies are kept (years) */
  archiveRetentionYears: number;
  /** Online storage price ($/PB-year) */
  activeCostPbYear: number;
  /** Archive storage price ($/PB-year) */
  archiveCostPbYear: number;
  /** FP32-equivalent per tile (TFLOP/tile) */
  flopsRegistrationPerTile: number;
  /** FP32-equivalent per voxel (TFLOP/voxel) */
  segmentationFlopsPerVoxel: number;
  /** PV processed per day per GPU (PV/day/GPU) */
  registrationRatePvPerDay: number;
  /** PV processed per day per GPU (PV/day/GPU) */
  segmentationRatePvPerDay: number;
}

/**
 * Imaging modality parameter values
 */
export interface ImagingModalityParams {
  /** Acquisition price per scope ($/scope) */
  microscopeCapitalCost: number;
  /** Straight-line depreciation years (years) */
  microscopeDepreciationYears: number;
  /** Imaging rate at given resolution per microscope (Mvox/s) */
  sustainedImagingRate: number;
  /** Simultaneous channel readouts (count) */
  parallelChannels: number;
  /** Imaging channels per voxel (count) */
  totalChannels: number;
  /** Electricity, licenses, maintenance ($/scope/year) */
  scopeAnnualService: number;
  /** Total salary incl benefits for microscopist ($/year) */
  technicianSalary: number;
  /** Microscopes per technician (count) */
  microscopeTechnicianRatio: number;
  /** Fraction of calendar time producing good data (fraction) */
  microscopeUptime: number;
  /** Expected fraction of volume to re-image (fraction) */
  reacquisitionRate: number;
  /** Fraction of samples that pass QC (fraction) */
  sampleYield: number;
  /** Linear expansion factor E; volume scales as E³ (×) */
  expansionFactor: number;
  /** Sample prep, expanding, staining per tissue volume ($/mm³) */
  consumablesPerMm3: number;
  /** Antibody costs per original tissue ($/mm³) */
  antibodyCostPerMm3: number;
  /** Human labor costs per original mm³ ($/mm³) */
  laborCostPerMm3: number;
  /** From tissue to ready-to-image (days) */
  initialPrepDays: number;
  /** Acquisition resolution X (nm) */
  voxelX: number;
  /** Acquisition resolution Y (nm) */
  voxelY: number;
  /** Acquisition resolution Z (nm) */
  voxelZ: number;
  /** Layers of Z in one sample (count) */
  sampleDepth: number;
  /** Lateral overlap for stitching (fraction) */
  tileOverlap: number;
}

/**
 * Available imaging modality IDs
 */
export type ModalityId = 'em' | 'exm' | 'exm_molecular' | 'wellcome';

/**
 * All imaging modalities indexed by ID
 */
export type ImagingModalities = Record<ModalityId, ImagingModalityParams>;

/**
 * Organism specifications
 */
export interface Organism {
  id: string;
  name: string;
  neurons: number;
  volumeMm3: number;
  synapses: number;
  source: string;
}

/**
 * Available organism IDs
 */
export type OrganismId = 'c_elegans' | 'drosophila' | 'zebrafish_larva' | 'mouse' | 'macaque' | 'human';

/**
 * Proofreading parameters
 */
export interface ProofreadingParams {
  /** Human proofreading hours per neuron (hours) */
  hoursPerNeuron: number;
  /** Hourly rate of proofreader ($/hour) */
  hourlyRate: number;
  /** Proofreading hours per day (hours/day) */
  hoursPerDay: number;
  /** Number of proofreaders (count) */
  numProofreaders: number;
}

/**
 * Proofreading scenario
 */
export type ProofreadingScenario = 'current' | 'improved_1000x';

/**
 * All calculated formula results
 */
export interface CalculationResults {
  /** Number of voxels in one cubic millimeter at acquisition resolution (voxels/mm³) */
  voxelsPerMm3: number;
  /** Total volume to image after tissue expansion (mm³) */
  effectiveVolume: number;
  /** Petavoxels per cubic millimeter (PV/mm³) */
  petavoxelsPerMm3: number;
  /** Total petavoxels to image (PV) */
  petavoxelsTotal: number;
  /** Effective imaging speed accounting for channel rounds and uptime (Mvox/s) */
  netImagingRate: number;
  /** Volume one microscope can image per day (mm³/day) */
  mm3PerDayPerScope: number;
  /** Number of microscopes affordable within budget (count) */
  numMicroscopes: number;
  /** Annual operating cost per microscope ($/year) */
  scopeOperatingCostPerYear: number;
  /** Total days to complete all imaging (days) */
  imagingDays: number;
  /** Imaging time in years (years) */
  imagingYears: number;
  /** Days for image registration/alignment (days) */
  registrationDays: number;
  /** Days for neural segmentation (days) */
  segmentationDays: number;
  /** Total GPU processing time (days) */
  processingDays: number;
  /** Processing time in years (years) */
  processingYears: number;
  /** Total human proofreading time (days) */
  proofreadingDays: number;
  /** Proofreading time in years (years) */
  proofreadingYears: number;
  /** Risk buffer time for first connectome (years) */
  bufferYears: number;
  /** Total years until first connectome complete (years) */
  timeToFirstYears: number;
  /** Years between subsequent connectomes (bottleneck) (years) */
  timeToMarginalYears: number;
  /** Number of connectomes achievable in project duration (count) */
  totalConnectomes: number;
  /** Sample preparation and consumables cost ($) */
  consumablesCost: number;
  /** Microscope operating costs during imaging ($) */
  scanningCost: number;
  /** Total imaging costs ($) */
  imagingCostTotal: number;
  /** Registration compute cost per petavoxel ($/PV) */
  processingCostRegistration: number;
  /** Segmentation compute cost per petavoxel ($/PV) */
  processingCostSegmentation: number;
  /** Total processing compute cost ($) */
  processingCostTotal: number;
  /** Total proofreading labor cost ($) */
  proofreadingCost: number;
  /** Personnel costs per connectome ($) */
  personnelCost: number;
  /** Miscellaneous costs (shipping, permits, etc.) ($) */
  otherCosts: number;
  /** First connectome costs before buffer ($) */
  firstSubtotal: number;
  /** Risk buffer for first connectome ($) */
  firstBuffer: number;
  /** Total cost for first connectome ($) */
  firstTotal: number;
  /** First connectome cost per neuron ($/neuron) */
  firstCostPerNeuron: number;
  /** Marginal connectome costs before buffer ($) */
  marginalSubtotal: number;
  /** Risk buffer for marginal connectomes ($) */
  marginalBuffer: number;
  /** Total cost for marginal connectome ($) */
  marginalTotal: number;
  /** Marginal connectome cost per neuron ($/neuron) */
  marginalCostPerNeuron: number;
  /** Average cost per connectome ($) */
  avgTotal: number;
  /** Average cost per neuron across all connectomes ($/neuron) */
  avgCostPerNeuron: number;
  /** Uncompressed bytes per cubic millimeter (bytes/mm³) */
  rawBytesPerMm3: number;
  /** Total raw uncompressed data size (PB) */
  rawPbTotal: number;
  /** Active storage size (lossy compressed + labels) (PB) */
  activePb: number;
  /** Archive storage size (lossless compressed + labels) (PB) */
  archivePb: number;
  /** Active storage cost for first connectome ($) */
  activeStorageCostFirst: number;
  /** Archive storage cost for first connectome ($) */
  archiveStorageCostFirst: number;
  /** Total storage cost for first connectome ($) */
  totalStorageCostFirst: number;
  /** Active storage cost for marginal connectomes ($) */
  activeStorageCostMarginal: number;
  /** Archive storage cost for marginal connectomes ($) */
  archiveStorageCostMarginal: number;
  /** Total storage cost for marginal connectomes ($) */
  totalStorageCostMarginal: number;
}

/**
 * Formula definition metadata
 */
export interface FormulaDefinition {
  id: string;
  formula: string;
  inputs: string[];
  unit: string;
  section: string;
  row: number;
  description: string;
}

/**
 * Input parameters for the calculator
 */
export interface CalculatorInput {
  organism: OrganismId;
  modality: ModalityId;
  proofreadingScenario: ProofreadingScenario;
  overrides?: Partial<SharedParameters & ImagingModalityParams & ProofreadingParams>;
}

/**
 * Full calculator output
 */
export interface CalculatorOutput {
  inputs: Record<string, number>;
  results: CalculationResults;
  metadata: {
    organism: Organism;
    modality: ModalityId;
    proofreadingScenario: ProofreadingScenario;
  };
}
