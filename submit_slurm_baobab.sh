#!/bin/bash
# ===========================================================================
# Slurm Job Submission Script for ATLAS Pipeline (Baobab, UNIGE)
# (ported from the Green Planet / UCI version)
#
# 4-phase chain, mirroring upstream submit_mc.sh (Slurm instead of PBS):
#
#   1. batch array   : each array task runs parsing + mass_calculating on its
#                       own slice of files, capped at FILES_PER_JOB files --
#                       a FRESH process per task -> im_arrays/im_batch_N.sqlite
#   2. post-process  : single job, after ALL batch tasks succeed. It must see
#                       every IM shard: final states below min_events_per_fs
#                       are pruned on the GLOBAL count across all shards.
#                       -> im_arrays_processed/processed_batch_1.sqlite
#   3. histograms    : single job (there is one processed shard)
#                       -> histograms/batch_1.root
#   4. merge         : hadd histograms/batch_*.root, aggregate stats, plots
#
# Histogram ranges are fixed ([0, 10] TeV), so there is no scan step anymore.
#
# WHY an array instead of one big job: a single long-lived process was
# observed to accumulate memory well beyond what EventAccumulator tracks
# (chunk_yield_threshold_bytes) and eventually gets OOM-killed by Slurm --
# reproduced twice, at file ~220/5000 (8 threads) and file ~382/500
# (4 threads). See CLAUDE.md "env_threshold_memory_mb is dead config" for the
# full writeup. Splitting work across many independent array tasks sidesteps
# this entirely: no single process ever accumulates enough history to
# approach the ~23.5GB physical memory of an `atlas`-partition node.
#
# Usage: bash submit_slurm_baobab.sh   (Baobab/UNIGE copy of submit_slurm.sh)
# ===========================================================================

set -e

# --- Configuration ---------------------------------------------------------
CONFIG="${CONFIG:-config.baobab_mc.yaml}"   # Baobab MC config (see header)

# Both can be overridden from the environment, e.g.
#   TOTAL_FILES=2 FILES_PER_JOB=1 bash submit_slurm_baobab.sh     (smoke test)
# NOTE: the full file list (2024r-pp MC: ~14.9k files / 625 DSIDs) is split
# into NUM_JOBS equal slices and each job only takes the FIRST FILES_PER_JOB
# files of its slice. If TOTAL_FILES < real file count, the rest is skipped.
# For full MC coverage: TOTAL_FILES=14855 (-> ~298 array tasks at 50 files/job).
TOTAL_FILES=${TOTAL_FILES:-100}     # total files to process across the whole run
FILES_PER_JOB=${FILES_PER_JOB:-50}  # per-job file cap -- comfortably under the ~220-382
                         # file range where single long-running jobs OOM'd.
                         # Lower this further if a batch task still OOMs.
NUM_JOBS=$(( (TOTAL_FILES + FILES_PER_JOB - 1) / FILES_PER_JOB ))  # ceil division
MAX_CONCURRENT=${MAX_CONCURRENT:-20}  # throttle: at most this many array tasks run at once
                         # (each task streams files from eospublic -- don't hammer it)

CPUS_PER_JOB=4
BATCH_MEM="16G"
BATCH_WALLTIME="12:00:00"     # shared-cpu max is 12h
POSTPROC_WALLTIME="04:00:00"
POSTPROC_MEM="32G"
HIST_WALLTIME="04:00:00"
MERGE_WALLTIME="02:00:00"
PARTITION="${PARTITION:-shared-cpu}"

# Slurm account. Blank = user's default association (fine on Baobab).
ACCOUNT=""

# Comma-separated node names to exclude, if a node misbehaves.
EXCLUDE_NODES=""

# Extra python packages missing from the LCG view (atlasopenmagic) are
# installed into this dedicated user base -- NOT ~/.local, which would leak
# into the conda python3.11 envs. One-time setup (on a compute node):
#   srun -p shared-cpu -t 00:20:00 bash -c 'source <LCG setup>; \
#     PYTHONUSERBASE=$HOME/resonex/pyuser pip install --user atlasopenmagic==1.5.0'
PY_USERBASE="${PY_USERBASE:-$HOME/resonex/pyuser}"

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

ACCOUNT_DIRECTIVE=""
if [ -n "$ACCOUNT" ]; then
    ACCOUNT_DIRECTIVE="#SBATCH --account=${ACCOUNT}"
fi

EXCLUDE_DIRECTIVE=""
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_DIRECTIVE="#SBATCH --exclude=${EXCLUDE_NODES}"
fi

# --- Read run_name and base_output_dir from config -------------------------
# Plain grep/sed (no python on the login node). run_name is a YAML anchor
# defined in the `names:` block; base_output_dir sits in `run_metadata:`.
RUN_NAME=$(grep -E '^[[:space:]]*run_name:[[:space:]]*&run_name' "${CONFIG}" | sed -E 's/.*&run_name[[:space:]]+"?([^"#[:space:]]+)"?.*/\1/')
BASE_DIR=$(grep -E '^[[:space:]]*base_output_dir:' "${CONFIG}" | head -1 | sed -E 's/^[[:space:]]*base_output_dir:[[:space:]]*"?([^"#[:space:]]+)"?.*/\1/')
RUN_NAME=${RUN_NAME:-pipeline_run}
BASE_DIR=${BASE_DIR:-./output}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BASE_DIR}/${RUN_NAME}_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

cp "${PIPELINE_DIR}/${CONFIG}" "${LOG_DIR}/config.yaml.orig"
cp "${PIPELINE_DIR}/submit_slurm_baobab.sh" "${LOG_DIR}/"

# --- Per-run config copy with max_files_to_process overridden ---------------
# main.py has no --max-files-to-process CLI flag, and this script must not
# silently rewrite the user's actual config.yaml (would destroy its comments
# on every run). Instead: copy it once per run and surgically patch just
# that one line with sed, preserving everything else -- this copy (not the
# original config.yaml) is what every array task/phase below actually reads.
RUN_CONFIG="${LOG_DIR}/config_run.yaml"
cp "${PIPELINE_DIR}/${CONFIG}" "${RUN_CONFIG}"
sed -i -E "s/^([[:space:]]*max_files_to_process:)[[:space:]]*.*/\1 ${FILES_PER_JOB}  # overridden by submit_slurm.sh (per-job cap for batch array)/" "${RUN_CONFIG}"

echo "=============================================="
echo "ATLAS Pipeline Submission (Slurm / Baobab)"
echo "=============================================="
echo "Config:        ${CONFIG} (per-job override: ${RUN_CONFIG})"
echo "Run dir:       ${RUN_DIR}"
echo "Total files:   ${TOTAL_FILES}"
echo "Files/job:     ${FILES_PER_JOB}"
echo "Array size:    ${NUM_JOBS} (max ${MAX_CONCURRENT} concurrent)"
echo "Partition:     ${PARTITION}"
echo "Account:       ${ACCOUNT:-(none set - omitting --account)}"
echo "CPUs/job:      ${CPUS_PER_JOB}"
echo "Mem/batch job: ${BATCH_MEM}"
echo "=============================================="

# --- Phase 1: batch array — parsing + mass_calculating ----------------------
BATCH_JOB_ID=$(sbatch --parsable --array=1-${NUM_JOBS}%${MAX_CONCURRENT} <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_batch
#SBATCH --partition=${PARTITION}
${ACCOUNT_DIRECTIVE}
${EXCLUDE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${BATCH_MEM}
#SBATCH --time=${BATCH_WALLTIME}
#SBATCH --output=${LOG_DIR}/batch_%a.out
#SBATCH --error=${LOG_DIR}/batch_%a.err

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
export PYTHONUSERBASE=${PY_USERBASE}
export SQLITE_JOURNAL_MODE=DELETE   # WAL readers hang on BeeGFS

python -u main.py --config "${RUN_CONFIG}" --run-dir "${RUN_DIR}" \
    --batch-job-index \${SLURM_ARRAY_TASK_ID} --total-batch-jobs ${NUM_JOBS} \
    --tasks parsing,mass_calculating
EOF
)
echo "Submitted batch array:     ${BATCH_JOB_ID} (${NUM_JOBS} tasks)"

# --- Phase 2: post-processing over ALL IM shards ---------------------------
POSTPROC_JOB_ID=$(sbatch --parsable --dependency=afterok:${BATCH_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_postproc
#SBATCH --partition=${PARTITION}
${ACCOUNT_DIRECTIVE}
${EXCLUDE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${POSTPROC_MEM}
#SBATCH --time=${POSTPROC_WALLTIME}
#SBATCH --output=${LOG_DIR}/postproc.out
#SBATCH --error=${LOG_DIR}/postproc.err

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
export PYTHONUSERBASE=${PY_USERBASE}
export SQLITE_JOURNAL_MODE=DELETE   # WAL readers hang on BeeGFS

python -u main.py --config "${RUN_CONFIG}" --run-dir "${RUN_DIR}" \
    --tasks post_processing
EOF
)
echo "Submitted post-processing: ${POSTPROC_JOB_ID} (depends on ${BATCH_JOB_ID})"

# --- Phase 3: histogram creation (one processed shard -> one job) -----------
HIST_JOB_ID=$(sbatch --parsable --dependency=afterok:${POSTPROC_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_hist
#SBATCH --partition=${PARTITION}
${ACCOUNT_DIRECTIVE}
${EXCLUDE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${BATCH_MEM}
#SBATCH --time=${HIST_WALLTIME}
#SBATCH --output=${LOG_DIR}/hist.out
#SBATCH --error=${LOG_DIR}/hist.err

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
export PYTHONUSERBASE=${PY_USERBASE}
export SQLITE_JOURNAL_MODE=DELETE   # WAL readers hang on BeeGFS

python -u main.py --config "${RUN_CONFIG}" --run-dir "${RUN_DIR}" \
    --batch-job-index 1 --total-batch-jobs 1 \
    --tasks histogram_creation
EOF
)
echo "Submitted histogram job:   ${HIST_JOB_ID} (depends on ${POSTPROC_JOB_ID})"

# --- Phase 4: merge — hadd histograms + aggregate stats + plots -------------
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:${HIST_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_merge
#SBATCH --partition=${PARTITION}
${ACCOUNT_DIRECTIVE}
${EXCLUDE_DIRECTIVE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=${MERGE_WALLTIME}
#SBATCH --output=${LOG_DIR}/merge.out
#SBATCH --error=${LOG_DIR}/merge.err

cd ${PIPELINE_DIR}

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source \${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_107 x86_64-el9-gcc14-opt"
export PYTHONUSERBASE=${PY_USERBASE}
export SQLITE_JOURNAL_MODE=DELETE   # WAL readers hang on BeeGFS

python -u main.py --config "${RUN_CONFIG}" --run-dir "${RUN_DIR}" --merge-only
EOF
)
echo "Submitted merge job:       ${MERGE_JOB_ID} (depends on ${HIST_JOB_ID})"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel ${BATCH_JOB_ID} ${POSTPROC_JOB_ID} ${HIST_JOB_ID} ${MERGE_JOB_ID}"
echo "Logs:     ${LOG_DIR}/"
echo "Output:   ${RUN_DIR}/"
