
export CUDA_VISIBLE_DEVICES=0,1,2,3

BASE_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH="Your Adapter Path Here"

QUANTIZATION_BITS=4
COMPUTE_DTYPE="bf16"
MERGE_ADAPTER=true

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ADAPTER_NAME=$(basename "${ADAPTER_PATH}")
OUTPUT_DIR="./evaluation_results/${ADAPTER_NAME}_${TIMESTAMP}"
OUTPUT_FILE="${OUTPUT_DIR}/metrics.json"
LOG_FILE="${OUTPUT_DIR}/eval.log"

mkdir -p ${OUTPUT_DIR}

echo "========================================"
echo "Starting Evaluation Script"
echo "========================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Base Model: ${BASE_MODEL_PATH}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Load Base Quantization: ${QUANTIZATION_BITS}"
[ ${QUANTIZATION_BITS} -eq 0 ] && echo "Compute Dtype: ${COMPUTE_DTYPE}"
echo "Merge Adapter: ${MERGE_ADAPTER}"
echo "Output File: ${OUTPUT_FILE}"
echo "Log File: ${LOG_FILE}"
echo "----------------------------------------"


MERGE_FLAG=""
if [ "$MERGE_ADAPTER" = true ]; then
    MERGE_FLAG="--merge_adapter"
fi

python -u src/eval.py \
    --base_model_path "${BASE_MODEL_PATH}" \
    --adapter_path "${ADAPTER_PATH}" \
    --quantization_bits ${QUANTIZATION_BITS} \
    --compute_dtype ${COMPUTE_DTYPE} \
    ${MERGE_FLAG} \
    --output_file "${OUTPUT_FILE}" \
    2>&1 | tee ${LOG_FILE}

EXIT_CODE=${PIPESTATUS[0]}

echo "----------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation finished successfully."
    echo "Results saved to: ${OUTPUT_FILE}"
else
    echo "Evaluation failed with exit code ${EXIT_CODE}."
    echo "Check log file for details: ${LOG_FILE}"
fi
echo "========================================"

exit $EXIT_CODE