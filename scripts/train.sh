#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

LORA_R=16
QUANTIZATION_BITS=4
USE_QUANTIZATION=true

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4

MAX_LEN=1024
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.05
SAVE_STEPS=200
LOGGING_STEPS=10
GRADIENT_CHECKPOINTING=true

OPTIM="paged_adamw_8bit"
EVALUATE=false

# Add the LoRA target modules if needed
# LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj,fc1,fc2,fc3,fc4,fc5,fc6,fc7,fc8"

LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p ${LOG_DIR} 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
OUTPUT_PATH="./lora_r${LORA_R}_use_quantization${USE_QUANTIZATION}_${QUANTIZATION_BITS}bit"

echo "Starting training..."
echo "Model Path: ${MODEL_PATH}"
echo "LoRA Rank: ${LORA_R}"
echo "Output Path: ${OUTPUT_PATH}"
if [ "${USE_QUANTIZATION}" = true ]; then
  echo "Quantization Bits: ${QUANTIZATION_BITS}"
fi

mkdir -p ${OUTPUT_PATH}

NUM_GPUS=4
# default: bf16， --evaluate means whether to evaluate the model by your own function
# If memory is not enough, you can run "python src/train.py...."（split model）, replace "torchrun --nproc_per_node=${NUM_GPUS} src/train.py" (load one whole model for each GPU, Implement data parallelism)
torchrun --nproc_per_node=${NUM_GPUS} src/train.py \
    --model_path "${MODEL_PATH}" \
    --lora_r "${LORA_R}" \
    --output_path "${OUTPUT_PATH}" \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --max_seq_length ${MAX_LEN} \
    --optim ${OPTIM} \
    --use_quantization ${USE_QUANTIZATION} \
    --evaluate ${EVALUATE} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --quantization_bits ${QUANTIZATION_BITS} 2>&1 | tee ${LOG_FILE}
    # --lora_target_modules ${LORA_TARGET_MODULES} \
    

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training finished successfully. Model adapter saved to ${OUTPUT_PATH}"
else
    echo "Training failed with exit code ${EXIT_CODE}."
fi
exit $EXIT_CODE