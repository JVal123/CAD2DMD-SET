#!/bin/bash

#>>> Conda initialization >>>
__conda_setup="$('~/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" # Change conda path
eval "$__conda_setup"
# <<< Conda initialization <<<

# ------------------------------------------------------------
# ------------------ Configurable Variables ------------------
# ------------------------------------------------------------

# Display Generator Arguments

DISPLAY_NUMBER=3 # Number of display images per device

# Foreground Generator Arguments

BLENDER_PATH="~/blender-4.3.2-linux-x64/blender" 
RENDER_NUMBER=3 # Number of foreground renders per device
LABEL_FORMAT="full_answers" # "full_answers" or "one_word"

# Image Composer Arguments

COMPOSITION_METHOD="fopa"  # "fopa" or "random"
RESULT_DIR="dataset/results/training_set"
HARMONIZATION="none" # "none" or "color_transfer"
TOTAL_COMPOSITES=3 # Number of total composite/training images (not per device)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Pipeline stage options: 
# "display",
# "foreground"
# "composition"
# "display,foreground"
# "foreground,composition"
# "display,foreground,composition"

RUN_STAGE=${RUN_STAGE:-"display,foreground,composition"}

# ----------------------------------------------------
# ------------------ Internal Logic ------------------
# ----------------------------------------------------

ensure_conda_env() {
    local ENV_NAME=$1
    local YAML_FILE=$2

    if ! conda info --envs | grep -q "^${ENV_NAME}[[:space:]]"; then
        echo "📦 Creating conda environment '$ENV_NAME'..."
        conda env create -f "$YAML_FILE" -n "$ENV_NAME"
    fi

    echo "🔄 Deactivating any active conda environment..."
    conda deactivate 2>/dev/null

    echo "🔄 Activating conda environment '$ENV_NAME'"
    conda activate "$ENV_NAME"
}

run_display_generator() {
    ensure_conda_env "blender" "blender.yml"
    python displays/create_display.py -display_number "$DISPLAY_NUMBER"
    echo 'Display Generation Stage Complete! ✅'
}

run_foreground_generator() {
    ensure_conda_env "blender" "blender.yml"
    python foreground_generator.py \
        -blender_path "$BLENDER_PATH" \
        -display_number "$DISPLAY_NUMBER" \
        -render_number "$RENDER_NUMBER" \
        -label_format "$LABEL_FORMAT"
    echo 'Foreground Generation Stage Complete! ✅'
}

run_image_composer() {
    ensure_conda_env "libcom" "libcom.yml"
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    python image_composition.py \
        --method "$COMPOSITION_METHOD" \
        --result_dir "$RESULT_DIR" \
        --total_images "$TOTAL_COMPOSITES" \
        --harmonization "$HARMONIZATION"
    echo 'Image Composition Stage Complete! ✅'
}

# ----------------------------------------------------
# ------------------ Execution Flow ------------------
# ----------------------------------------------------

IFS=',' read -ra STAGES <<< "$RUN_STAGE"
for STAGE in "${STAGES[@]}"; do
    case "$STAGE" in
        display)
            run_display_generator
            ;;
        foreground)
            run_foreground_generator
            ;;
        composition)
            run_image_composer
            ;;
        *)
            echo "❌ Unknown stage: $STAGE"
            ;;
    esac
done

echo "✅ Pipeline execution complete!"
