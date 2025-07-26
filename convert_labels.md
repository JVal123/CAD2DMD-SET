# Create new training labels format without regeneration

To create new training labels in the alternate format without regenerating the entire dataset, you can run this [script](labeler.py).

The process has **2 stages**:

1. Generate the new label format for the foreground object

    ```bash
    python labeler.py --stage=foreground --mode=one_word --csv_file=dataset/foreground/foreground.csv --foreground_labels_json=dataset/foreground/foreground_labels_short.json
    ```

2. Use the newly generated foreground labels for the final training label generation

    ```bash
    python labeler.py --stage=training --csv_file=dataset/results/training_set/training.csv --foreground_labels_json=dataset/foreground/foreground_labels_short.json --training_labels_json=dataset/results/training_set/training_labels_short.json
    ```
**Note:** The example provided illustrates how to create **one-word** label answers when the dataset was originally generated using **full-sentence** answers.


# Convert training labels for LVLM format

During the finetuning stage, each large vision-language model (LVLM) requires training labels to be structured in a specific format, which often varies across models. To accommodate this diversity, a set of conversion scripts has been developed to transform the labels from the CAD2DMD-SET dataset into the required formats for individual LVLMs. Currently, the framework supports two LVLMs, for which conversion pipelines are readily available:


* [LLaVA](https://github.com/haotian-liu/LLaVA.git)
* [InternVL](https://github.com/OpenGVLab/InternVL.git)


## Arguments

* **input_labels** - Path to the training labels json file, from CAD2DMD-SET
* **output_labels** - Path to the output labels json file, in the LLaVA compatible format
* **image_dir** - Path to the training set image folder

### LLaVA

If you want to convert your CAD2DMD-SET labels to a LLaVA compatible version, please use this [script](llava_converter.py).

#### Example:

```bash
python llava_converter.py -input_labels=dataset/results/training_set/training_labels.json -output_labels=dataset/results/training_set/training_llava_labels.json -image_dir=dataset/results/training_set
```

### InternVL

If you want to convert your CAD2DMD-SET labels to an InternVL compatible version, please use this [script](internvl_labels_converter.py).

#### Example:

```bash
python internvl_labels_converter.py -input_labels=dataset/results/training_set/training_labels.json -output_labels=dataset/results/training_set/training_internvl_labels.json -image_dir=dataset/results/training_set
```

