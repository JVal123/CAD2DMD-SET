# Convert labels to LVLM format

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

