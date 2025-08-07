# CAD2DMD-SET

Synthetic Generation Tool of Digital Measurement Device CAD Model Datasets for finetuning Large Vision-Language Models. [[Paper]]()

## Prerequisites

* Blender - Download [here](https://www.blender.org/).
* Conda - Download [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

## Install 

If you are not using Linux, do NOT proceed:

1. Clone this repository and navigate to CAD2DMD-SET folder

```bash
git clone (add)
cd CAD2DMD-SET
```
2. Install Packages and Create Conda Environments

```bash
conda env create -f blender.yml
conda env create -f libcom.yml
```

3. Clone [libcom](https://github.com/bcmi/libcom) repository to the same parent folder as CAD2DMD-SET, as follows:

```bash
Parent folder/
├── CAD2DMD-SET/
├── libcom/  
```


## Run

In order to use the CAD2DMD-SET tool, you need to run the [run.sh](run.sh) script. Prior to using it, you should:

1. Open the bash script.
1. Update conda path on line 4.
1. Create a "dataset" folder.
1. Inside it, create a "background" folder with your desired background images, to be used during the image composition stage. The folder stucture should be as follows:

    ```bash
    CAD2DMD-SET/
    ├── dataset/
    │   ├── background/  
    ```



### Variables

| Variable          | Description |
| :---------------- | :------      |
| DISPLAY_NUMBER    |   Number of display images per device        |
| BLENDER_PATH           |   Path to blender executable        |
| RENDER_NUMBER    |  Number of foreground renders per device        |
| LABEL_FORMAT |  Defines labels format (full answers vs one word answers)     |
|COMPOSITION_METHOD | Defines composition method ([fopa](https://github.com/bcmi/FOPA-Fast-Object-Placement-Assessment.git) vs random)
|RESULT_DIR | Path to training set directory
|TOTAL_COMPOSITES | Total number of training (composite) images
|CUDA_VISIBLE_DEVICES | Defines visible GPUs
|RUN_STAGE | Defines what stage or stages of CAD2DMD-SET are executed


#### 🧩 `RUN_STAGE` Variable

The `RUN_STAGE` variable controls **which parts of the dataset generation pipeline will be executed** when running the `run.sh` script.

You can specify one or more stages as a **comma-separated list**, depending on what you want to run.


#### ✅ Available Stages

| Value         | Description |
|---------------|-------------|
| `display`     | Runs the **Display Generator** to create images of digital displays with random measurements. |
| `foreground`  | Runs the **Foreground Generator** to render synthetic foreground images using Blender. |
| `composition` | Runs the **Image Composer** to blend foregrounds and backgrounds using a model or random placement. |



#### 🛠 How to Use

You can pass the value inline when executing the script, for example:

```bash
RUN_STAGE="display,foreground" bash run.sh
```

This will run only the Display Generator and Foreground Generator, skipping the Image Composition stage. By **default**, the script runs **all stages**.


## Label Conversion

Please check our label conversion [guide](convert_labels.md), if you want to: 

1. **Change** the tool's **label format without generating a new dataset**.

1. **Finetune** one of the supported **LVLMs** using the generated training set.


## Add new device

If you want to add a new device, please check our [guide](add_device.md).

## Citation

If you find CAD2DMD-SET useful for your research and applications, please cite using this BibTeX:

```BibTeX

```
