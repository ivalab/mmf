# SGL: Symbolic Goal Learning for Human Instruction Following in Robot Manipulation

<img width="385" alt="1645388250(1)" src="https://user-images.githubusercontent.com/27162640/154862585-a4e5f8a7-01a6-4e06-ae1b-65f57e7653ec.png">

>[**SGL: Symbolic Goal Learning for Human Instruction Following in Robot Manipulation**](),
> Ruinian Xu, Hongyi, Chen, Yunzhi, Lin and Patricio A. Vela

## Table of Contents
- [Abstract](#Abstract)
- [Vision and Language Symbolic Goal Learning](#Vision-and-Language-Symbolic-Goal-Learning)
- [Scene Graph Parsing](#Scene-Graph-Parsing)
- [Semantic Textual Similarity](#Semantic-Textual-Similarity)
- [Dataset](#Dataset)
- [Data Generation](#Data-Generation)
- [Manipulation Experiment](#Manipulation-Experiment)

## Abstract
This paper investigates robot manipulation based on human instruction with
ambiguous requests.  The intent is to compensate for imperfect natural
language via visual observations.
Early methods built modular frameworks based on semantic parsing and task
planning for producing sequences of actions from natural language requests. 
Modern methods employ deep neural networks to automatically learn visual 
and linguistic features and map to a sequence of low-level actions, in an
end-to-end fashion. 
These two approaches are blended to create a hybrid, modular framework: 
it formulates instruction following as high-level symbolic goal learning
via deep neural networks followed by low-level task planning via tradtional
PDDL-based planners. 
The vision-and-language learning network predicts the goal state of PDDL,
which is sent to a planner for producing a task-completing action sequence. 
For improving the flexibility of natural language, we further 
incorporate implicit human intents with explicit human instructions.
To learn generic features for vision and language, we propose to separately
pretrain vision and language encoders on scene graph parsing and semantic
textual similarity tasks.
Benchmarking evaluates the impacts of different components of, or options
for, the vision-and-language learning model and shows the effectiveness of
pretraining strategies.
Manipulation experiments conducted in the simulator AI2THOR show 
the robustness of the framework to novel scenarios.

## Vision and Language Symbolic Goal Learning

### Installation
This work is built based on the codebase of mmf. Follow installation instructions provided in their [document](https://mmf.sh/docs/). 

### Usage
Dataset should be downloaded and placed under the folder of `/home/account_name/.cache/torch/mmf/data/datasets`

To train the model with grid feature, BERT and concatenation, run the following command under the root folder:
~~~
mmf_run config=projects/resnet_bert_concat/configs/sgl/defaults.yaml model=resnet_bert_concat dataset=sgl run_type=train training.evaluation_interval=6444
~~~

To evaluate the model with grid feature, BERT and concatenation, first download the pretrained model and place it in the folder of `saved_models`. Then run the following
command:
~~~
mmf_run config=projects/resnet_bert_concat/configs/sgl/defaults.yaml model=resnet_bert_concat dataset=sgl run_type=test checkpoint.resume_file=/path_to_mmf/saved_models/resnet_bert_concat/model.ckpt
~~~

### Development
To develop further advanced models or your own tasks, we will first recommend you to refer to the official [tutorial](https://mmf.sh/docs/). We will also provide a brief introduction.

If you'd like to evaluate or propose some other models consist of different components not benchmarked in this work, you could first create the corresponding model such as resnet_bert_add.py under the folder of `mmf/models`. Since different visual and linguistic encoders require different data processors, we might need to change the `defaults.yaml` under 'mmf/configs/datasets/sgl/'. The current yaml file is set for grid feature and BERT. Other three types of yaml files are provided. b represents BERT, g represents grid, r represents region and no label is for LSTM. Lastly, you will need to create a yaml file under `projects/your_model_name/configs/sgl` which documents training policy.

If you'd like to employ this codebase on your own tasks, you may need to create a new dataset, create a new evaluation metric and even add a new loss. mmf provides official tutorials for all these modifications. You could also refer to the implementation of SGL dataset for the reference.

## Scene Graph Parsing
The potential issue of vague or ambiguous natural language is that it can't deterministically
find a valid sequence of actions with specific objects.
In human instruction following, vision and language are serving different roles.
Vision, which contains the information of objects and their potential interactions in the scene,
provides a potential search space for robotic tasks.
Language helps narrow down or determine the final target task over the reasoned task space.
Applying this insight in vision-and-langauge goal learning framework, we propose to first
pretrain the visual encoder on scene graph parsing tasks.
Scene graph parsing task in deep learning can be formulated as detecting a list of objects
and classifying their relationships. 
Code for pretraining scene graph parsing can be referred to [**here**]().

## Semantic Textual Similarity
Beyond explicit human instruction, we further include implicit human intent as natural language input in this work.
Implicit human intent, which commonly encodes the desire from human, requires incorporating environmental 
information for interpreting its full meanings. 
Even though being structured via quite different words, explicit instruction and implicit intent share
the similar semantic meaning in the robotic task domain. 
To learn such kind of embeddings for both types of natural language, we propose to 
pretrain the linguistic encoder on semantic textual similarity task. 
Code for pretraining BERT on semantic textual similarity task is provided [**here**]().

## Dataset
There are three proposed datasets in this work for vision-and-language symbolic goal learning,
scene graph parsing and semantic textual learning,  respectively.
### Symbolic Goal Learning Dataset
For learning symbolic goal representation from vision and language, we propose a 
dataset contains 32,070 images paired with either human instruction or intent.
It covers five daily activities which are picking and placing, object delivery, cutting, cooking and cleaning. 
The dataset is generated via [**AI2THOR**](https://ai2thor.allenai.org/) and the ground-truth
PDDL goal state is automatically annotated. To be noticed, besides imperfect natural language, 
we further consider the scenario with imperfect vision where partial or full objects involved
in the task miss in the image. With such kind of image, the proposed network is required to 
predict the missing object to be unknown. The dataset is stored in [**SmartTech**]().

### Scene Graph Parsing
The Scene Graph Parsing dataset is also generated via AI2THOR.
It focuses on common daily objects with their object categories, affordances,
attributes and relationships. 
It covers 32 categories, 4 affordances and 5 attributes. 
The ground-truth bounding box, category, affordance and attribute
are automatically annotated during the generation process, which is labor-free.
The dataset is also stored in [**SmartTech**]().

### Semantic Textual Similarity
The Semantic Textual Similarity dataset is generating for learning similar
semantic embeddings for explicit human instructions and implicit human intents.
Semantic meanings of different types of natural langauge are closely embedded in
robotic task domain. There are five daily activities considered in the dataset, which
are picking and placing, object delivery, cutting, cleaning and cooking.
It contains 90,000 pairs of instruction and intent, which are generated based on
a list of templates. To further improve the complexity and diversity, 
sentences are paraphrased by [**Parrot**](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser).
To automatically rank the similarity of two sentences, three different scores are 
assigned based on the following rules:
- 5.0 if two sentences contain the same object and subject,
- 3.3 if sentences match either subject or object, and 
- 1.7 if sentences describe the same task.

The dataset is also stored in [**SmartTech**]().

## Data Generation

## Manipulation Experiment

## License

## Citation
