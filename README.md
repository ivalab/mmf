# SGL
Symbolic Goal Learning for Human Instruction Following in Robot Manipulation

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

## Dataset

## Data Generation

## Manipulation Experiment
