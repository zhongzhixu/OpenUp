# KARA

## Introduction

This repository introduces a knowledge-aware NLP model namely KARA. KARA automatically identifies the help-seekers at a high suicide risk in real-time by mining their online counselling conversations.

### The general design of the KARA
![figure](https://github.com/zhongzhixu/OpenUp/blob/master/design.png)

A knowledge-aware risk assessment (KARA) model is developed to detect the risk of suicide through mining conversations between help-seekers and counsellors. 

Refer to the above figure for the general architecture of the model. In general, there are two components of the KARA model: the knowledge encoder component and the conversation encoder component. We rely on the knowledge encoder component to learn a representation of the suicide-related domain knowledge, and then feed the encoded domain knowledge into the conversation encoder component which analyses the text of conversations, with the encoded domain knowledge, to determine the risk of suicide. 

## Data
We released the suicide knowledge graph (SKG) (see SKG.csv). The original Data is restricted by the ethics of OpenUp. 

### A putative chat between helpseeker and counsellor </br>
~ 你好啊,我叫Julien. 今日你上嚟呢度
有D咩想同我傾啊?</br>
~	唔知咩事，只係覺得攰。我忍左好
耐唔傷害自己，但真係好想</br>
~ 我見到你話想傷害自己, 我好擔心你,可唔可
以講多少少？發生咗咩事啊?咁點解覺得攰嘅,？
仲想傷害自己?</br>
~ 試過介手，食過好多藥，想試埋上吊，我忍
住咗，但很想……</br>
~ 雖然唔知係你身上發生咩事,我都感覺到你好
唔開心。你有嘗試過傷害自己嗎?你唔介意的
話,可唔可以同我下你之前點傷害自己? 我擔
心你有生命危機危險呢！</br>
~ 有咩用？我又係邊個？我只係好想睇著啲血流
落嚟.</br>
~	係你試過做定係想做?你之前都試過介手，食藥，上
吊? 而家都有計劃傷害自己嗎? 聽到你講傷害過自己, 
我好擔心你，而且都好心痛。你一直忍住左咩令你
咁辛苦?睇嚟你都好慨嘆, 係米經歷左啲野令你咁嘅
諗法？</br>	
~ 算罷啦！ 所有嘢都無改變過。我可以同邊個講？
想嚇死人？想多一個人痛苦咩？</br>

### The SKG 
![figure](https://github.com/zhongzhixu/OpenUp/blob/master/skg.png)


## Code
conversation_level_KG_multi_input.py

## Environment
Python 3.6</br>

Keras 2.2.4</br>

TensorFlow 1.13.1</br>



