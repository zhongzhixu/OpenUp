# KARA

## Introduction

This repository introduces a knowledge-aware NLP model namely KARA. KARA automatically identifies the help-seekers at a high suicide risk in real-time by mining their online counselling conversations.

### The general design of the KARA
![figure](https://github.com/zhongzhixu/OpenUp/blob/figure 2.tif)

## Data
The original Data used in the paper is restricted by the ethics of Hospital Authority of Hong Kong. Simulated data is provided instead.
### Columns </br>
~ obs_start</br>
~	obs_end</br>
~ first_treatment_start</br>
~ first_treatment_end</br>
~ eventdate</br>
~ case_id</br>
~	sub_treatment_start</br>	
~ sub_treatment_end</br>

## Example
```
library(SCCS)
dats <- read.csv('simulated_data.csv',header = TRUE)
dat <- dats
#-----------------------------------
# Multiple exposure types
ageg=c(300,600,900)
con.mod <- standardsccs(event~first_treatment_start+sub_treatment_start+age, indiv=case_id,
                        astart=obs_start, aend=obs_end, aevent=eventdate,
                        adrug =cbind(first_treatment_start,sub_treatment_start),
                        aedrug=cbind(first_treatment_end,sub_treatment_end),
                        expogrp=list(c(-50,1), c(1)), washout=list(c(0,50), c(0)),
                        agegrp=ageg, data=dat)
con.mod
```
## Environment
R (version 3.3.2)</br>
Package "SCCS" (version 1.1)



