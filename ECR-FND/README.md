# ECR-FND: Event Consistency-aware Robust Fake News Detection
## Introduction
The implementation of **ECR-FND**, a novel model for short video fake news detection. It effectively removes event-irrelevant
segments in videos and adaptively captures audio tampering information.


<!-- ## File Structure
```shell
.
├── README  # * Instruction to this repo
├── requirements  # * Requirements for Conda Environment
├── data  # * Place data split & preprocessed data
├── models  # * Codes for ECR-FND Model
├── utils  # * Codes for Training and Inference
├── main  # * Codes for Training and Inference
└── run  # * Codes for Training and Inference
    
``` -->

## Dataset
We conduct experiments on two datasets: FakeSV and FakeTT. 
### FakeSV
FakeSV is the largest publicly available Chinese dataset for fake news detection on short video platforms, featuring samples from
Douyin and Kuaishou, two popular Chinese short video platforms. Each sample in FakeSV contains the video itself, its title, comments, metadata, and publisher profiles. For the details, please refer to [this repo](https://github.com/ICTMCG/FakeSV).
### FakeTT
FakeTT is a newly constructed English dataset for a comprehensive evaluation in English-speaking contexts. For the details, please refer to [this repo](https://github.com/ICTMCG/FakingRecipe). 



## Quick Start
You can utilize FakeRecipe to infer the authenticity of the samples from the test set by following code:
 ```
 # Train the model on FakeSV
  python main.py  --dataset fakesv  

  # Train the model on FakeTT
  python main.py  --dataset fakett  
  ```


