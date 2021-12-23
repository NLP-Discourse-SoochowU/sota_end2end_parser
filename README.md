## Introduction

In this project, we provide an end2end DRS parser where the EDU segmenter
in "Zhang et al., **Syntax-Guided Sequence to Sequence Modeling for Discourse Segmentation**" 
and the DRS parser in "Zhang et al., **Adversarial Learning for Discourse Rhetorical Structure Parsing**" are applied.

Any questions, just send e-mails to zzlynx@outlook.com (Longyin Zhang).


#### Usage

1. Before running this end2end parser, you need to download the following
packages in your software environment:

- Python 3.6.10
- transformers 3.0.2
- pytorch 1.5.0
- numpy 1.19.1
- cudatoolkit 9.2 cudnn 7.6.5
- stanford-corenlp-full-2018-02-27
- other necessary python packages (etc.)
If you have troubles in downloading the stanford CoreNLP toolkits, we provide
a duplication of it for you at "https://pan.baidu.com/s/1FGZe9vBZap2ZNv_D3XyRjA",
and the extraction code is **n6hx**.

2. Due to the file size limitation, we put the pre-trained model at
https://pan.baidu.com/s/1u0FLgPydISKR-MVcs2SFdg, and the password is **lynx**.
After obtaining the pre-trained models, you need to put the two files at the
following two places. **It should be noted that since many real-life data usually do not contain standard paragraph boundaries, we only provide the pre-trained model with sentence boundaries considered, which is closer to practical application.**
```
   (parser) data/models_saved/model.pth
   (LM) data/models_saved/xl_model.pth
```

3. Prepare your own data with reference to the examples in "data/e2e", where
"raw.txt" refers to the article with several sentences, "edu.txt" refers to the
texts after EDU segmentation, and "trees.pkl" refers to the generated DRS trees.
When everything is ready, run the following command:
```
   python pipeline.py
```


<b>-- License</b>
```
   Copyright (c) 2019, Soochow University NLP research group. All rights reserved.
   Redistribution and use in source and binary forms, with or without modification,
   are permitted provided that the following conditions are met:
   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this
      list of conditions and the following disclaimer in the documentation and/or other
      materials provided with the distribution.
```



