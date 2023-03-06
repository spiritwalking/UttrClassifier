[Introduction]

This dataset is described in the paper "NaturalConv: A Chinese Dialogue Dataset Towards Multi-turn Topic-driven Conversation". The entire dataset contains 6 other files besides ReadMe.txt. 


1. license.pdf:

It is a pdf file describing the terms and conditions of this dataset.


2. dialog_release.json: 

It is a json file containing a list of dictionaries.

After loading in python this way:

import json
import codecs
dialog_list = json.loads(codecs.open("dialog_release.json", "r", "utf-8").read())

dialog_list is a list whose element is a dictionary.

Each dictionary contains three keys: "dialog_id", "document_id" and "content".
"dialog_id" is an unique id for this dialogue.
"document_id" represents which doucment this dialogue is grounded on.
"content" is a list of the whole dialogue session. 

Altogether there are 19919 dialogs.


3. document_url_release.json:

It is a json file containing a list of dictionaries.

After loading in python this way:

import json
import codecs
document_list = json.loads(codecs.open("document_url_release.json", "r", "utf-8").read())

document_list is a list whose element is a dictionary.

Each dictionary contains three keys: "document_id", "topic", and "url".
"document_id" is an unique id for this document.
"topic" represents which topic this document comes from.
"url" represents the url of the original document.

Altogether there are 6500 documents.


4, 5, and 6. train.txt, dev.txt, and test.txt:

Each file contains the "dialog_id" for train, dev and test, respectively.


[Document Downloading]

For research purpose only, you can refer to the code shared in the repositary below for downloading the document texts through our provided urls.

https://github.com/naturalconv/NaturalConvDataSet


[Citation]

Please kindly cite our paper if you find this dataset useful:

@inproceedings{aaai-2021-naturalconv,
    title={NaturalConv: A Chinese Dialogue Dataset Towards Multi-turn Topic-driven Conversation},
    author={Wang, Xiaoyang and Li, Chen and Zhao, Jianqiao and Yu, Dong},
    booktitle={Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI-21)},
    year={2021}
}


[License]

The dataset is released for non-commercial usage only. By downloading, you agree to the terms and conditions in the license.pdf file. For the authorization of commercial usage, please contact ailab@tencent.com for details.


[Disclaimers]

The dataset is provided AS-IS, without warranty of any kind, express or implied. The views and opinions expressed in the dataset including the documents and the dialogues do not necessarily reflect those of Tencent or the authors of the above paper.