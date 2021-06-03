# QuASE Demo
This is the code repository for the demo of the ACL paper [QuASE: Question-Answer Driven Sentence Encoding](https://hornhehhf.github.io/hangfenghe/papers/ACL_QuASE_final.pdf).
If you use this code for your work, please cite
```
@inproceedings{HeNgRo20,
    author = {Hangfeng He and Qiang Ning and Dan Roth},
    title = {{QuASE: Question-Answer Driven Sentence Encoding}},
    booktitle = {Proc. of the Annual Meeting of the Association for Computational Linguistics (ACL)},
    year = {2020},
}

```
## Play with our [online demo](https://cogcomp.seas.upenn.edu/page/demo_view/QuASE).

## Install dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python>=3.6\
pip install -r requirements.txt

## Download the pre-trained models
Our pre-trained models can be found in the [google drive](https://drive.google.com/drive/folders/1j6ufXtxFekPM9CfM5CxKfmwHsqLR8kNY?usp=sharing).

## Change the model path
Change the /path/to/models in the app/bert_lm.py to the path of the downloaded pre-trained models.

## Run the online model
python manage.py runserver 0.0.0.0:4003 (change the 4003 to other port number as needed)


