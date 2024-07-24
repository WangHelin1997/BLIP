## BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

In addition, install `torch torchaudio` and
```
!git clone https://github.com/WangHelin1997/s3prl
!cd s3prl
!pip install -e "[.all]"
```

### Pre-train:
1. Prepare Hugginface-like audio dataset. 
2. In configs/pretrain.yaml, set up to your own path.
3. Train example:  <pre/>sh pretrain.sh</pre>

### Image-Text Captioning:
1. Train example:  <pre/>sh train_caption.sh</pre> 

### Image-Text Retrieval:
1. Train example:  <pre/>sh train_retrieval.sh</pre> 



