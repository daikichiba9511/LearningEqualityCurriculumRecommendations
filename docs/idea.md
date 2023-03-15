# Idea memo

## Unsupervised Model

- [ ] Adversarial Training
  - <https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/143764>
  - <https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/372892>

- [ ] Topic Tree
  - <https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/376873>
  - <https://www.kaggle.com/code/daimarusui3/tips-and-recommendations-from-hosts/edit>
- [ ] emsemble
  - <https://www.kaggle.com/code/kojimar/retriever-ensemble>
- [ ] change pretrained roberta to SBert(All-MiniLM-L6)
  1. pretrain SBert by unsupervised manner.
  2. extract data at topk=1200(or more) using KNN that is trained by the step1 model
  3. train SBert by supervised manner using the petrained weight (step1.)
- [x] increase topk when creating KNN
  - topk=1000 => 0.59508
  - topk=1200 => 0.60495

## Supervised Model

- [ ] train with more larger size of topics_id (e.g topk=1200)

## Pipeline