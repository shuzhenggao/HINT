# HINT

The repo of our paper "Learning in the Wild: Leveraging the Unlabeled Data Task-Specifically for Pre-trained Code Models". 

## Dependency
* Python 3.9
* PyTorch 1.8.0
* Huggingface transformers
* editdistance
* jsonlines
* gensim
* Java (for METEOR metric calculation)


## Data
Our processed datasets can be downloaded [here](https://figshare.com/articles/dataset/Data_for_ASE_peer_review/22760795).


## Usage'
To reproduce the results, you can first move to the direcotory of each pre-trained code model ``cd {CodeBERT,CodeT5,Unixcoder}``. 

### Code Summarization

```bash
# Run one iteration
bash sh/run.sh your_lang your_threshold
## such as bash sh/run.sh java 30
 
# Run more
bash sh/circle.sh your_lang your_threshold

#Copy evall for evaluation on codet5 and unixcoder
cp -r ../CodeBERT/evall ./
```

### Defect Detetction

```bash
cd Defect
# Run baseline model and HINT
bash run_ssl.sh 

# Run more
bash circle.sh 
```

### Assertion Generation

```bash
# Run one iteration
bash sh/run-defect.sh  your_threshold
## such as bash sh/run.sh  30
 
# Run more
bash sh/circle.sh your_threshold
```
You can change the model's hyperparameter in each bash file. 


## Extra Experiment results of ablation study and parameter analysis



###  Ablation study
**Code Summarization (Unixcoder Python)**
|               | BLEU            | ROUGE                       | METEOR | CIDER |
| ------------- | --------------------- | --------------------------- | ------ | ----- |
| Random selection                | 23.36            | 41.13          | 15.99  | 1.42  |
| -w/o loss-based selection       | 23.10            | 40.70          | 16.12  | 1.40  |
| -w/o retrieval-based selection  | 23.66            | 41.54          | 16.62  | 1.47  |
| -w/o data selection             | 23.33            | 40.96          | 18.97  | 1.41  |
| -w/o noise tolerant loss        | 23.67            | 41.49          | 16.42  | 1.46  |
| -w/o consistency regularization | 23.68            | 41.54          | 16.58  | 1.47  |


**Code Summarization (CodeBERT Java)**
|               | BLEU            | ROUGE | METEOR | CIDER |
| ------------- | --------------------- | --------------------------- | ------ | ----- |
| Random selection                | 14.48 |	28.86 |	8.63 |	0.69
| -w/o loss-based selection       | 14.28 |	28.62 |	8.77 |	0.67 
| -w/o retrieval-based selection  | 14.39 |	29.02 |	8.77 |	0.68 
| -w/o data selection             | 13.70 |	27.60 |	8.14 |	0.60 
| -w/o noise tolerant loss        | 14.55 |	29.41 |	8.62 |	0.68 
| -w/o consistency regularization | 14.49 |	29.25 |	8.65 |	0.68 



###  Analysis of the parameter threshold K
**Code Summarization (CodeT5 Java)**
|               | **BLEU**              | ROUGE                       | METEOR | CIDER |
| ------------- | --------------------- | --------------------------- | ------ | ----- |
| threshold 10% | **18.16**             | 35.52                       | 12.34  | 1.21  |
| threshold 15% | **18.27**             | 35.27                       | 12.30  | 1.21  |
| threshold 20% | **18.31**             | 35.49                       | 12.36  | 1.22  |
| threshold 25% | **18.32**             | 35.39                       | 12.34  | 1.22  |
| threshold 30% | **18.16**             | 35.40                       | 12.37  | 1.21  |
| threshold 35% | **18.01**             | 35.01                       | 12.3   | 1.19  |

**Code Summarization (CodeT5 python)**
|               | **BLEU**              | ROUGE                       | METEOR | CIDER |
| ------------- | --------------------- | --------------------------- | ------ | ----- |
| threshold 10% | **22.20**             | 41.32                       | 16.21  | 1.34  |
| threshold 15% | **22.18**             | 41.24                       | 16.27  | 1.33  |
| threshold 20% | **22.23**             | 41.31                       |16.31   | 1.34  |
| threshold 25% | **22.35**             | 41.45                       | 16.31  | 1.35  |
| threshold 30% | **22.24**             | 41.31                       | 16.19  | 1.34  |
| threshold 35% | **22.24**             | 41.36                       | 16.34  | 1.34  |

**Code Summarization (Unixcoder Java)**
|               | **BLEU**  | ROUGE | METEOR | CIDER |
| ------------- | --------- | ----- | ------ | ----- |
| threshold 10% | **18.86** | 34.64 | 12.49  | 1.27  |
| threshold 15% | **18.82** | 34.78 | 12.36  | 1.27  |
| threshold 20% | **18.83** | 34.94 | 12.36  | 1.27  |
| threshold 25% | **18.90** | 35.16 | 12.38  | 1.28  |
| threshold 30% | **18.84** | 35.12 | 12.36  | 1.27  |
| threshold 35% | **18.80** | 35.02 | 12.29  | 1.27  |

**Code Summarization (Unixcoder python)**
|               | **BLEU**  | ROUGE | METEOR | CIDER |
| ------------- | --------- | ----- | ------ | ----- |
| threshold 10% | **23.73** | 41.34 | 16.45  | 1.46  |
| threshold 15% | **23.75** | 41.48 | 16.53  | 1.47  |
| threshold 20% | **23.70** | 41.53 | 16.53  | 1.47  |
| threshold 25% | **23.77** | 41.67 | 16.64  | 1.48  |
| threshold 30% | **23.71** | 41.63 | 16.62  | 1.47  |
| threshold 35% | **23.65** | 41.58 | 16.58  | 1.46  |

**Defect Detection**
|               | P     | R     | **F1**    |
| ------------- | ----- | ----- | --------- |
| threshold 10% | 37.96 | 18.00 | **24.42** |
| threshold 15% | 29.99 | 20.50 | **24.35** |
| threshold 20% | 31.90 | 21.34 | **25.57** |
| threshold 25% | 33.28 | 20.96 | **25.73** |
| threshold 30% | 26.90 | 23.01 | **24.80** |
| threshold 35% | 32.81 | 19.57 | **24.52** |

**Assertion Generation**
|               | **ACC**   | PM    | LCS   | ED    |
| ------------- | --------- | ----- | ----- | ----- |
| threshold 10% | **45.67** | 48.36 | 74.05 | 16.93 |
| threshold 15% | **45.80** | 48.41 | 74.10 | 16.93 |
| threshold 20% | **46.02** | 48.66 | 74.28 | 16.77 |
| threshold 25% | **46.64** | 49.18 | 74.42 | 16.64 |
| threshold 30% | **46.78** | 49.42 | 74.56 | 16.74 |
| threshold 35% | **47.13** | 49.65 | 74.72 | 16.61 |


### Analysis of the parameter μ
**Code Summarization (Unixcoder Java)**
|         |      **BLEU-4**      | ROUGE | METEOR | CIDER |
|---------|----------------------|-------|--------|-------|
| μ(0)    |      **18.93**       | 35.11 | 12.46  | 1.28  |
| μ(0.25) |      **18.90**       | 35.16 | 12.43  | 1.28  |
| μ(0.5)  |      **18.90**       | 35.16 | 12.38  | 1.28  |
| μ(0.75) |      **18.84**       | 35.17 | 12.37  | 1.27  |
| μ(1)    |      **18.73**       | 34.98 | 12.28  | 1.26  |

**Defect Detection**
|         | P     | R     | **F1**    |
|---------| ----- | ----- | --------- |
| μ(0)    | 32.30 | 19.39 | **24.23** |
| μ(0.25) | 32.40 | 20.32 | **24.97** |
| μ(0.5)  | 33.28 | 20.96 | **25.73** |
| μ(0.75) | 27.71 | 23.75 | **25.57** |
| μ(1)    | 35.81 | 19.20 | **25.00** |

**Assertion Generation**
|         | **ACC**               | PM    | LCS   | ED    |
|---------| --------------------- | ----- | ----- | ----- |
| μ(0)    | **46.76**             | 49.24 | 74.52 | 16.79 |
| μ(0.25) | **46.98**             | 49.52 | 74.65 | 16.58 |
| μ(0.5)  | **47.13**             | 49.65 | 74.72 | 16.61 |
| μ(0.75) | **46.99**             | 49.50 | 74.75 | 16.59 |
| μ(1)    | **47.02**             | 49.44 | 74.61 | 16.52 |



### Analysis of the parameter t
**Code Summarization (Unixcoder Java)**
|         |      **BLEU-4**      | ROUGE | METEOR | CIDER |
|---------|----------------------|-------|--------|-------|
| t(0.2)  |      **18.83**       | 35.08 | 12.44  | 1.27  |
| t(0.3)  |      **18.75**       | 34.96 | 12.36  | 1.27  |
| t(0.4)  |      **18.90**       | 35.16 | 12.38  | 1.28  |
| t(0.5)  |      **18.91**       | 35.10 | 12.38  | 1.28  |

**Defect Detection**
|         | P     | R     | **F1**    |
|---------| ----- | ----- | --------- |
| t(0.2)  | 36.17 | 18.55 | **24.52** |
| t(0.3)  | 23.52 | 26.81 | **25.05** |
| t(0.4)  | 33.28 | 20.96 | **25.73** |
| t(0.5)  | 29.51 | 22.45 | **25.50** |

**Assertion Generation**
|         | **ACC**               | PM    | LCS   | ED    |
|---------| --------------------- | ----- | ----- | ----- |
| t(0.2)  | **46.96**             | 49.52 | 74.61 | 16.63 |
| t(0.3)  | **46.78**             | 49.33 | 74.52 | 16.68 |
| t(0.4)  | **47.13**             | 49.65 | 74.72 | 16.61 |
| t(0.5)  | **46.64**             | 49.11 | 74.49 | 16.71 |
