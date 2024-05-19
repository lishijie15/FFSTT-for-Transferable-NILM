# FFSTT for Transferable NILM
Neural NILM is used as a platform for comparing multiple disaggregation approaches. The detailed hyper-parameters of FFSTT is summarized in the Anonymous GitHub. The detailed hyper-parameters of FFSTT is summarized in the following table . The setting of the $D$, $D_{\text{ff}}$ and the number of attention heads takes into account the computational performance of the GPU, as well as the complexity of the datasets and its optimal feature representation. The learning rate of the Adam optimizer and batch size are initially set to empirical values, followed by multiple result-driven simulations to obtain the optimal values.
Codes for the Transferable NILM will be prioritized for public release, followed by a gradual disclosure of the complete codebase.

The hyper-parameter settings for FFSTT are as follows:

| Parameters Description                            | Value                             |
| :------------------------------------------------ | --------------------------------- |
| Batch size                                        | 128                               |
| Learning rate                                     | 0.001                             |
| Number of epochs                                  | 100                               |
| Dimension of embedding ($D$)                      | 64                                |
| Dimension of feed-forward layer ($D_{\text{ff}}$) | 64                                |
| Sequence length                                   | 799 (W.M), 699 (FG)               |
| Number of attention heads                         | 8                                 |
| Random Seed                                       | 2000                              |
| Loss function                                     | $\mathcal{L}_2\text{Loss}(\cdot)$ |
| Dropout probability                               | 0.1                               |
| Gaussian noise                                    | $\mu =0,\sigma =0.1$              |
| Optimizer                                         | Adam                              |
| Initialization function                           | Standard                          |

### Requirements

All neural networks are implemented using `PyTorch==2.1.1+cuda121` versions as the underlying module and trained on a single NVIDIA RTX 4090 GPU with 24 GB RAM.

Please code as below to install some necessary libraries.

```
pip install -r requirements.txt
```

## Data Preparation

Two real-world datasets, U.K.-DALE (**U**) and REDD (**R**) are used to evaluate the performance and transferability of our proposed model.

The *U.K.-DALE* dataset contains power consumption measurements from diverse appliances in UK households, recording low-frequency sampling data at 6s and high-frequency sampling data at 16kHz.

The *REDD* dataset includes measurements from six houses with multiple submeters, covering appliances like refrigerators and air conditioners. The sampling rate varies from 1 kHz to 15 kHz.

For the U.K.-DALE (5s sampling rate), House1 (U1: 2013/01/06-2013/31/07) and House2 (U2: 2013/01/06-2013/31/07) during a two-month period (2013/01/06-2013/31/07) are selected, focusing on five typical household appliances for NILM: kettle (KT), washing machine (W.M), dishwasher (D.W), microwave (MV), and fridge (FG). For the REDD (5s sampling rate), all the data for House1 (R1) and House3 (R3) are used. However, due to the absence of KT data in the REDD dataset, NILM is performed using the other four appliances. For both datasets, 80% of the monitoring data allocated to the training set and 20% to the testing set.

## Domain Adaptation Analysis

In this paper, three cases are designed and compared for both intra- and interdomain adaptations. The proposed model is superior to other methods, especially for the appliances with a relatively complicated active profile, such as the W.M with multiple functions, which is very difficult to be monitored. 

1) *Case 1: Intradomain transfer.* The U.K.-DALE with its extensive record duration and the REDD with its high resolution are selected for this case study. All methods are evaluated through two intradomain tasks: $U_i$ → $U_j$ and $R_i$ → $R_j$. In these tasks, $i$ and $j$ respectively denote the IDs of the source house, which possesses 100% labeled data for training, and the target house, which contains 25% labeled data for fine-tuning. Specifically, for the U.K.-DALE, we set $i = 2$ as the source domain and $j = 1$ as the target domain. Similarly, for the REDD, we set $i = 2$ and $j = 6$ for the intradomain adaptation evaluation. These results underscore the effectiveness and superiority of the proposed method, even in the presence of substantial distribution differences between the source and target domains ($U_2$ → $U_1$). It is worth noting that the performance of models closely aligns with the activity profiles of the appliances, beyond just the label distribution. Hence, while the proposed model remains the most suitable for stable and low power consumption MV, the extent of performance improvement is limited.

2) *Case 2: Interdomain transfer.* To analyze adaptability in interdomain transfer, we adopt the notation $i$ and $j/y$ to represent the IDs of the source house (with 100% labeled data) and the target house (with 25% labeled data), respectively. Specifically, our evaluation encompasses four interdomain tasks: $U_i$ → $R_j$, $U_i$ → $R_y$, $R_i$ → $U_j$, and $R_i$ → $U_y$, to comprehensively assess all methods. When U.K.-DALE serves as the source domain, we set $i=2$, and $j=1/3$. Similarly, when REDD serves as the source domain, we assign $i=1$, with $j$ being either $1$ or $2$. As illustrated in Table I, the proposed model not only demonstrates superior performance on most appliances but also significantly exhibits improvement compared to the non-pretrained approach across all appliances.

## Usage

### Requirements

All neural networks are implemented using `PyTorch==2.1.1+cuda121` versions as the underlying module and trained on a single NVIDIA RTX 4090 GPU with 24 GB RAM.

Please code as below to install some necessary libraries.

```
pip install -r requirements.txt
```

### Getting started
To perform the FFSTT, first download the [UK-DALE](https://jack-kelly.com/data/) and [REDD](http://redd.csail.mit.edu/) datasets and place them in the root directory folder `./data/`. If you only have a single GPU, you can directly execute the following code:

```
python main.py
```

If you are using multiple GPU servers, you can set the GPU index by executing the following code:

```
bash NILM_uk.sh
```

**File Directory**

Since the original nilmtk toolkit may seem to be redundant for testing Neural NILM algorithms (Deep Learning method), we sorely use it  for the generation of power data within a specific period of time. Thus you can **only focus on  these files or folders**: `\nilmtk\api.py`, `\nilmtk\loss.py` and `\nilmtk\disaggregate`.

- **References**

  [1] https://github.com/nilmtk/nilmtk.

  [2] KELLY J, KNOTTENBELT W. Neural NILM: Deep neural networks applied to energy disaggregation. In Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments. November 4-5, 2015, New York, NY, USA, 55–64.

  [3] https://github.com/Ming-er/NeuralNILM_Pytorch.
