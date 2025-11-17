# State of the Art: Large Language Models for Cybersecurity

## Table of Contents

1. Introduction
   - Objectives
   - Context
   - Motivation for LLMs
2. Key Concepts
    2.1 Large Language Models (LLMs)
   - Advantages
   - Challenges
   - Use Cases
   - Why Large Language Models?
   - How LLMs Work
3. Detecting Cyberattacks with LLMs
    3.1 What Kinds of Attacks Can Be Detected with LLMs?
    3.2 How LLMs Detect These Attacks
    3.3 What Makes LLMs Special for Cybersecurity?
4. LLM Architectures and Optimization Techniques for Cybersecurity
    4.1 Classical ML/DL vs. LLM Approaches
    4.2 Advanced and Recent LLM-based Models
5. Offensive Applications and Vulnerabilities of LLMs
    5.1 Offensive Applications ("The Bad")
    5.2 Inherent Vulnerabilities ("The Ugly")
6. Comparison with Existing Work
    A. Summary of Related Studies
    B. Metrics and Performance Comparisons
    C. Security and Application Insights
7. Common Techniques and Innovations
8. Limitations, Challenges, and Future Directions in LLM-based Cybersecurity
9. Conclusion
10. References and Resources

------

## 1. Introduction

**Objectives:**
 This State of the Art (SOTA) presents a comprehensive overview of the advantages and drawbacks of using Large Language Models (LLMs) for cybersecurity. It emphasizes their role in both defensive and offensive applications, provides quantitative comparisons with traditional methods, and highlights current challenges.

**Context:**
 With the rise of generative AI, LLMs such as GPT-based architectures are increasingly being applied to cybersecurity tasks. They can process code, logs, and network packets as text sequences, enabling new detection and repair capabilities. At the same time, these same generative strengths empower adversaries to launch sophisticated attacks.

**Motivation for LLMs:**

- Superior detection accuracy and speed compared to classical ML/DL methods
- Ability to generalize across different security tasks (code, network, phishing, malware)
- Potential to unify defense strategies under a text-driven paradigm
- Necessity to understand and mitigate their offensive misuse and inherent vulnerabilities

------

## 2. Key Concepts

### 2.1 Large Language Models (LLMs)

**Definition and Principles:**
 LLMs are transformer-based deep learning models trained on massive textual datasets. They capture long-range dependencies and semantic meaning, making them suitable for cybersecurity tasks when data (code, packets, logs) is framed as sequential text.

**Advantages:**

- Outperform traditional ML/DL approaches in many detection and repair tasks [1, 3, 4]
- Effective in code vulnerability detection, repair, and intrusion detection [3, 6]
- Enable high-speed analysis (e.g., 539 predictions per second [11])

**Challenges:**

- Vulnerabilities to prompt injection, denial of service, and membership inference [1, 12]
- Heavy dependence on quality and completeness of training data [13, 14]
- High computational resource requirements

**Use Cases:**

- Code vulnerability discovery and repair
- Intrusion detection (DDoS, IoT, IoV, phishing, spam)
- Threat intelligence analysis and automated incident response

**Why Large Language Models?**
 Traditional models struggle to integrate heterogeneous cybersecurity data and adapt to new attack vectors. LLMs, by treating security data as text, provide scalability and adaptability.

**How LLMs Work in Security Context:**

- Input: network packets, logs, or code as tokenized text
- Model: fine-tuned transformer (e.g., BERT, GPT, BART)
- Output: classification (attack/benign), vulnerability detection, or automated repair

------

## 3. Detecting Cyberattacks with LLMs

### 3.1 What Kinds of Attacks Can Be Detected?

- General network intrusions (malware, phishing, spam)
- DDoS attacks (UDP, ICMP flooding)
- IoT and IoV intrusions
- Web attacks (SQL Injection, XSS)
- Code vulnerabilities in software repositories

### 3.2 How LLMs Detect These Attacks

- Fine-tuning on cybersecurity datasets (e.g., CICIoT2023, EdgeIIoTset, Car-Hacking)
- Hybrid frameworks integrating BERT, BART, GANs, and MLPs [4–8]
- Leveraging semantic understanding of text for anomaly detection

### 3.3 What Makes LLMs Special?

- Ability to treat diverse cybersecurity data as sequences
- Near-perfect classification in specialized domains (IoV-BERT-IDS F1 ≈ 0.9997 [6])
- High adaptability to domain-specific fine-tuning

### 3.4 Specific Analysis: The Role and Impact of ChatGPT/OpenAI

OpenAI’s ChatGPT and its underlying GPT models (GPT-3, GPT-4, and InstructGPT) have played a pivotal role in reshaping the cybersecurity landscape, with measurable impacts on both defense and offense [2, 13].

**Scale and Adoption.** GPT-3 was built with 175 billion parameters, compared to just 1.5 billion for GPT-2. Subsequent models, such as InstructGPT, introduced supervised fine-tuning on top of unsupervised pre-training. ChatGPT reached a user base of 100 million within only two months of launch, highlighting unprecedented adoption rates [2, 13].

**Code Security Performance.** GPT-4 demonstrated significantly stronger code security capabilities than traditional tools, detecting roughly four times more vulnerabilities than leading static analyzers. Furthermore, GPT-3.5-turbo was leveraged to generate the FormAI dataset, containing over 112,000 compilable C programs for vulnerability research and detection studies [3, 6, 13].

**Limitations and Security Risks.** Despite these strengths, limitations persist. ChatGPT’s knowledge cutoff (typically September 2021) can lead to outdated outputs [13]. InstructGPT reduced toxic generations by ~25% compared to GPT-3, but still exhibited bias and reasoning errors [8]. Moreover, even advanced models such as GPT-4 and GPT-3.5-turbo remain vulnerable to **prompt injection attacks**, which bypass safety mechanisms and enable malicious use [1, 2]. Finally, GPT-4 consistently outperformed GPT-3.5-turbo on complex reasoning tasks (e.g., autonomous agent simulations), underscoring its superior ability to handle sophisticated scenarios [1].

------

## 4. LLM Architectures and Optimization Techniques for Cybersecurity

### 4.1 Classical ML/DL vs. LLM Approaches

- Traditional ML (Random Forest, SVM): accuracy ~78–81% [4]
- DL (CNN, RNN): accuracy ~94–95% [4]
- LLMs: accuracy up to 98–99.98% [4, 7, 8]

### 4.2 Advanced and Recent LLM-based Models

- **SecurityLLM**: 98% detection accuracy across 14 attack types [4]
- **GAN + Transformer (6G IoT)**: Great precision/recall for DDoS_ICMP/UDP [5]
- **IoV-BERT-IDS**: Accuracy 0.9999 in vehicle networks [6]
- **BARTPredict (IoT)**: Accuracy 98% [7]
- **Web Attack Detection (BERT+MLP)**: >99.98% accuracy with 0.4 ms detection time [8]
- **Phishing/Spam Detection (IPSDM)**: 99% accuracy [9]

------

## 5. Offensive Applications and Vulnerabilities of LLMs

### 5.1 Offensive Applications ("The Bad")

- **Social engineering & misinformation**: Most common offensive use (32+ papers [1, 2])
- **Malware generation**: Detection rates as low as 4% by AV [3, 13]
- **Phishing**: LLM-crafted phishing emails outperform human detection [2, 14]
- **Limitations**: No direct OS/hardware access, but can guide strategy [1, 13]

### 5.2 Inherent Vulnerabilities ("The Ugly")

- **Membership Inference Attacks (MIA):** Extract private info from model outputs [1, 11]
- **Prompt Injection:** Circumvents safeguards [1, 12]
- **DoS (Sponge Prompts):** Up to 200× increased resource usage [1, 12]
- **Data Quality Dependence:** Bias/incompleteness reduces robustness [13, 14]

------

## 6. Comparison with Existing Work

### A. Summary of Related Studies

A comprehensive review gathered 281 papers on LLMs and cybersecurity [1].

- 83 papers on **defensive applications ("The Good")**
- 54 papers on **offensive use ("The Bad")**
- 144 papers on **vulnerabilities and defenses ("The Ugly")**

### B. Metrics and Performance Comparisons

**Table 6.1: Comparison of Methods for Cyberattack Detection and Code Security**

| Method / Model                | Domain / Task                    | Dataset / Context        | Accuracy / F1-score                          | Key Insights                                      |
| ----------------------------- | -------------------------------- | ------------------------ | -------------------------------------------- | ------------------------------------------------- |
| Random Forest (Classical ML)  | General intrusion detection      | EdgeIIoTset              | 81% accuracy [4]                             | Baseline ML model, struggles with complex attacks |
| SVM (Classical ML)            | General intrusion detection      | EdgeIIoTset              | 78% accuracy [4]                             | Limited scalability                               |
| CNN (Deep Learning)           | Network intrusion detection      | EdgeIIoTset              | 95% accuracy [4]                             | Better sequence modeling                          |
| RNN (Deep Learning)           | Network intrusion detection      | EdgeIIoTset              | 94% accuracy [4]                             | Captures temporal patterns                        |
| **SecurityLLM (LLM)**         | General intrusion detection      | EdgeIIoTset              | **98% accuracy** [4]                         | Outperforms ML/DL baselines                       |
| **GAN + Transformer**         | 6G IoT threat hunting (DDoS)     | Simulated 6G IoT traffic | **95% precision/recall (ICMP/UDP)** [5]      | Great DDoS classification                         |
| **IoV-BERT-IDS (LLM)**        | Internet of Vehicles intrusion   | IVN-IDS / Car-Hacking    | **0.9996–0.9999 accuracy** [6]               | Extremely high performance                        |
| **BARTPredict (LLM)**         | IoT intrusion prediction         | CICIoT2023               | 98% accuracy [7]                             | Robust for IoT traffic                            |
| **BERT+MLP Hybrid**           | Web attack detection (SQLi, XSS) | HTTP anomaly dataset     | >**99.98% accuracy**, F1 >98.7% [8]          | Very low detection latency (0.4 ms)               |
| **IPSDM (BERT family)**       | Phishing / spam email detection  | Email datasets           | 99% accuracy, F1 = 0.98 [9]                  | Outperforms human/manual detection                |
| **GPT-3 vs. Commercial Tool** | Code vulnerability discovery     | Large codebase           | GPT-3: 213 vulns vs. Tool: 99 [6]            | GPT-3 uncovered >2× vulnerabilities               |
| **ChatRepair (LLM repair)**   | Bug fixing                       | Benchmark (337 bugs)     | 162 bugs fixed (~48%), cost ≈ $0.42 each [3] | Cost-efficient automated repair                   |

### C. Security and Application Insights

- **LLMs consistently outperform ML/DL** across intrusion detection tasks, especially when accuracy >98–99%.
- LLMs enable **near-real-time detection** (e.g., 0.4 ms per HTTP request [8]) while maintaining high accuracy.
- For **code vulnerability discovery**, GPT-3 detects >2× more issues than leading commercial tools [6].
- Despite defensive superiority, **LLMs can generate malware with AV detection rates as low as 4%** [13], highlighting their double-edged nature.

### 6.1 In-Depth Comparison: LLM Detection Performance vs. Traditional Models

LLM architectures, particularly Transformer-based approaches (e.g., BERT and fine-tuned GPT variants), show statistically superior performance over conventional ML and DL approaches in intrusion detection [1, 4].

**Table 6.2: Comparative Performance Metrics for Attack Detection**

| Attack Type / Metric           | LLM Model / Framework      | Performance Result                         | Traditional Comparison         | Source |
| ------------------------------ | -------------------------- | ------------------------------------------ | ------------------------------ | ------ |
| **Overall Detection Accuracy** | SecurityLLM (SecurityBERT) | 98% across 14 attack types                 | CNN: 95%, RNN: 94%             | [1, 4] |
| **DDoS (ICMP/UDP) Detection**  | Transformer-based CTH      | Precision, Recall, and F1 = **1.00**       | N/A (surpasses baselines)      | [5]    |
| **IoV Intrusion Prediction**   | BARTPredict (fine-tuned)   | 98% accuracy on CICIoT2023 dataset         | Competitive with DL IDS models | [7]    |
| **DDoS HTTP Recall**           | SecurityBERT               | 0.99 recall                                | N/A                            | [8]    |
| **Ransomware Detection (F1)**  | SecurityBERT               | Precision = 1.00, Recall = 0.40, F1 = 0.57 | N/A (lower effectiveness here) | [8]    |

Beyond raw metrics, results consistently show that LLM-based detection systems outperform BiLSTM, BiGRU, and other sequential DL architectures by capturing richer contextual patterns in traffic data [1, 4]. For instance, the IoV-BERT-IDS framework demonstrated robust generalization across datasets like CICIDS and BoT-IoT [6].

------

### 6.2 Quantifying the Efficiency of Offensive LLM Attacks

While LLMs provide superior defensive capabilities, they also introduce measurable risks when exploited for offensive purposes.

**A. Denial of Service (DoS) via Computational Exhaustion.**
 Due to their resource-heavy design, LLMs can be targeted with adversarial “sponge examples.” These malicious prompts increase computation and latency, reducing system availability by a factor of 10–200 [1].

**B. Malware Generation and Evasion.**
 Research shows LLMs can autonomously generate malware, polymorphic code, and phishing attacks [2, 13]. Detection rates for LLM-generated malware by antivirus systems vary between 4% and 55%, indicating concerning levels of evasion [12]. Furthermore, their natural language generation enables large-scale social engineering: one reported campaign used AI-crafted phishing messages to target over 600 Members of the UK Parliament [2].

------

### 6.3 Focus on ChatGPT/OpenAI: Specific Metrics and Impacts

OpenAI’s GPT family (GPT-3, InstructGPT, GPT-4) has set performance benchmarks for cybersecurity applications while raising unique challenges [2, 13].

**A. Scale and Adoption.**
 GPT-3 contains 175 billion parameters, a dramatic increase from GPT-2’s 1.5 billion. ChatGPT’s release demonstrated unprecedented adoption, reaching 100 million users in two months [2, 13].

**B. Vulnerability Detection and Repair.**

- **Vulnerability Detection:** GPT-4 identified nearly four times as many vulnerabilities as static code analyzers such as Snyk or Fortify [3, 6].
- **Repair Efficiency:** The ChatRepair framework successfully fixed 162 out of 337 bugs at an average cost of $0.42 per bug [3].

**C. Safety Alignment and Limitations.**

- **Toxicity Reduction:** InstructGPT reduced toxic outputs by ~25% compared to GPT-3 [8].
- **Residual Vulnerabilities:** Despite alignment, commercial LLMs remain susceptible to prompt injection attacks [1, 2].
- **Real-World Risks:** An empirical study on LLM-integrated applications revealed that 17 of 51 tested apps contained vulnerabilities—16 exploitable through Remote Code Execution (RCE) and 1 through SQL injection [13].

------

## 7. Common Techniques and Innovations

- Hybrid LLM + GAN models for IoT threat hunting
- Domain-specific LLM fine-tuning (IoV-BERT, IPSDM)
- Ultra-fast detection (0.4 ms per HTTP request [8])
- Automated vulnerability repair with cost-efficiency (ChatRepair [3])

------

## 8. Limitations, Challenges, and Future Directions

**Key Limitations:**

- Resource intensity (energy, computation)
- Vulnerabilities to adversarial attacks
- Over-reliance on training data quality

**Critical Observations:**

- State-of-the-art LLMs surpass traditional ML/DL in detection tasks
- Adversaries exploit LLM strengths for phishing and malware generation

**Future Research Directions:**

- Robust defense against prompt injection and DoS
- Standardized benchmarks for LLM-based intrusion detection
- Explainability for cybersecurity analysts
- Real-world deployment in 5G/6G, IoT, and critical infrastructures

------

## 9. Conclusion

LLMs have become central to cybersecurity, offering **state-of-the-art detection accuracy** across diverse domains such as web attack detection (>99.98%) and IoV intrusion detection (≈0.9999). They consistently outperform traditional ML/DL methods.
 However, their dual-use nature makes them equally powerful tools for adversaries, enabling scalable phishing, malware generation, and exploitation of model vulnerabilities.
 The future of secure LLM deployment requires balancing defensive applications with robust safeguards against offensive misuse.

------

## 10. References and Resources

[1] Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, and Yue Zhang. *A Survey on Large Language Model (LLM) Security and Privacy: The Good, The Bad, and The Ugly.* 

PII: S2667-2952(24)00014-X DOI: https://doi.org/10.1016/j.hcc.2024.100211 Reference: HCC 100211



[2] Maanak Gupta, CharanKumar Akiri, Kshitiz Aryal, Eli Parker, and Lopamudra Praharaj. *From ChatGPT to ThreatGPT: Impact of Generative AI in Cybersecurity and Privacy.*

arXiv:2307.00691v1 [cs.CR] 3 Jul 2023



[3] Gabriel de Jesus Coelho da Silva and Carlos Becker Westphall. *A Survey of Large Language Models in Cybersecurity.*

arXiv:2402.16968v1 [cs.CR] 26 Feb 2024



[4] Mohamed Amine Ferrag, Mthandazo Ndhlovu, Norbert Tihanyi, Lucas C. Cordeiro, Merouane Debbah, and Thierry Lestable. *Revolutionizing Cyber Threat Detection with Large Language Models.* (2023)

arXiv:2306.14263v1 [cs.CR] 25 Jun 2023



[5] Mohamed Amine Ferrag, Merouane Debbah, and Muna Al-Hawawreh. *Generative AI for Cyber Threat-Hunting in 6G-enabled IoT Networks.*

arXiv:2303.11751v1 [cs.CR] 21 Mar 2023



[6] Mengyi Fu, Pan Wang, Minyao Liu, Ze Zhang, and Xiaokang Zhou. *IoV-BERT-IDS: Hybrid Network Intrusion Detection System in IoV Using Large Language Models.*

DOI 10.1109/TVT.2024.3402366



[7] Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, and Yacine Ghamri-Doudane. *BARTPredict: Empowering IoT Security with LLM-Driven Cyber Threat Prediction.*

arXiv:2501.01664v1 [cs.CR] 3 Jan 2025



[8] Yunus Emre Seyyar, Ali Gökhan Yavuz, and Halil Murat Ünver. *An Attack Detection Framework Based on BERT and Deep Learning.*

*Digital Object Identifier 10.1109/ACCESS.2022.3185748*



[9] Xuemei Li and Huirong Fu. *SecureBERT and LLAMA 2 Empowered Control Area Network Intrusion Detection and Classification.*

arXiv:2311.12074v1 [cs.CR] 19 Nov 2023



[10] Kimia Ameri, Michael Hempel, Hamid Sharif, Juan Lopez Jr., and Kalyan Perumalla. *CyBERT: Cybersecurity Claim Classification by Fine-Tuning the BERT Language Model.*

2021 ,*1*,615–637. https://doi.org/10.3390/jcp1040031



[11] Victor Jüttner, Martin Grimmer, and Erik Buchmann. *ChatIDS: Explainable Cybersecurity Using Generative AI.*

arXiv:2306.14504v1 [cs.CR] 26 Jun 2023



[12] Matthew G. Gaber, Mohiuddin Ahmed, and Helge Janicke. *Malware Detection with Artificial Intelligence: A Systematic Literature Review.*

ACM Comput. Surv. 56, 6, Article 148 (January 2024), 33 pages. https://doi.org/10.1145/3638552



[13] Farzad Nourmohammadzadeh Motlagh, Mehrdad Hajizadeh, Mehryar Majd, Pejman Najafi, Feng Cheng, and Christoph Meinel. *Large Language Models in Cybersecurity: State-of-the-Art.*

arXiv:2402.00891v1 [cs.CR] 30 Jan 2024



[14] Additional supporting works (e.g., datasets, benchmarks) as cited in text.