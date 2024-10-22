# Doctor_Llama
An autoregressive large language model specifically fine-tuned on medical datasets to assist with healthcare-related tasks.

## Model Description

Doctor Llama is an autoregressive transformer decoder-only model built upon **LLaMA-3.2-3B-Instruct**, a supervised fine-tuned version of the **LLaMA-3.2-3B** base model.

**LLaMA-3.2-3B**
- A compact pre-trained model from the LLaMA 3 family
- Trained on raw text data in a self-supervised manner
- Uses next-token prediction as its primary objective function
- Designed to acquire general language understanding

**LLaMA-3.2-3B-Instruct**
- Supervised fine-tuned (SFT) version of LLaMA-3.2-3B
- Optimized to follow user instructions
- Enhanced instruction-following capabilities

**Doctor LLaMA**
- Further fine-tuned using SFT on medical datasets
- Specialized for medical domain tasks
- Optimized for healthcare applications

## Datasets

The model was trained on a small fraction of two primary datasets:

**AI-Medical-Chatbot**
- Contains patient-doctor question-answer interactions
- Simulates real clinical dialogue scenarios
- Helps model learn medical conversation patterns

**PubMedQA**
- Consists of Q&A pairs from medical citations and research statements
- Enhances scientific and research-based medical knowledge
- Incorporates evidence-based medical information

The datasets were processed, merged, and shuffled to enable multi-task learning, allowing the model to develop comprehensive medical knowledge across different contexts.

Additional datasets can be incorporated for specific capabilities depending on desired model behavior for exemple :
- MedQA: For enhanced medical question answering
- DrugBank: For pharmaceutical knowledge


## Training

The primary challenge in developing Doctor LLaMA was to achieve effective training within a restricted resource environment. To address this constraint, the 3B parameters version of the model, aligning with the growing trend of using compact yet powerful models that prioritize deployability and portability.

To optimize the training process, a 4-bit quantization combined with Low-Rank Adaptation (QLoRA) focuse on only updating Query and Value matrices during SFT phase. This approach significantly reduced memory requirements while maintaining model quality. Additional optimizations implemented in the training notebook enabled the model to fit and train effectively on affordable GPUs.

Further performance can be achieved by compiling the model in graph mode, which improved both training and inference speed.


## Limitations & Potential Improvements

This model can't be used for real-world scenarios as it was only trained on 10% of the datasets. For improvement:

**Datasets :** Use bigger portion of the dataset and add other task-specific datasets

**Domain Adaptation :**  Further pre-training of the base model on medical content before fine-tuning

**RLHF :** Perform reinforcement learning from human feedback to correct model behavior

**Evaluation :** Evaluate models with metrics like BLEU score, ROUGE score, perplexity and try different sizes and configurations

## Links
https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot

https://huggingface.co/datasets/qiaojin/PubMedQA

https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

https://huggingface.co/youzarsif/Doctor_Llama-3.2-3B-Instruct