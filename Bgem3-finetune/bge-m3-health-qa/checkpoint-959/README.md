---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:61366
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-m3
widget:
- source_sentence: If a person looks strong and healthy, can they still have HIV?
    Is it dangerous?
  sentences:
  - Does if a person looks strong and healthy, can she still have HIV?
  - If a person with HIV is taking their medicine and they look healthy, are they
    still contagious?
  - Is anal sex more likely to spread HIV that vaginal sex?
- source_sentence: Which type of TB does not cause symptoms or health problems?
  sentences:
  - What is TB?
  - Are there specific indicators or signs that may suggest a person has TB?
  - What is the history of Human Immunodeficiency Virus?, please answer in detail.
- source_sentence: ኤችፒቪ (HPV)ወንዶችን ሊጎዳ ይችላል፣ እና ምን አይነት ሁኔታዎችን ሊያስከትል ይችላል?
  sentences:
  - Ngeri ki ze nnyinza okukozesa okwewala okuboola mukwano gwange alina Akawuka akaleeta
    Siriimu?
  - HPV ወንዶችን ሊያጠቃ ይችላል?
  - ሰዎች በኤችፒቪ እንዴት ይይዛሉ፣ እና ሊድን ይችላል?
- source_sentence: Maambukizi ya pamoja ya Virusi vya Ukimwi na TB ni nini?
  sentences:
  - Is it true that is there protection for men other than condoms?
  - Nini kupatikana na Virusi Vya Ukimwi?
  - Ina maana gani kuwa na maambukizi ya Virusi Vya Ukimwi na TB?
- source_sentence: Nze ne munnange tuyinza tutya okwekuuma obutafuna bulwadde bwa
    Trichomoniasis?
  sentences:
  - Some men believe they can stop the transfer of infection by withdrawing quickly
    right before ejaculation. Is this safe?
  - Tuyinza tutya okukendeeza ku bulwadde bwa Trichomoniasis?
  - Nsobola ntya okumanya oba omubeezi wange alina Obuwuka obuleeta Obulwadde bw’Ekikaba?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-m3

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) <!-- at revision 5617a9f61b028005a4858fdac845db406aefb181 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'embedding_dimension': 1024, 'pooling_mode': 'cls', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Nze ne munnange tuyinza tutya okwekuuma obutafuna bulwadde bwa Trichomoniasis?',
    'Tuyinza tutya okukendeeza ku bulwadde bwa Trichomoniasis?',
    'Nsobola ntya okumanya oba omubeezi wange alina Obuwuka obuleeta Obulwadde bw’Ekikaba?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7033, 0.4094],
#         [0.7033, 1.0000, 0.2613],
#         [0.4094, 0.2613, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 61,366 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                            | positive                                                                           | negative                                                                          |
  |:---------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                             | string                                                                            |
  | modality | text                                                                              | text                                                                               | text                                                                              |
  | details  | <ul><li>min: 9 tokens</li><li>mean: 22.29 tokens</li><li>max: 85 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 22.26 tokens</li><li>max: 125 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 20.19 tokens</li><li>max: 86 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                        | positive                                                                                                                     | negative                                                                                                                 |
  |:------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
  | <code>In what way can individuals best manage the situation if they are infected by the disease?</code>                       | <code>What are the best measures to do if you get to know you are HIV positive?</code>                                       | <code>Through which ways can someone realize that they are infected?</code>                                              |
  | <code>Can you explain the significance of HIV testing during pregnancy?</code>                                                | <code>Why should I test for HIV during pregnancy?</code>                                                                     | <code>What is involved in testing the baby for HIV?</code>                                                               |
  | <code>What is the best medication for Neisseria Gonorrhoeae Infection?, please answer this using simple medical terms.</code> | <code>What is the best treatment for Neisseria Gonorrhoeae Infection?, please answer this using simple medical terms.</code> | <code>What can you tell me about Neisseria Gonorrhoeae Infection?, please answer this using simple medical terms.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 1
- `learning_rate`: 2e-05
- `warmup_steps`: 0.1
- `gradient_accumulation_steps`: 8
- `fp16`: True
- `gradient_checkpointing`: True
- `dataloader_num_workers`: 2
- `remove_unused_columns`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 2e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 8
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 2
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: False
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0521 | 50   | 0.3435        |
| 0.1043 | 100  | 0.3384        |
| 0.1564 | 150  | 0.3058        |
| 0.2086 | 200  | 0.2723        |
| 0.2607 | 250  | 0.2595        |
| 0.3129 | 300  | 0.2591        |
| 0.3650 | 350  | 0.2411        |
| 0.4172 | 400  | 0.2317        |
| 0.4693 | 450  | 0.2313        |
| 0.5214 | 500  | 0.2152        |
| 0.5736 | 550  | 0.2154        |
| 0.6257 | 600  | 0.2081        |
| 0.6779 | 650  | 0.2150        |
| 0.7300 | 700  | 0.2128        |
| 0.7822 | 750  | 0.2133        |
| 0.8343 | 800  | 0.2084        |
| 0.8865 | 850  | 0.2011        |
| 0.9386 | 900  | 0.1988        |
| 0.9907 | 950  | 0.1903        |


### Training Time
- **Training**: 5.5 hours

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.5.1
- Transformers: 5.9.0
- PyTorch: 2.10.0+cu128
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->