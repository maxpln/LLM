# Large Languge models

## LLM for text

| language model | Authors | Release Date | Checkpoints | Params | Context Length | Nb tokens trained | Try it | Paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bloom | [BigScience](https://bigscience.huggingface.co/) | 2022/07 | [Bloom](https://huggingface.co/docs/transformers/model_doc/bloom) | 0.560, 1.1, 1.7, 3, 7.1, 176  | 2048 | 350B | ? | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) |
| Falcon| [TII](https://www.tii.ae/)  | 2023/05 | [falcon-7b](https://huggingface.co/tiiuae/falcon-7b), [falcon-40b](https://huggingface.co/tiiuae/falcon-40b), [falcon-180b](https://huggingface.co/tiiuae/falcon-180B)  | 7, 40, 180 | 2048 | 1500B-3500B | ? | [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/abs/2306.01116) |
| LLaMA 2| Meta AI  | 2023/07 | [LLaMA 2 weight](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) | 7, 13, 70 | 4000 | 2000B | [HuggingChat](https://huggingface.co/blog/llama2#demo) | [ LLaMA: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) |

## LLM for code

| language model | Authors | Release Date | Checkpoints | Params | Context Length | Nb tokens trained | Try it | Paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SantaCoder | [BigCode](https://www.bigcode-project.org/) | 2023/01 | [santacoder](https://huggingface.co/bigcode/santacoder) | 1.1 | 2048 | 236B | [SantaCoder](https://github.com/slai-labs/get-beam/tree/main/examples/santacoder) | [SantaCoder: don't reach for the stars!](https://arxiv.org/abs/2301.03988)


| StarCoder | --- | 2023/05 | [starcoder](https://huggingface.co/bigcode/starcoder) | [StarCoder: A State-of-the-Art LLM for Code](https://huggingface.co/blog/starcoder), [StarCoder: May the source be with you!](https://drive.google.com/file/d/1cN-b9GnWtHzQRoE7M7gAEyivY0kl4BYs/view) | 1.1-15 | [8192](https://huggingface.co/bigcode/starcoder#model-summary)                         | [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |                                                                                    |
| StarChat Alpha | --- | 2023/05 | [starchat-alpha](https://huggingface.co/HuggingFaceH4/starchat-alpha) | [Creating a Coding Assistant with StarCoder](https://huggingface.co/blog/starchat-alpha) | 16 | [8192](https://huggingface.co/bigcode/starcoder#model-summary)  | [OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) |                                                                                           |
| Replit Code | --- | 2023/05 | [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) | [Training a SOTA Code LLM in 1 week and Quantifying the Vibes — with Reza Shabani of Replit](https://www.latent.space/p/reza-shabani#details) | 2.7 | [infinity? (ALiBi)](https://huggingface.co/replit/replit-code-v1-3b#model-description) | CC BY-SA-4.0 | [Replit-Code-v1-3B](https://github.com/slai-labs/get-beam/tree/main/examples/replit-code) |
| CodeGen2 | --- | 2023/04 | [codegen2 1B-16B](https://github.com/salesforce/CodeGen2) | [CodeGen2: Lessons for Training LLMs on Programming and Natural Languages](https://arxiv.org/abs/2305.02309) | 1 - 16 | [2048](https://arxiv.org/abs/2305.02309) | [Apache 2.0](https://github.com/salesforce/CodeGen2/blob/main/LICENSE)|                                                                                           |
| CodeT5+ | --- | 2023/05 | [CodeT5+](https://github.com/salesforce/CodeT5/tree/main/CodeT5+)     | [CodeT5+: Open Code Large Language Models for Code Understanding and Generation](https://arxiv.org/abs/2305.07922) | 0.22 - 16 | [512](https://arxiv.org/abs/2305.07922)                                                                                | [BSD-3-Clause](https://github.com/salesforce/CodeT5/blob/main/LICENSE.txt)                                                           | [Codet5+-6B](https://github.com/slai-labs/get-beam/tree/main/examples/codeT5%2B)          |
| XGen-7B | --- | 2023/06 | [XGen-7B-8K-Base](https://huggingface.co/Salesforce/xgen-7b-8k-base) | [Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length](https://blog.salesforceairesearch.com/xgen/) | 7 | [8192](https://huggingface.co/Salesforce/xgen-7b-8k-base/blob/main/config.json) | [Apache 2.0](https://github.com/salesforce/xgen/blob/main/LICENSE) | 
| CodeGen2.5 | --- | 2023/07 | [CodeGen2.5-7B-multi](https://huggingface.co/Salesforce/codegen25-7b-multi) | [CodeGen2.5: Small, but mighty](https://blog.salesforceairesearch.com/codegen25/) | 7 | [2048](https://huggingface.co/Salesforce/codegen25-7b-multi/blob/main/config.json) | [Apache 2.0](https://huggingface.co/Salesforce/codegen25-7b-multi/blob/main/README.md) | 
| DeciCoder-1B | --- | 2023/08 | [DeciCoder-1B](https://huggingface.co/Deci/DeciCoder-1b#how-to-use) | [Introducing DeciCoder: The New Gold Standard in Efficient and Accurate Code Generation](https://deci.ai/blog/decicoder-efficient-and-accurate-code-generation-llm/) | 1.1 | [2048](https://huggingface.co/Deci/DeciCoder-1b#model-architecture) | Apache 2.0 |  [DeciCoder Demo](https://huggingface.co/spaces/Deci/DeciCoder-Demo)|
| Code Llama  | --- | 2023 | [Inference Code for CodeLlama models]([https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://github.com/facebookresearch/codellama)) | [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)     | 7 - 34       | [4096](https://scontent-zrh1-1.xx.fbcdn.net/v/t39.2365-6/369856151_1754812304950972_1159666448927483931_n.pdf?_nc_cat=107&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=wURKmnWKaloAX-ib5XW&_nc_ht=scontent-zrh1-1.xx&oh=00_AfAN1GB2K_XwIz54PqXTr-dhilI3CfCwdQoaLMyaYEEECg&oe=64F0A68F)  | [Custom](https://github.com/facebookresearch/llama/blob/main/LICENSE) Free if you have under 700M users and you cannot use LLaMA outputs to train other LLMs besides LLaMA and its derivatives   | [HuggingChat](https://huggingface.co/blog/codellama) | 







