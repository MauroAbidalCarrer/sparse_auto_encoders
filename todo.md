- Better understand the OpenAI paper:
    - L(N) and L(c) 
    - N2G: Seems a bit comlicated it would take a long time to understand it so I will try to ignore it for now.
    - metrics:
        - Downstrean loss: Difference (KL divergence or cross entropy) between tokens logits emitted from reconstructed residual stream through remaining layers.
        - Probe loss: Logistic regression loss with token latent and labeled dataset "belonging".
        - Explainability: Based on N2G so I will ignore it for now.
        - Ablation loss sparsity loss:
          Let's break it down:
          - Ablation loss:
            Downstream loss computed over T token prediction instead of just the next token.
            Here the residual stream is reconstructed from latents with one zeroed element.
            We then take the mean difference (kl divergence or cross entropy) of the TxV differne matrix where T is the number of predicted tokens and V the vocab size.
          - Ablation loss sparsity: Sparsity of the ablation loss, how spread the loss is accross the sequences, high sparsity means that the ablation affects just a few localized tokens, low sparsity means that the ablation affects a wide spread part of the sequence.
          - Not sure if I will use this one in the loss since I actually would like to have high abstraction level features in the latent such as safe/unsafe question, racist/non-racist subject, ect..
        - Auxiliary loss:
        - Loss normalization
    - architecture:
        - "We scale decoder latent directions to be unit norm at initialization (and also after each training step)" open ai paper A.1 Initialization
        - rescale_acts_by_decoder_norm from SAE lens SAETraining class
        - Why have pre_bias and sbstract it to the input then add it to the decoded output?
        - Why no bias in encoder but bias in decoder?
    - model optimization:
        - decoder columns unit normalazation
        - parallel gradient to decoder removal?

- Better match the OpenAI implementation:
    - data:
        - 10x number of tokens in dataset.
        - Add more general dataset with, ideally, labeled questions.
    - architecture
        - Build on top of GPT-2
        - Increase latent size, k
        - Use activations in later layers like the ~15th (out of 20) layer for 7B unscensored model, layer 6(?) of GPT-2.
        - decoder normalization
    - evaluation:
        - Use downstream loss
        - Use normalized MSE
        - Still use L1 loss?
        - Auxiliary loss
    - model optimization
        - Use gradient clipping 
        - Increase batch size to maximize GPU efficiency
    - processing optimization:
        - Use sparse/dense matmul kernels

next steps:
- Use GPT-2 (done)
- Improve dataset (done)
- Use OpenAI SAE architecture (done)
    - Match latent size and k
- Use same losses