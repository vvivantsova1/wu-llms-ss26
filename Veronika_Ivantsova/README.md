

MODELS

I used two different models for this project.
The first model was GPT-4o-mini, which is as the baseline model. I accessed it through the OpenAI API. For generation, I used a temperature of 0 to make the outputs deterministic and set the maximum output length to 200 tokens.
The second model was Falcon-RW-1B, that I fine-tuned for the task. During inference.

FINE-TUNING
I fine-tuned the Falcon-RW-1B model using LoRA (Low-Rank Adaptation).
For the fine-tuning data, I created a custom dataset with the help of ChatGPT. The dataset consists of 100 question–answer pairs.

The main hyper-parameters I used:

- LoRA rank: 16 (controls how much the model can adapt to the new task. I chose a moderate value to give the model enough flexibility without making training too heavy.)
- LoRA alpha: 32(scales the LoRA updates and helps stabilize learning.)
- Target modules: query_key_value(This tells the model which part to train. In this case, we only train the attention part of Falcon, so it learns better which words in a question are important.)
- Dropout: 0.05(helpes reduce overfitting.)
- Epochs: 4(the model sees the full dataset four times)
- Batch size: 1 per device(small because of GPU memory limits in Google Colab)
- Gradient accumulation steps: 4 (so effective batch size is 4)
- Learning rate: 2e-4(how fast the model learns from new data.)
- Max sequence length: 512
- Quantization: 4-bit via bitsandbytes(reduces memory usage)
- Precision: fp16(speeds up training and further reduces memory usage compared to full precision)

CHALLENGES

At first, I tried fine-tuning with LLaMA, but it failed because the model was too large for the GPU memory available in Google Colab. Because of that, I switched to the smaller Falcon-RW-1B model.
Even with Falcon, I still ran into memory issues. I initially used larger batch sizes, but kept getting out-of-memory errors, so I reduced the batch size to 1 and used gradient accumulation instead. To prevent crashes, I also had to manually clear the GPU memory using torch.cuda.empty_cache() and gc.collect().
Another challenge was that Falcon-RW-1B was mainly trained on English text, so it struggled with German special characters like umlauts. This led to incorrect outputs.

EVALUATION

I evaluated both models using three metrics.

ROUGE-1 checks how many individual words overlap between the model’s answer and the reference answer.
ROUGE-L looks at the longest matching sequence of words in both texts.
BERTScore uses BERT embeddings to compare meaning, so it can understand if two answers are similar even when they are written in different words.

The evaluation was run on all 643 questions from the test set. The reference answers were taken from dataset_clean_fixed.csv, and the matching was done using the question ID.

RESULTS

Overall, GPT-4o-mini clearly performs better across all evaluation metrics. It was able to answer all 643 questions, while Falcon-RW-1B-FT failed to produce answers for 21 of them. It also achieved higher scores on all metrics.
The fine-tuned Falcon model performs noticeably worse. In many cases, its answers are very short or incomplete, and sometimes it just repeats parts of the question instead of giving a proper explanation. This is most likely because it was trained on a very small dataset of only 100 question–answer pairs, which is not enough to fully learn the complexity of the test questions

The fine-tuned Falcon model performs worse. In many cases, its answers are short or incomplete, and sometimes it just repeats parts of the question instead of giving a proper explanation. This is most likely because it was trained on a very small dataset of only 100 question–answer pairs, which is not enough to fully learn the complexity of the test questions. Another issue is that it struggles with German umlauts, which can lead to incorrect outputs.

We can try to solve these issues by making sure the dataset is properly UTF-8 encoded and handling German characters correctly (for example by converting them like ä to ae). Another improvement would be to tune the hyperparameters more carefully, such as adjusting the learning rate or experimenting with different LoRA settings. Performance could also be improved by using a stronger base model like LLaMA or Mistral.
