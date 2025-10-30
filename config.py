## Create a config class
import os
from transformers import T5TokenizerFast, T5ForConditionalGeneration

class config:
    num_workers = os.cpu_count()
    n_epochs = 3
    batch_size = 8

    dataset_path = "./data/news_summary.csv"

    text_token_max_length = 1024
    summary_token_max_length = 256

    learning_rate = 0.0001

    use_gpu = False

    # T5 transfromer
    t5_model_path = "t5-base"

    # T5 Fast Tokenizer
    t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_path)

    # T5 pretrained model
    t5_pretrained_model = T5ForConditionalGeneration.from_pretrained(t5_model_path, return_dict = True)