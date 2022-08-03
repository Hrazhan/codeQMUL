from datasets import load_dataset
from tqdm import tqdm

from arguments import TokenizerTrainingArguments
from transformers import AutoTokenizer, HfArgumentParser
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
 

# Iterator for Training
def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, args.n_examples, batch_size)):
        yield [next(iter_dataset)[args.text_column] for _ in range(batch_size)]


# # Configuration
parser = HfArgumentParser(TokenizerTrainingArguments)
args = parser.parse_args()

# Base tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(args.base_tokenizer,
    unk_token="<|unkoftext|>", bos_token="<|beginoftext|>", eos_token="<|endoftext|>", pad_token="<|padoftext|>",
    model_max_length=2048,
    # kwargs={ "model_max_length": 2048}
)
# tokenizer.model_max_length = 2048
base_vocab = list(bytes_to_unicode().values())

# Load dataset
dataset = load_dataset(args.dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)


# Training and saving
tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(), vocab_size=args.vocab_size, initial_alphabet=base_vocab,
)
tokenizer.save_pretrained(args.tokenizer_name, push_to_hub=args.push_to_hub)





## Training the tokenizer from scratch
# # Initialize a tokenizer
# tokenizer = Tokenizer(models.BPE())

# # # Customize pre-tokenization and decoding
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
# tokenizer.decoder = decoders.ByteLevel()
# tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
# tokenizer.normalizer = NFKC()
# tokenizer.enable_truncation(max_length=2048)

# # # And then train
# trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=["<|endoftext|>", "<|padding|>"])
# tokenizer.train_from_iterator(
#     batch_iterator(), trainer=trainer
# )
# # tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,model_max_length=2048)
# # # And Save it
# new_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer, max_length=2048)
# new_tokenizer.save_pretrained(args.tokenizer_name, push_to_hub=args.push_to_hub)
print(f'tokenizer saved at {str(args.tokenizer_name)}')