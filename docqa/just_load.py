import os

from llama import Llama
from rich import print as rprint


from docqa.constants import DEFAULT_TOKENIZER_PATH
from docqa.constants import DEFAULT_TEMPERATURE
from docqa.constants import DEFAULT_CKPT_DIR
from docqa.constants import DEFAULT_TOP_P
from docqa.constants import DEFAULT_MAX_SEQ_LEN
from docqa.constants import DEFAULT_MAX_BATCH_SIZE
from docqa.constants import DEFAULT_MAX_GEN_LEN


def main():

    if not os.path.exists(DEFAULT_CKPT_DIR):
        rprint(f"[red] Checkpoint '{DEFAULT_CKPT_DIR}' not found!")
        exit(1)

    if not os.path.exists(DEFAULT_TOKENIZER_PATH):
        rprint(f"[red] Tokenizer '{DEFAULT_TOKENIZER_PATH}' not found!")
        exit(1)

    generator = Llama.build(
        ckpt_dir=DEFAULT_CKPT_DIR,
        tokenizer_path=DEFAULT_TOKENIZER_PATH,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        max_batch_size=DEFAULT_MAX_BATCH_SIZE,
    )
    # Generate
    results = generator.chat_completion(
        [[{"role": "user", "content": "Hello 👋🏼"}]],  # type: ignore
        max_gen_len=DEFAULT_MAX_GEN_LEN,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
    )
    rprint(results)


if __name__ == "__main__":
    main()
