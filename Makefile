run-chat:
	torchrun --nproc_per_node 1 chat.py \
		--ckpt_dir llama-2-7b-chat/ \
		--tokenizer_path tokenizer.model \
		--max_seq_len 512 \
		--max_batch_size 4

run-docqa:
	torchrun --nproc_per_node 1 doc_qa.py \
		--ckpt_dir llama-2-7b-chat/ \
		--tokenizer_path tokenizer.model \
		--max_seq_len 512 \
		--max_batch_size 4

index:
	docker-compose up -d && python index.py $$doc
