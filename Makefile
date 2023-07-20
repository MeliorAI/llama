run-chat:
	torchrun --nproc_per_node 1 chat.py \
		--ckpt_dir llama-2-7b-chat/ \
		--tokenizer_path tokenizer.model \
		--max_seq_len 512 \
		--max_batch_size 4

run-qa:
	torchrun --nproc_per_node 1 qa_chat.py \
		--ckpt_dir llama-2-7b-chat/ \
		--tokenizer_path tokenizer.model \
		--max_seq_len 512 \
		--max_batch_size 4

index-doc:
	docker-compose up -d && python docqa.py $$doc
