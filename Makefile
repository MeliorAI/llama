run-chat:
	torchrun --nproc_per_node 1 chat.py \
		--ckpt_dir llama-2-7b-chat/ \
		--tokenizer_path tokenizer.model \
		--max_seq_len 512 \
		--max_batch_size 4

run-docqa:
	torchrun --nproc_per_node 1 -m docqa.chat \
		--ckpt_dir llama-2-7b-chat/ \
		--tokenizer_path tokenizer.model \
		--max_seq_len 1024 \
		--max_batch_size 4 \
		--n_results 2

index:
	docker-compose up -d && python -m docqa.index $$doc
