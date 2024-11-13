# Example of command to run

CUDA_VISIBLE_DEVICES=0 python3 eval_Lazy_XMem_with_lazy_agent.py --output ./output_test --model './saves/XMem-s012.pth' --dataset lvos-val --Mem_upd 0.8 --derivatX 0.5 --save_prompts --working_upd --update_mem_with_prompt --pseudo_derivatX 0.2 --mode 1 --using_mask_in_pseudo # --debug # --update_mem_with_pseudo_prompt
