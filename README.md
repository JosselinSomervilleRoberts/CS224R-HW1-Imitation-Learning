# How to run

## Behavior cloning

### Ant environment

```bash
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 --eval_batch_size 5000
```

### Half Cheetah environment

```bash
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v4 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_HalfCheetah-v4.pkl \
--video_log_freq -1 --eval_batch_size 5000
```

### Hopper environment

```bash
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Hopper.pkl \
--env_name Hopper-v4 --exp_name bc_hopper --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Hopper-v4.pkl \
--video_log_freq -1 --eval_batch_size 5000
Loading expert policy from... cs224r/policies/experts/Hopper.pkl
```

### Walker2d environment

```bash
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 --exp_name bc_walker --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Walker2d-v4.pkl \
--video_log_freq -1 --eval_batch_size 5000
```

## DAgger

### Ant environment

```bash
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 --eval_batch_size 5000
```

### Half Cheetah environment

```bash
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v4 --exp_name dagger_halfcheetah --n_iter 10 \
--do_dagger \
--expert_data cs224r/expert_data/expert_data_HalfCheetah-v4.pkl \
--video_log_freq -1 --eval_batch_size 5000
```

### Hopper environment

```bash
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Hopper.pkl \
--env_name Hopper-v4 --exp_name dagger_hopper --n_iter 10 \
--do_dagger \
--expert_data cs224r/expert_data/expert_data_Hopper-v4.pkl \
--video_log_freq -1 --eval_batch_size 5000
```

### Walker2d environment

```bash
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 --exp_name dagger_walker --n_iter 10 \
--do_dagger \
--expert_data cs224r/expert_data/expert_data_Walker2d-v4.pkl \
--video_log_freq -1 --eval_batch_size 5000
```