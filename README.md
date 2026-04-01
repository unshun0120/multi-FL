# Multi-FL
## Step

### download datasets
```
python data/prepare_dataset.py
```

### experiment run
```
python main.py --seed=15698 --device=cuda:1
```

```
python main.py --device=cuda:1
```

```
python new_client_helpTrain_single_dataset.py --model_path=logs/2026-02-23_12-37-52/Ours/checkpoint.pth --device=cuda:1 --dataset_mode=super --new_client_epochs=30
```

```
python new_client_helpTest.py --model_path=logs/2026-02-23_12-37-52/Ours/checkpoint.pth --device=cuda:1 --dataset_mode=single --new_client_epochs=30
```

```
python new_client_train_scratch.py --model_path=logs/2026-02-23_12-37-52/Ours/checkpoint.pth --device=cuda:1 --dataset_mode=single --new_client_epochs=30
```

```
python utils/plot_comparison.py \
        --csvs helpTest:logs/2026-02-23_12-37-52/Ours/new_client_helpPredict_single_dataset_2026-02-24_11-31-58/accuracy_log.csv \
        --output_dir logs/2026-02-23_12-37-52/Ours/plot_comparison
```

```
python utils/plot_comparison.py \
        --csvs helpTrain:logs/2026-02-23_12-37-52/Ours/new_client_single_dataset_2026-02-24_13-57-12/accuracy_log.csv \
               helpTest:logs/2026-02-23_12-37-52/Ours/new_client_helpPredict_single_dataset_2026-02-24_20-43-29/accuracy_log.csv \
               baseline:logs/2026-02-23_12-37-52/Ours/new_client_baseline_single_dataset_2026-02-25_00-12-46/accuracy_log.csv \
        --output_dir logs/2026-02-23_12-37-52/Ours/plot_comparison
```

```
python utils/plot_comparison.py \
        --csvs helpTrain:logs/2026-02-23_12-37-52/Ours/new_client_super_dataset_2026-02-24_15-28-55/accuracy_log.csv \
               helpTest:logs/2026-02-23_12-37-52/Ours/new_client_helpPredict_super_dataset_2026-02-24_21-34-20/accuracy_log.csv \
               baseline:logs/2026-02-23_12-37-52/Ours/new_client_baseline_super_dataset_2026-02-25_00-55-23/accuracy_log.csv \
        --output_dir logs/2026-02-23_12-37-52/Ours/plot_comparison
```

```
python main.py --seed=15698 --algorithm=Local --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_cifar10=0 --num_train_emnist=0 --device=cuda:1
```

```
python train_generator.py --model_path=logs/2026-03-10_14-04-48/Local/
```

```
python label_mapping.py --model_path=logs/2026-03-16_15-55-40/Local/ --device=cuda:1 --seed=15698
```