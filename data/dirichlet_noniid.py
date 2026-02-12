import numpy as np

# ==========================================
# Mixed Partition (Non-IID for Train, IID for New)
# ==========================================
def partition_data(train_labels, test_labels, alpha, total_clients, num_new_clients):
    """
    先平均其中一份給new client, 剩下的用non-IID給訓練client
    """
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    num_train_clients = total_clients - num_new_clients
    classes = np.unique(train_labels)
    
    train_client_idcs = {k: [] for k in range(total_clients)}
    test_client_idcs = {k: [] for k in range(total_clients)}
    
    for k in classes:
        # 找出該類別所有 index
        train_idx_k = np.where(train_labels == k)[0]
        test_idx_k = np.where(test_labels == k)[0]
        
        np.random.shuffle(train_idx_k)
        np.random.shuffle(test_idx_k)
        
        # new client (IID)
        # 按 client 比例分配資料, e.g. 10個client有2個是new那就拿20%的資料給他們平分
        total_samples_train = len(train_idx_k)
        total_samples_test = len(test_idx_k)

        iid_ratio = num_new_clients / total_clients
        n_iid_train = int(total_samples_train * iid_ratio)
        n_iid_test = int(total_samples_test * iid_ratio)

        # IID pool
        train_idx_iid = train_idx_k[:n_iid_train]
        test_idx_iid = test_idx_k[:n_iid_test]
        
        # Non-IID pool
        train_idx_noniid = train_idx_k[n_iid_train:]
        test_idx_noniid = test_idx_k[n_iid_test:]
        
        # ---------------------------------------------
        # New Client
        # ---------------------------------------------
        if num_new_clients > 0:
            train_splits_iid = np.array_split(train_idx_iid, num_new_clients)
            test_splits_iid = np.array_split(test_idx_iid, num_new_clients)
            
            for i in range(num_new_clients):
                # new client id = last training client id+1
                cid = num_train_clients + i 
                train_client_idcs[cid] += train_splits_iid[i].tolist()
                test_client_idcs[cid] += test_splits_iid[i].tolist()

        # ---------------------------------------------
        # Training Clients (Dirichlet Non-IID)
        # ---------------------------------------------
        if num_train_clients > 0:
            # Dirichlet Proportions
            proportions = np.random.dirichlet(np.repeat(alpha, num_train_clients))
            
            # Normalize logic same as before
            proportions = np.array([p * (len(idx_j) < len(train_idx_noniid) / num_train_clients) 
                                    for p, idx_j in zip(proportions, [[]]*num_train_clients)])
            proportions_norm = proportions / proportions.sum()
            
            # Split Points
            train_split_pts = (np.cumsum(proportions_norm) * len(train_idx_noniid)).astype(int)[:-1]
            test_split_pts = (np.cumsum(proportions_norm) * len(test_idx_noniid)).astype(int)[:-1]
            
            train_splits_noniid = np.split(train_idx_noniid, train_split_pts)
            test_splits_noniid = np.split(test_idx_noniid, test_split_pts)
            
            for i in range(num_train_clients):
                cid = i 
                
                tr_idxs = train_splits_noniid[i].tolist()
                te_idxs = test_splits_noniid[i].tolist()
                
                if len(tr_idxs) > 0 and len(te_idxs) > 0:
                    train_client_idcs[cid] += tr_idxs
                    test_client_idcs[cid] += te_idxs
                else:
                    pass 

    return train_client_idcs, test_client_idcs