import copy


def identity_mapping(num_classes, offset=0):
    return {i: i + offset for i in range(num_classes)}


def shifted_mapping(local_ids, start_gid):
    return {local_id: start_gid + i for i, local_id in enumerate(local_ids)}


def list_mapping(global_ids):
    return {i: gid for i, gid in enumerate(global_ids)}


def get_cs_mapping_25():
    mnist = list_mapping([
        3, 0, 6, 4, 5, 1, 9, 7, 2, 8
    ])

    emnist = list_mapping([
        3, 10, 6, 11, 5, 1, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41,
        42, 43, 44, 45, 46, 47, 48, 9, 0, 7,
        49, 50, 51, 52, 53, 54, 2, 55, 4, 8,
        56, 57, 58, 59, 60, 61
    ])

    cifar10 = identity_mapping(10, offset=62)

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }


def get_feature_mapping_25():
    mnist = identity_mapping(10, offset=0)
    emnist = identity_mapping(62, offset=10)
    cifar10 = identity_mapping(10, offset=72)

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }


def get_single_mapping_25():
    mnist = list_mapping([
        1, 5, 4, 3, 2, 7, 6, 8, 0, 11
    ])

    emnist = list_mapping([
        9, 5, 4, 3, 2, 7, 6, 8, 0, 11,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 12, 26, 1, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 13, 56,
        57, 58, 59, 60, 61, 10
    ])

    cifar10 = list_mapping([
        1, 13, 62, 6, 63, 9, 64, 12, 10, 0
    ])

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

# ------------------------------
# Ours
# ------------------------------

def get_ours_5():
    mnist = identity_mapping(10, offset=0)
    
    emnist = identity_mapping(62, offset=0)

    cifar10 = identity_mapping(10, offset=62)

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

def get_ours_10():
    mnist = list_mapping([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ])

    emnist = list_mapping([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62
    ])

    cifar10 = identity_mapping(10, offset=63)

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

def get_ours_15():
    mnist = list_mapping([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ])

    emnist = list_mapping([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62
    ])

    cifar10 = identity_mapping(10, offset=63)

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

def get_ours_20():
    mnist = list_mapping([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ])

    emnist = list_mapping([
        10, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 0, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61
    ])

    cifar10 = identity_mapping(10, offset=62)

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

def get_ours_25():
    mnist = identity_mapping(10, offset=0)

    emnist = identity_mapping(62, offset=0)

    cifar10 = shifted_mapping([0, 1, 2, 3, 4, 6, 7, 8, 9], start_gid=62)
    cifar10[5] = 0

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }


def get_ours_5_noEntropy():
    mnist = list_mapping([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ])

    emnist = list_mapping([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61
    ])

    cifar10 = list_mapping([
        62, 63, 64, 65, 66, 8, 67, 68, 69, 70
    ])

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

# ------------------------------
# SlamDunk
# ------------------------------

def get_slam_dunk_mapping_5():
    mnist = list_mapping([
        13, 12, 6, 3, 7, 14, 11, 1, 5, 9
    ])

    emnist = list_mapping([
        13, 15, 6, 3, 7, 16, 17, 1, 5, 9,
        18, 19, 20, 21, 4, 22, 23, 24, 25, 10,
        26, 27, 28, 29, 30, 31, 32, 33, 14, 0,
        34, 35, 36, 37, 38, 39,
        40, 11, 2, 41, 8, 42, 43, 44, 45, 46,
        47, 12, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61
    ])

    cifar10 = list_mapping([
        4, 10, 8, 0, 62, 2, 63, 64, 65, 66
    ])

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

def get_slam_dunk_mapping_10():
    mnist = list_mapping([
        12, 13, 9, 5, 4, 11, 7, 1, 3, 6
    ])

    emnist = list_mapping([
        12, 13, 9, 5, 4, 11, 14, 1, 3, 6,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 0,
        34, 35, 36, 37, 38, 39,
        40, 7, 41, 42, 2, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 8, 55, 56,
        57, 58, 59, 60, 61, 10
    ])

    cifar10 = list_mapping([
        10, 2, 8, 62, 63, 0, 64, 65, 66, 67
    ])

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

def get_slam_dunk_mapping_15():
    mnist = list_mapping([
        9, 14, 10, 3, 8, 7, 2, 5, 6, 12
    ])

    emnist = list_mapping([
        9, 14, 10, 3, 8, 7, 11, 5, 6, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 4, 32, 33, 34,
        35, 36, 37, 13, 38, 39,
        40, 2, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 1, 51, 52, 53, 12, 54, 55, 56,
        57, 58, 59, 0, 60, 61
    ])

    cifar10 = list_mapping([
        62, 63, 64, 1, 13, 65, 0, 66, 11, 4
    ])

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

def get_slam_dunk_mapping_20():
    mnist = list_mapping([
        8, 11, 7, 0, 6, 9, 4, 2, 1, 10
    ])

    emnist = list_mapping([
        12, 13, 7, 0, 6, 14, 3, 2, 1, 10,
        15, 16, 17, 18, 19, 20, 4, 21, 22, 23,
        24, 25, 26, 27, 28, 5, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 11, 47,
        48, 49, 50, 51, 8, 52, 53, 54, 9, 55,
        56, 57, 58, 59, 60, 61
    ])

    cifar10 = list_mapping([
        62, 3, 5, 63, 64, 65, 66, 67, 68, 69
    ])

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }

def get_slam_dunk_mapping_25():
    mnist = list_mapping([
        13, 15, 12, 2, 8, 11, 4, 6, 0, 14
    ])

    emnist = list_mapping([
        13, 16, 12, 2, 8, 11, 10, 6, 0, 14,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        3, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 9, 39, 40,
        41, 4, 42, 43, 5, 44, 45, 46, 47, 48,
        49, 15, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 1, 61, 7
    ])

    cifar10 = list_mapping([
        7, 10, 3, 62, 63, 9, 1, 64, 5, 65
    ])

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }


def get_class_name_mapping():
    mnist = identity_mapping(10, offset=0)

    emnist = identity_mapping(62, offset=0)

    cifar10 = identity_mapping(10, offset=62)

    return {
        "MNIST": mnist,
        "EMNIST": emnist,
        "CIFAR10": cifar10,
    }


def get_mapping(name):
    mappings = {
        "ours_5": get_ours_5,
        "ours_10": get_ours_10,
        "ours_15": get_ours_15,
        "ours_20": get_ours_20,
        "ours_25": get_ours_25,

        "ours_5_noEntropy": get_ours_5_noEntropy,
        
        "class_name": get_class_name_mapping,
        "cs_mapping_25": get_cs_mapping_25,
        "feature_mapping_25": get_feature_mapping_25,
        "single_mapping_25": get_single_mapping_25,

        "slam_dunk_mapping_5": get_slam_dunk_mapping_5,
        "slam_dunk_mapping_10": get_slam_dunk_mapping_10,
        "slam_dunk_mapping_15": get_slam_dunk_mapping_15,   
        "slam_dunk_mapping_20": get_slam_dunk_mapping_20,
        "slam_dunk_mapping_25": get_slam_dunk_mapping_25,
    }

    return copy.deepcopy(mappings[name]())