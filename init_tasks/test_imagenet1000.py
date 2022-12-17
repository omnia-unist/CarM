from torchvision.datasets import ImageNet
import numpy as np
from PIL import Image

from torchvision import transforms
from trainer import Continual
import sys
from types import SimpleNamespace
import yaml
import argparse
import os
import torch
import random

def load_data(fpath, num_task, num_classes_per_task):
    data = []
    labels = []

    lines = open(fpath)
    
    for i in range(num_task * num_classes_per_task):
        data.append([])
        labels.append([])

    for line in lines:
        arr = line.strip().split()
        data[int(arr[1])].append(arr[0])
        labels[int(arr[1])].append(int(arr[1]))

    return data, labels

def experiment(final_params):

    runs = final_params.run

    for num_run in range(runs):
        print(f"#RUN{num_run}")
        
        if num_run == 0:
            if hasattr(final_params, 'filename'):
                org_filename = final_params.filename
            else:
                org_filename = ""
        
        final_params.filename = org_filename + f'run{num_run}'

        
        if num_run == 0 and hasattr(final_params, 'rb_path'):
            org_rb_path = final_params.rb_path
            print(final_params.rb_path)
        if hasattr(final_params, 'rb_path'):
            final_params.rb_path = org_rb_path + '/' + f'{final_params.filename}'
            os.makedirs(final_params.rb_path, exist_ok=True)

            print(final_params.rb_path)
        print(final_params.filename)

        
        
        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
            print(final_params.result_save_path)

        print(final_params.filename)


        if hasattr(final_params, 'seed_start'):
            if final_params.seed_start is not None:
                seed = final_params.seed_start + num_run
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                print("SEED : ", seed)
                
                final_params.seed = seed

        num_task = final_params.num_task_cls_per_task[0]
        num_classes_per_task = final_params.num_task_cls_per_task[1]

        class_order = np.arange(1000)

        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)
        # order from https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/options/data/imagenet1000_1order.yaml
        elif final_params.data_order == 'fixed':
            class_order = [54, 7, 894, 512, 126, 337, 988, 11, 284, 493, 133, 783, 192, 979, 622, 215, 240, 548, 238, 419, 274, 108, 928, 856, 494, 836, 473, 650, 85, 262, 508, 590, 390, 174, 637, 288, 658, 219, 912, 142, 852, 160, 704, 289, 123, 323, 600, 542, 999, 634, 391, 761, 490, 842, 127, 850, 665, 990, 597, 722, 748, 14, 77, 437, 394, 859, 279, 539, 75, 466, 886, 312, 303, 62, 966, 413, 959, 782, 509, 400, 471, 632, 275, 730, 105, 523, 224, 186, 478, 507, 470, 906, 699, 989, 324, 812, 260, 911, 446, 44, 765, 759, 67, 36, 5, 30, 184, 797, 159, 741, 954, 465, 533, 585, 150, 101, 897, 363, 818, 620, 824, 154, 956, 176, 588, 986, 172, 223, 461, 94, 141, 621, 659, 360, 136, 578, 163, 427, 70, 226, 925, 596, 336, 412, 731, 755, 381, 810, 69, 898, 310, 120, 752, 93, 39, 326, 537, 905, 448, 347, 51, 615, 601, 229, 947, 348, 220, 949, 972, 73, 913, 522, 193, 753, 921, 257, 957, 691, 155, 820, 584, 948, 92, 582, 89, 379, 392, 64, 904, 169, 216, 694, 103, 410, 374, 515, 484, 624, 409, 156, 455, 846, 344, 371, 468, 844, 276, 740, 562, 503, 831, 516, 663, 630, 763, 456, 179, 996, 936, 248, 333, 941, 63, 738, 802, 372, 828, 74, 540, 299, 750, 335, 177, 822, 643, 593, 800, 459, 580, 933, 306, 378, 76, 227, 426, 403, 322, 321, 808, 393, 27, 200, 764, 651, 244, 479, 3, 415, 23, 964, 671, 195, 569, 917, 611, 644, 707, 355, 855, 8, 534, 657, 571, 811, 681, 543, 313, 129, 978, 592, 573, 128, 243, 520, 887, 892, 696, 26, 551, 168, 71, 398, 778, 529, 526, 792, 868, 266, 443, 24, 57, 15, 871, 678, 745, 845, 208, 188, 674, 175, 406, 421, 833, 106, 994, 815, 581, 676, 49, 619, 217, 631, 934, 932, 568, 353, 863, 827, 425, 420, 99, 823, 113, 974, 438, 874, 343, 118, 340, 472, 552, 937, 0, 10, 675, 316, 879, 561, 387, 726, 255, 407, 56, 927, 655, 809, 839, 640, 297, 34, 497, 210, 606, 971, 589, 138, 263, 587, 993, 973, 382, 572, 735, 535, 139, 524, 314, 463, 895, 376, 939, 157, 858, 457, 935, 183, 114, 903, 767, 666, 22, 525, 902, 233, 250, 825, 79, 843, 221, 214, 205, 166, 431, 860, 292, 976, 739, 899, 475, 242, 961, 531, 110, 769, 55, 701, 532, 586, 729, 253, 486, 787, 774, 165, 627, 32, 291, 962, 922, 222, 705, 454, 356, 445, 746, 776, 404, 950, 241, 452, 245, 487, 706, 2, 137, 6, 98, 647, 50, 91, 202, 556, 38, 68, 649, 258, 345, 361, 464, 514, 958, 504, 826, 668, 880, 28, 920, 918, 339, 315, 320, 768, 201, 733, 575, 781, 864, 617, 171, 795, 132, 145, 368, 147, 327, 713, 688, 848, 690, 975, 354, 853, 148, 648, 300, 436, 780, 693, 682, 246, 449, 492, 162, 97, 59, 357, 198, 519, 90, 236, 375, 359, 230, 476, 784, 117, 940, 396, 849, 102, 122, 282, 181, 130, 467, 88, 271, 793, 151, 847, 914, 42, 834, 521, 121, 29, 806, 607, 510, 837, 301, 669, 78, 256, 474, 840, 52, 505, 547, 641, 987, 801, 629, 491, 605, 112, 429, 401, 742, 528, 87, 442, 910, 638, 785, 264, 711, 369, 428, 805, 744, 380, 725, 480, 318, 997, 153, 384, 252, 985, 538, 654, 388, 100, 432, 832, 565, 908, 367, 591, 294, 272, 231, 213, 196, 743, 817, 433, 328, 970, 969, 4, 613, 182, 685, 724, 915, 311, 931, 865, 86, 119, 203, 268, 718, 317, 926, 269, 161, 209, 807, 645, 513, 261, 518, 305, 758, 872, 58, 65, 146, 395, 481, 747, 41, 283, 204, 564, 185, 777, 33, 500, 609, 286, 567, 80, 228, 683, 757, 942, 134, 673, 616, 960, 450, 350, 544, 830, 736, 170, 679, 838, 819, 485, 430, 190, 566, 511, 482, 232, 527, 411, 560, 281, 342, 614, 662, 47, 771, 861, 692, 686, 277, 373, 16, 946, 265, 35, 9, 884, 909, 610, 358, 18, 737, 977, 677, 803, 595, 135, 458, 12, 46, 418, 599, 187, 107, 992, 770, 298, 104, 351, 893, 698, 929, 502, 273, 20, 96, 791, 636, 708, 267, 867, 772, 604, 618, 346, 330, 554, 816, 664, 716, 189, 31, 721, 712, 397, 43, 943, 804, 296, 109, 576, 869, 955, 17, 506, 963, 786, 720, 628, 779, 982, 633, 891, 734, 980, 386, 365, 794, 325, 841, 878, 370, 695, 293, 951, 66, 594, 717, 116, 488, 796, 983, 646, 499, 53, 1, 603, 45, 424, 875, 254, 237, 199, 414, 307, 362, 557, 866, 341, 19, 965, 143, 555, 687, 235, 790, 125, 173, 364, 882, 727, 728, 563, 495, 21, 558, 709, 719, 877, 352, 83, 998, 991, 469, 967, 760, 498, 814, 612, 715, 290, 72, 131, 259, 441, 924, 773, 48, 625, 501, 440, 82, 684, 862, 574, 309, 408, 680, 623, 439, 180, 652, 968, 889, 334, 61, 766, 399, 598, 798, 653, 930, 149, 249, 890, 308, 881, 40, 835, 577, 422, 703, 813, 857, 995, 602, 583, 167, 670, 212, 751, 496, 608, 84, 639, 579, 178, 489, 37, 197, 789, 530, 111, 876, 570, 700, 444, 287, 366, 883, 385, 536, 460, 851, 81, 144, 60, 251, 13, 953, 270, 944, 319, 885, 710, 952, 517, 278, 656, 919, 377, 550, 207, 660, 984, 447, 553, 338, 234, 383, 749, 916, 626, 462, 788, 434, 714, 799, 821, 477, 549, 661, 206, 667, 541, 642, 689, 194, 152, 981, 938, 854, 483, 332, 280, 546, 389, 405, 545, 239, 896, 672, 923, 402, 423, 907, 888, 140, 870, 559, 756, 25, 211, 158, 723, 635, 302, 702, 453, 218, 164, 829, 247, 775, 191, 732, 115, 331, 901, 416, 873, 754, 900, 435, 762, 124, 304, 329, 349, 295, 95, 451, 285, 225, 945, 697, 417]

        print(class_order)

        data_path = 'data/imagenet-1000'
        fpath = os.path.join(data_path, 'train.txt')

        data, labels = load_data(fpath, num_task, num_classes_per_task)
        
        continual = Continual(**vars(final_params))

        for task_id in range(num_task):
            label_st = task_id * num_classes_per_task
            for label in range(label_st, label_st + num_classes_per_task):
                print(class_order[label])
                for data_path in data[class_order[label]]:
                    data_path = os.path.join('/data/Imagenet', data_path)
                    with open(data_path,'rb') as f:
                        img = Image.open(f)
                        img = img.convert('RGB')
                    continual.send_stream_data(img, class_order[label], task_id)
                print(f"data {class_order[label]} added")

            print("train!")
            continual.train_disjoint(task_id)
