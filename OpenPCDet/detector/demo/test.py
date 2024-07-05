import numpy as np

# 定义数组1和数组2
array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array2 = np.array([[1, 2, 3], [6,5,4]])

# 将数组1转化为一组元组集合，以便快速查找
set_array1 = {tuple(row) for row in array1}

# 检查数组2中的每个一维数组是否都包含在数组1中
def arrays_are_contained(set_array1, array2):
    for sub_array in array2:
        if tuple(sub_array) not in set_array1:
            return False
    return True

all_contained = arrays_are_contained(set_array1, array2)

print(all_contained)  # 输出: True