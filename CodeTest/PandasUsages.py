import pandas as pd
data = [[1, 2, 3], [4, 5, 6]]
index = ['a', 'b']#行号
columns = ['c', 'd', 'e']#列号
df = pd.DataFrame(data,
                  index=index,
                  columns=columns
                  )  #  生成一个数据框
print("df :", df)
print("loc : ", df.iloc[0][2])