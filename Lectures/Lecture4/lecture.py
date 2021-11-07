import pandas as pd
import numpy as np

pupulation = pd.Series({
    'New York': 123,
    'Cali': 123123,
    'Almaty': 63456
})

area = pd.Series({
    'New York': 678678,
    'Cali': 4567567,
    'Almaty': 6354674456
})

alias = pd.Series({
    'New York': 'NY',
    'Cali': 'Cali',
    'Almaty': 'ATY',
    'Atlanta': 'ATL'
})

data_frame = pd.DataFrame({
    'pupulation': pupulation,
    'area': area,
    'alias': alias
})

# print(data_frame)

dummy = pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c'])

# print(dummy)

A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])

# print(A)

# print(pupulation[(pupulation > 100) & pupulation < 1000000])

# print(pupulation.iloc[2])

# print(data_frame['area'] is data_frame.area)

data_frame['density'] = data_frame.pupulation / data_frame.area

# print(data_frame)

# list = [1, 23, 43, 56, 5]

# print(list.pop())
# print(list)

# print(data_frame)
# print(data_frame.T)

# print(data_frame.iloc[0:3, 1:3])
# print(data_frame['Almaty': 'Atlanta'])
# print(data_frame.iloc[0, 2])

vals = np.array([1, np.nan, 2, 4])
print(vals.sum())

