import pandas as pd

def read_mouse_data(file_path):
    # Read the excel file
    df = pd.read_excel(file_path)

    # Reshape the data
    columns_to_remove = [3, 4, 7, 8, 11, 12]  
    df = df.drop(df.columns[columns_to_remove], axis=1)
    df = df.dropna(how='all')
    rows_to_remove = [21, 43]
    df = df.reset_index(drop=True)
    df = df.drop(rows_to_remove)
    df = df.reset_index(drop=True)
    df_subset1 = df.iloc[0:21]
    df_subset1 = df_subset1.reset_index(drop=True)
    df_subset2 = df.iloc[21:42]
    df_subset2 = df_subset2.reset_index(drop=True)
    df_subset3 = df.iloc[42:63]
    df_subset3 = df_subset3.reset_index(drop=True)
    df = pd.concat([df_subset1, df_subset2, df_subset3], axis=1)
    df = df.iloc[:, :-4]
    df.columns = [f'Column_{i}' for i in range(df.shape[1])]
    df = df.drop(df.columns[[9, 18]], axis=1)
    df = df.drop(0)
    
    # Rename the columns
    df.columns = ['time', 
                'mouse1_l', 'mouse1_wc',
                'mouse2_l', 'mouse2_wc',
                'mouse3_l', 'mouse3_wc',
                'mouse4_l', 'mouse4_wc',
                'mouse5_l', 'mouse5_wc',
                'mouse6_l', 'mouse6_wc',
                'mouse7_l', 'mouse7_wc',
                'mouse8_l', 'mouse8_wc',
                'mouse9_l', 'mouse9_wc',
                'mouse10_l', 'mouse10_wc',]
    

    return df

