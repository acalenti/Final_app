import pandas as pd


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 3500)
pd.set_option('display.max_colwidth', 5000)
pd.set_option('display.width', 15000)


shape_file = pd.read_csv('az_state_new.csv')
zip_with_count = pd.read_csv('az_zip_to_county.csv')

final_df = pd.merge(shape_file, zip_with_count, on='BillingZipCode', how='left')


print(shape_file.ix[5:,7])