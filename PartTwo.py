import pandas as pd

df = pd.read_csv('p2-texts/hansard40000.csv')
# Rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
