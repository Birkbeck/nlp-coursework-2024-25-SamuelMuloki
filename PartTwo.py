import pandas as pd

df = pd.read_csv('p2-texts/hansard40000.csv')
# Rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
# Remove any rows where the value of the ‘party’ column is not one of the
# four most common party names, and remove the ‘Speaker’ value
top_parties = df['party'].value_counts().drop('Speaker').nlargest(4).index
df = df[df['party'].isin(top_parties)]
