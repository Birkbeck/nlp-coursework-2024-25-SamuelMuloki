import pandas as pd

df = pd.read_csv('p2-texts/hansard40000.csv')
# Rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
# Remove any rows where the value of the ‘party’ column is not one of the
# four most common party names, and remove the ‘Speaker’ value
top_parties = df['party'].value_counts().drop('Speaker').nlargest(4).index
df = df[df['party'].isin(top_parties)]
# Remove any rows where the value in the ‘speech_class’ column is not ‘Speech’.
df = df[df['speech_class'] == 'Speech']
# Remove any rows where the text in the ‘speech’ column is less than 1000
# characters long.
df = df[df['speech'].str.len() >= 1000]
