import pandas as pd
import numpy as np

# Učitavanje train.csv da bismo dobili imena kolona
train_df = pd.read_csv('train.csv')

# Generisanje novih podataka
np.random.seed(0)  # Postavljanje seed-a za reproduktivnost

# Generisanje slučajnih podataka za nove redove
num_rows = 200  # Broj novih redova
new_data = {
    'region': np.random.choice(train_df['region'], num_rows),
    'Year': np.random.choice(train_df['Year'], num_rows),
    'Population': np.random.randint(10000, 100000000, num_rows),
    'GDP per Capita': np.random.uniform(500, 50000, num_rows),
    'Urban Population': np.random.uniform(10, 100, num_rows),
    'Life Expectancy': np.random.uniform(50, 85, num_rows),
    'Surface Area': np.random.randint(1000, 10000000, num_rows),
    'Literacy Rate': np.random.uniform(50, 100, num_rows)
}

# Pretvaranje u DataFrame
new_df = pd.DataFrame(new_data)

# Čuvanje u test.csv
new_df.to_csv('test.csv', index=False)
