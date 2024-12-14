
import pandas as pd

# Load the uploaded CSV file
file_path = 'president_county_candidate.csv'  # Replace with your input file path
data = pd.read_csv(file_path)

# Summing up votes for each state and candidate
state_votes = data.groupby(['state', 'candidate', 'party']).agg({'total_votes': 'sum'}).reset_index()

# Determining the winner in each state
state_winners = state_votes.loc[state_votes.groupby('state')['total_votes'].idxmax()]
state_winners = state_winners[['state', 'candidate', 'party', 'total_votes']]

# Adding total state votes
total_state_votes = state_votes.groupby('state')['total_votes'].sum().reset_index()
state_results = pd.merge(state_winners, total_state_votes, on='state', suffixes=('_winner', '_total'))

# Save the state election results to a CSV file
output_file_path = 'state_election_results.csv'  # Replace with your desired output path
state_results.to_csv(output_file_path, index=False)

print(f'Results saved to {output_file_path}')
