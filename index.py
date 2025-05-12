from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/owid-covid-data.csv'

import pandas as pd

df = pd.read_csv(file_path)
# Check column names
print(df.columns)

# Preview the first 5 rows
print(df.head())

# Identify missing values
print(df.isnull().sum())

# Filter countries of interest
countries = ['Austria', 'India', 'Angola']
df_filtered = df[df['location'].isin(countries)]

# Drop rows with missing dates or total cases
df_filtered = df_filtered.dropna(subset=['date', 'total_cases'])

# Convert 'date' to datetime
df_filtered['date'] = pd.to_datetime(df_filtered['date'])

# Fill or interpolate missing numeric values
numeric_cols = df_filtered.select_dtypes(include='number').columns
df_filtered[numeric_cols] = df_filtered[numeric_cols].interpolate(method='linear', limit_direction='forward')

# Confirm changes
print(df_filtered.info())
print(df_filtered.head())



import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
sns.set_palette("Set2")

# Plot total cases over time
plt.figure(figsize=(12, 6))
for country in countries:
    data = df_filtered[df_filtered['location'] == country]
    plt.plot(data['date'], data['total_cases'], label=country)

plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.tight_layout()
plt.show()



# Plot total deaths over time
plt.figure(figsize=(12, 6))
for country in countries:
    data = df_filtered[df_filtered['location'] == country]
    plt.plot(data['date'], data['total_deaths'], label=country)

plt.title('Total COVID-19 Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.legend()
plt.tight_layout()
plt.show()



# Daily New Cases Comparison
plt.figure(figsize=(12, 6))
for country in countries:
    data = df_filtered[df_filtered['location'] == country]
    plt.plot(data['date'], data['new_cases'], label=country)

plt.title('Daily New COVID-19 Cases Comparison')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.tight_layout()
plt.show()



# Death Rate Calculation
df_filtered['death_rate'] = df_filtered['total_deaths'] / df_filtered['total_cases']

plt.figure(figsize=(12, 6))
for country in countries:
    data = df_filtered[df_filtered['location'] == country]
    plt.plot(data['date'], data['death_rate'], label=country)

plt.title('COVID-19 Death Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Death Rate')
plt.legend()
plt.tight_layout()
plt.show()



# Top Countries by Total Cases
# Last available date for each country
latest_data = df_filtered.groupby('location').last().reset_index()
top_countries = latest_data.sort_values('total_cases', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='total_cases', y='location', data=top_countries)
plt.title('Top 10 Countries by Total COVID-19 Cases')
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.tight_layout()
plt.show()



# Calculate correlations on relevant columns
corr = df_filtered[['total_cases', 'total_deaths', 'new_cases', 'death_rate']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()



# Plot Cumulative Vaccinations Over Time
plt.figure(figsize=(12, 6))

for country in countries:
    data = df_filtered[df_filtered['location'] == country]
    plt.plot(data['date'], data['people_vaccinated'], label=country)

plt.title('Cumulative People Vaccinated Over Time')
plt.xlabel('Date')
plt.ylabel('People Vaccinated')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Compare % Vaccinated Population
# Get latest available data per country
latest_vax = df_filtered.sort_values('date').groupby('location').tail(1)

# Calculate % vaccinated
latest_vax['pct_vaccinated'] = (latest_vax['people_vaccinated'] / latest_vax['population']) * 100

plt.figure(figsize=(8, 5))
sns.barplot(data=latest_vax, x='pct_vaccinated', y='location', palette='Blues_d')

plt.title('Percentage of Population Vaccinated')
plt.xlabel('% of Population Vaccinated')
plt.ylabel('Country')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()



# Pie Charts (Vaccinated vs. Unvaccinated)
for _, row in latest_vax.iterrows():
    vaccinated = row['people_vaccinated']
    unvaccinated = row['population'] - vaccinated
    labels = ['Vaccinated', 'Unvaccinated']
    sizes = [vaccinated, unvaccinated]

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF7043'])
    plt.title(f'Vaccination Share in {row["location"]}')
    plt.tight_layout()
    plt.show()



import plotly.express as px

# Get the latest date per country
latest_df = df.sort_values('date').groupby('location').tail(1)

# Keep only relevant columns
choropleth_df = latest_df[['iso_code', 'location', 'total_cases', 'people_vaccinated', 'population']].copy()

# Calculate metrics
choropleth_df['cases_per_100k'] = (choropleth_df['total_cases'] / choropleth_df['population']) * 100000
choropleth_df['vaccination_rate'] = (choropleth_df['people_vaccinated'] / choropleth_df['population']) * 100

fig = px.choropleth(
    choropleth_df,
    locations='iso_code',
    color='cases_per_100k',
    hover_name='location',
    color_continuous_scale='Reds',
    title='COVID-19 Cases per 100,000 People (Latest Available Data)',
    labels={'cases_per_100k': 'Cases per 100k'}
)
fig.update_geos(showframe=False, showcoastlines=False)
fig.show()


