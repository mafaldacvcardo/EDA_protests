import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8,5)

df = pd.read_csv('data/FARPE-data_1.3.csv', encoding='latin-1', low_memory=False)
print("Dataset loaded successfully! Shape:", df.shape)
df.head()

def clean_col(s):
    return s.astype(str).str.strip(" '\"").replace({'nan': np.nan, 'None': np.nan})

df['countermob'] = clean_col(df['Countermob_string'])
df['issue1'] = clean_col(df['Issue1_string'])
df['event_level'] = clean_col(df['Event_lev_string'])
df['protform'] = clean_col(df['Protform_string'])

print("Cleaned columns created successfully!")
df[['countermob','issue1','event_level','protform']].head()

df_clean = df[df['countermob'].notna() | df['issue1'].notna() | df['protform'].notna()].copy()
print("Rows before:", len(df), "Rows after cleaning:", len(df_clean))

#Hypothesis 1
df_h1 = df_clean[['issue1','countermob']].copy().dropna(subset=['issue1'])
df_h1['has_counter'] = (~df_h1['countermob'].isna()) & (df_h1['countermob'] != 'No counter-mobilization')

issue_stats = df_h1.groupby('issue1')['has_counter'].agg(['sum','count'])
issue_stats['prop_counter'] = issue_stats['sum'] / issue_stats['count']
issue_stats = issue_stats.sort_values('count', ascending=False)
issue_stats.head(10)

min_n = 30
plot_df = issue_stats[issue_stats['count'] >= min_n].sort_values('prop_counter', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=plot_df['prop_counter'], y=plot_df.index)
plt.xlabel('Proportion with counter-mobilization')
plt.ylabel('Issue')
plt.title('Counter-mobilization rate by protest issue (min n=30)')
plt.subplots_adjust(left=0.3)
plt.show()

#Hypothesis 2
# --- Cell 1: map event levels and prepare df_h2 (keep only Local, National, Supranational) ---
def map_level(x):
    # Normalize various strings into three main categories plus others
    if isinstance(x, str):
        if 'Local' in x:
            return 'Local'
        if 'National' in x:
            return 'National'
        if 'Supranational' in x or 'EU' in x:
            return 'Supranational'
        if 'Subnational' in x or 'Regional' in x:
            return 'Regional'
    return x

# make a working copy and drop rows missing event_level
df_h2 = df_clean[['event_level','countermob']].copy().dropna(subset=['event_level'])
df_h2['level_simple'] = df_h2['event_level'].apply(map_level)

# keep only the three levels we want to output for:
keep_levels = ['Local', 'National', 'Supranational']
df_h2 = df_h2[df_h2['level_simple'].isin(keep_levels)].copy()

# sanity check: show value counts of level_simple
print("Counts by level_simple (only Local / National / Supranational):")
print(df_h2['level_simple'].value_counts())

# Creating boolean flag for verbal counter-mobilization
df_h2['is_verbal'] = df_h2['countermob'] == 'Verbal'

# Grouping and computing proportion and counts
table_h2 = df_h2.groupby('level_simple')['is_verbal'].agg(['mean','count']).rename(columns={'mean':'prop_verbal','count':'n'})
table_h2 = table_h2.sort_values('prop_verbal', ascending=False)

# Displaying the table
print(table_h2)

# Plotting
table_plot = table_h2.reset_index().sort_values('prop_verbal', ascending=False)
plt.figure(figsize=(10,8))
plt.pie(table_plot['prop_verbal'],      
        labels=table_plot['level_simple'], 
        autopct='%1.1f%%',                   
        startangle=90,                     
        colors=['#1f77b4', '#d62728', '#2ca02c'])
plt.title('Verbal counter-mobilization by event level (Local / National / Supranational)')
plt.axis('equal')
plt.tight_layout()
plt.show()

#Hypothesis 3

violent_keywords = ['Physical','Violent','Arson','destruction','destruct','Limited destruction','severe']
prot = df_clean[['protform','countermob']].copy()
prot['is_violent'] = prot['protform'].fillna('').apply(lambda x: any(kw.lower() in str(x).lower() for kw in violent_keywords))

prot_group = prot[prot['countermob'].isin(['Contentious','No counter-mobilization'])].copy()
tab = prot_group.groupby('countermob')['is_violent'].agg(['sum','count']).rename(columns={'sum':'n_violent','count':'n_total'})
tab['prop_violent'] = tab['n_violent'] / tab['n_total']
tab

plot_df3 = tab.reset_index()
plt.figure(figsize=(6,4))
sns.barplot(x='prop_violent', y='countermob', data=plot_df3)
plt.xlabel('Proportion with violent protest form')
plt.ylabel('Counter-mobilization type')
plt.title('Violence by counter-mobilization type')
plt.subplots_adjust(left=0.3)
plt.show()

#Hypothesis 4
df_h4 = df[['Year', 'Issue1_string', 'Issue2_string', 'Issue3_string']].copy()

keywords = ['Islam', 'Immigration', 'Identity']

# Function to check if any of the issue columns contain these topics
def related_issue(row):
    text = ' '.join(str(x).lower() for x in row if pd.notnull(x))
    return any(k.lower() in text for k in keywords)

# Applying the function to each row
df_h4['related_issue'] = df_h4.apply(related_issue, axis=1)

# Creating two periods: before 2014 and 2014 & after
df_h4['Period'] = df_h4['Year'].apply(lambda y: 'Before 2014' if y < 2014 else '2014 & After')

# Counting percentage of these issues in each period
table_h4 = df_h4.groupby('Period')['related_issue'].mean().reset_index()
table_h4['related_issue'] = table_h4['related_issue'] * 100  # converting to %

print(table_h4)

# Plotting the comparison
def check_individual_issues(row):
    text = ' '.join(str(x).lower() for x in row if pd.notnull(x))
    results = {}
    for keyword in keywords:
        results[keyword] = keyword.lower() in text
    return pd.Series(results)

df_h4[['Islam', 'Immigration', 'Identity']] = df_h4.apply(check_individual_issues, axis=1)

def assign_period(year):
    if year < 2014:
        return 'Before 2014'
    else:
        return '2014 & After'

df_h4['Period'] = df_h4['Year'].apply(assign_period)

table_h4 = df_h4.groupby('Period')[['Islam', 'Immigration', 'Identity']].mean().reset_index()
table_h4[['Islam', 'Immigration', 'Identity']] = table_h4[['Islam', 'Immigration', 'Identity']] * 100

print(table_h4)

table_h4_melted = table_h4.melt(id_vars='Period', value_vars=['Islam', 'Immigration', 'Identity'], var_name='Issue_Type', value_name='Percentage')

table_h4_pivot = table_h4_melted.pivot(index='Period', columns='Issue_Type', values='Percentage')


table_h4_pivot.plot(kind='bar', stacked=True, color=["#d62728", "#1f77b4", "#2ca02c"], figsize=(8,6))
plt.title('Share of Islam / Immigration / Identity Issues Before vs After 2014')
plt.ylabel('Percentage of Events (%)')
plt.xlabel('Period')
plt.legend(title='Issue Type')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



#Hypothesis 5
df_h5 = df[['Country_string']].copy().dropna()

# Counting how many events per country
country_counts = df_h5['Country_string'].value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

print(country_counts.head(10))

# Plotting top 5 countries
plt.figure(figsize=(8,5))
sns.barplot(x='Count', y='Country', data=country_counts.head(10))
plt.title('Top 10 Countries by Number of Far-Right Mobilizations')
plt.xlabel('Number of Events')
plt.ylabel('Country')
plt.show()

top10_share = country_counts.head(5)['Count'].sum() / country_counts['Count'].sum() * 100
print(f"Top 5 countries represent about {top10_share:.1f}% of all mobilizations.")

#Hypothesis 6
df_h6 = df[['Year', 'Issue1_string', 'Issue2_string', 'Issue3_string']].copy().dropna(how='all')

# Defining welfare keywords
welfare_keywords = ['welfare', 'economic', 'inequality', 'poverty', 'employment', 'austerity']

# Checking if any issue column mentions these
def welfare_issue(row):
    text = ' '.join(str(x).lower() for x in row if pd.notnull(x))
    return any(k in text for k in welfare_keywords)

df_h6['is_welfare'] = df_h6.apply(welfare_issue, axis=1)

# Calculating yearly share of welfare issues
yearly_welfare = df_h6.groupby('Year')['is_welfare'].mean().reset_index()
yearly_welfare['is_welfare'] = yearly_welfare['is_welfare'] * 100  # convert to %

plt.figure(figsize=(8,5))
sns.lineplot(x='Year', y='is_welfare', data=yearly_welfare, marker='o')
plt.title('Share of Welfare/Economic-Related Mobilizations Over Time')
plt.ylabel('Percentage of Events (%)')
plt.xlabel('Year')
plt.axvspan(2012, 2014, color='orange', alpha=0.2, label='Crisis Years (2012–2014)')
plt.legend()
plt.show()

# Printing the year with the highest welfare focus
peak = yearly_welfare.loc[yearly_welfare['is_welfare'].idxmax()]
print(f"Highest welfare issue share in {int(peak['Year'])}: {peak['is_welfare']:.1f}% of events.")

