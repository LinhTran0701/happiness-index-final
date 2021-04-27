import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import streamlit as st

######### lists #########
MENU = ['Preparing and Reading Datasets',
        'Part 1: A general overview',
        'Part 2: Analysis of Happiness index and contributing factors',
        'Part 3: Regression model',
        'Conclusions']

factors = ['Ladder score','Logged GDP per capita','Social support','Healthy life expectancy',
           'Freedom to make life choices','Generosity', 'Perceptions of corruption']

factor_distribution = ['Explained by: Log GDP per capita','Explained by: Social support',
                       'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices',
                       'Explained by: Generosity', 'Explained by: Perceptions of corruption', 'Dystopia + residual']


######### functions #########
def import_data():
    df1 = pd.read_csv('world-happiness-report-2019.csv').sort_values(by=['Ladder score'], ascending=False)
    df2 = pd.read_csv('world-happiness-report-2020.csv').sort_values(by=['Ladder score'], ascending=False)
    return df1, df2

def viz_change(y_value, df):
    fig,ax = plt.subplots(figsize=(10, 12))
    splot = sns.barplot(x='Country', y=y_value, data=df, ax=ax,
                        palette=['C0', 'C0', 'C0', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2'])
    x = df.iloc[5,1]*0.03
    i = 0
    for p in splot.patches:
        if i<5:
            splot.annotate(list(df['Country'])[i], ha = 'center', va = 'bottom', color='white',
                           rotation=90, size=9, xy=(i, -x))
        else:
            splot.annotate(list(df['Country'])[i], ha = 'center', va = 'top', color='white',
                           rotation=90, size=9, xy=(i, x))
        i += 1

    for s in ['top','left', 'bottom', 'right']:
        splot.spines[s].set_visible(False)

    plt.xticks([])
    plt.yticks([])
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(y_value)
    st.pyplot(fig)

def viz_country(df, year):
    fig,ax = plt.subplots(figsize=(18,10))
    f = df.sort_values('Ladder score', ascending=False).plot(x='Country name', y=factor_distribution, kind='barh', stacked=True, ax=ax)
    plt.xticks(rotation=0)
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.title(f'Ladder score breakdown for top 10 and bottom 10 countries in {year}')
    plt.ylabel(None)
    f.invert_yaxis()
    l = plt.legend(loc='lower right', frameon=False)
    l.get_frame().set_facecolor('none')
    st.pyplot(fig)

def viz_factors(df, y, f1, f2):
    fig,ax = plt.subplots(figsize=(10,7))
    sns.scatterplot(data=df, x=f1, y='Ladder score', hue='Category',
                    alpha=0.9, size=df[f2],
                    legend=True, sizes=(10, 500), ax=ax)
    plt.axhline(y=y, color='black', linestyle='--', label='World mean Ladder score')
    l = ax.legend(frameon=False,loc="upper center", bbox_to_anchor=(1.2, 0.8), ncol=1)
    l.get_frame().set_facecolor('none')
    plt.title(f'{f1} and {f2} on Ladder score')
    for s in ['top','right']:
        ax.spines[s].set_visible(False)
    st.pyplot(fig)

def countries_excluded(df1, df2):
    countries_2019 = list(df1['Country name'].unique())
    countries_2020 = list(df2['Country name'].unique())
    countries_excluded_1 = [country for country in countries_2019 if country not in countries_2020]
    countries_excluded_2 = [country for country in countries_2020 if country not in countries_2019]
    return countries_excluded_1, countries_excluded_2

def update_data(df, column, old, new):
    for ind,val in df.iterrows():
        if val[column] == old:
            df.iloc[ind,0] = new
    return df

def get_top_bottom(df):
    top_bottom = df.copy()
    top_bottom.drop(top_bottom.index[10:-10],0,inplace=True)
    return top_bottom

def get_world_comparison(df1, df2):
    mean_world_2019 = pd.DataFrame(df1[factors].mean()).reset_index()
    mean_world_2020 = pd.DataFrame(df2[factors].mean()).reset_index()

    world_comparison = pd.merge(mean_world_2019, mean_world_2020, on='index', suffixes=[' 2019', ' 2020'])
    world_comparison.columns = ['Indicator', '2019', '2020']
    return world_comparison

def get_corr_map(df, title):
    fig,ax = plt.subplots(figsize=(10,10))
    factor_corr_2019 = df[factors+factor_distribution].corr()
    sns.heatmap(factor_corr_2019, annot=True, vmin=-1, vmax=1, center=0, annot_kws={"size": 9},
                cmap=sns.diverging_palette(20, 220, n=256), square=True, cbar_kws={"shrink": 0.8})
    plt.title(title)
    st.pyplot(fig)


######### pages #########
def page0(df1, df2):
    st.title('Preparing and Reading Datasets')

    # import data
    st.header('Import datasets')
    st.write('Happiness index report 2019:', df1)
    st.write('Happiness index report 2020:', df2)

    # examine null cells/mismatches
    st.header('Examine null cells/mismatches')
    st.write('Before doing any further analysis, we have to examine if there are any mismatches in country names and fix it. '
             'We would want to see if there are any countries that are not included in either of the datasets as well.')

    st.write('Null cells count of 2019 dataset:', df1.isnull().sum())
    st.write('Null cells count of 2020 dataset:', df2.isnull().sum())

    countries_excluded_1, countries_excluded_2 = countries_excluded(df1, df2)
    st.write('Countries in 2019 not included in 2020 dataset:', countries_excluded_1)
    st.write('Countries in 2020 not included in 2019 dataset:', countries_excluded_2)

    st.write("It is good that there are no null cells in our datasets. However, I found that 'Macedonia' in 2019 dataset "
             "and 'North Macedonia' in 2020 dataset indicate the same country, so it will be appropriate to change "
             "'Macedonia' into 'North Macedonia' in the 2019 dataset to make it easier to do comparisons on the country "
             "level between 2019 and 2020. Afterwards, we have to make sure that the country names between 2 datasets match.")

    # change 'Macedonia' to 'North Macedonia'
    df1 = update_data(df1, 'Country name', 'Macedonia', 'North Macedonia')

    # recheck country name matches
    countries_excluded_1, countries_excluded_2 = countries_excluded(df1, df2)
    st.write('Countries in 2019 not included in 2020 dataset:', countries_excluded_1)
    st.write('Countries in 2020 not included in 2019 dataset:', countries_excluded_2)
    st.write('**Names matched!**')

def page1(df1, df2):
    df1 = update_data(df1, 'Country name', 'Macedonia', 'North Macedonia')
    st.title('Part 1: A general overview')

    ex1 = st.beta_expander('Country overview')
    with ex1:
        c1, c2, c3 = st.beta_columns((4,1,4))
        c1.write(f'Happiest country 2019: **{df1.iloc[0,0]}**')
        c1.write(f'Unhappiest country 2019: **{df1.iloc[-1, 0]}**')
        c3.write(f'Happiest country 2020: **{df2.iloc[0,0]}**')
        c3.write(f'Unhappiest country 2020: **{df2.iloc[-1,0]}**')

        # 2019
        st.subheader('Top 10 and bottom 10 countries 2019')
        top_bottom_2019 = get_top_bottom(df1)
        top2019_count = top_bottom_2019[:10].value_counts('Regional indicator')
        bottom2019_count = top_bottom_2019[-10:].value_counts('Regional indicator')
        c1, c2, c3 = st.beta_columns((4,1,4))
        c1.write('Top 10 in 2019:')
        c1.write(top2019_count)
        c3.write('Bottom 10 in 2019:')
        c3.write(bottom2019_count)

        # 2020
        st.subheader('Top 10 and bottom 10 countries 2020')
        top_bottom_2020 = get_top_bottom(df2)
        top2020_count = top_bottom_2020[:10].value_counts('Regional indicator')
        bottom2020_count = top_bottom_2020[-10:].value_counts('Regional indicator')
        c1, c2, c3 = st.beta_columns((4,1,4))
        c1.write('Top 10 in 2020:')
        c1.write(top2020_count)
        c3.write('Bottom 10 in 2020:')
        c3.write(bottom2020_count)

        st.write("Western Europe are dominantly the happiest countries with Finland holding the first position in the world "
                 "both before and during the pandemic. On the other hand, 7 of 10 least happiest countries are Sub-Saharan "
                 "African countries, while Afghanistan remains the country with lowest Ladder score from 2019 to 2020.")

        st.subheader('Biggeset movers')
        st.write('We will test to see which countries have the most drastic change before and during the pandemic. '
                 'We have to exclude countries not included in either of the datasets before that.')

        countries_excluded_1, countries_excluded_2 = countries_excluded(df1, df2)
        df1_excluded = df1[~df1['Country name'].isin(countries_excluded_1)].sort_values('Country name').iloc[:,[0,2]].reset_index()
        df1_excluded['Ranking 2019'] = df1_excluded['index'] + 1
        df1_excluded = df1_excluded.drop(columns=['index'])

        df2_excluded = df2[~df2['Country name'].isin(countries_excluded_2)].sort_values('Country name').iloc[:,[0,2]].reset_index()
        df2_excluded['Ranking 2020'] = df2_excluded['index'] + 1
        df2_excluded = df2_excluded.drop(columns=['index', 'Country name'])

        change_df = pd.concat([df1_excluded, df2_excluded], axis=1)
        change_df.columns = ['Country', 'Ladder 2019', 'Ranking 2019', 'Ladder 2020', 'Ranking 2020']
        change_df['Happiness change'] = change_df['Ladder 2020'] - change_df['Ladder 2019']
        change_df['Ranking change'] = change_df['Ranking 2019'] - change_df['Ranking 2020']

        ladder_change_df = change_df.sort_values('Happiness change', ascending=False).reset_index()
        ladder_change_df = ladder_change_df.drop(ladder_change_df.index[5:144]).iloc[:,[1,-2]]

        ranking_change_df = change_df.sort_values('Ranking change', ascending=False).reset_index()
        ranking_change_df = ranking_change_df.drop(ranking_change_df.index[5:144]).iloc[:,[1,-1]]

        c1, c2, c3 = st.beta_columns((7,1,7))
        c1.write(ladder_change_df)
        c3.write(ranking_change_df)
        st.write('It was interesting to see that most of the countries in this list have the most significant changes in '
                 'both Ladder score and ranking. Making bar charts helped with visualizing the changes seen in the '
                 'biggest movers in the world.\n\n')

        viz_change('Happiness change', ladder_change_df)
        viz_change('Ranking change', ranking_change_df)

    ex2 = st.beta_expander('Regional overview')
    with ex2:
        # factors by region
        mean_region_2019 = pd.DataFrame(df1.groupby('Regional indicator')[factors].mean())
        mean_region_2019['Year'] = 2019
        mean_region_2020 = pd.DataFrame(df2.groupby('Regional indicator')[factors].mean())
        mean_region_2020['Year'] = 2020
        mean_region = (np.array(mean_region_2019.iloc[:,:-1]) + np.array(mean_region_2020.iloc[:,:-1])) / 2
        mean_region = pd.DataFrame(mean_region).set_index(mean_region_2019.index)
        mean_region.columns = factors
        mean_region['Year'] = 'Mean'
        region_comparison = pd.concat([mean_region_2019, mean_region_2020, mean_region]).pivot(columns='Year')

        # Ladder score
        fig,ax = plt.subplots(figsize=(10,7))
        f = region_comparison.sort_values([(factors[0], 'Mean')], ascending=False)[factors[0]].plot(kind='barh',ax=ax)
        f.set_ylabel(None)
        f.set_title(f'Mean {factors[0]} comparison by Region')
        f.invert_yaxis()
        f.set_xlim(3.5)
        l = f.legend(loc='lower right', frameon=False)
        l.get_frame().set_facecolor('none')
        plt.subplots_adjust(wspace = 0.5, hspace = 0.2)
        st.pyplot(fig)

        # 6 factors
        fig,ax = plt.subplots(3,2,figsize=(20,18))
        n = 1
        for i in range(3):
            for j in range(2):
                f = region_comparison.sort_values([(factors[n], 'Mean')], ascending=False)[factors[n]].plot(kind='barh', ax=ax[i,j])
                f.set_ylabel(None)
                f.set_title(f'Mean {factors[n]} comparison by Region')
                f.invert_yaxis()
                if n != 5:
                    f.set_xlim(region_comparison[(factors[n], 'Mean')].max() * 0.45)
                l = f.legend(loc='lower right', frameon=False)
                l.get_frame().set_facecolor('none')
                plt.subplots_adjust(wspace = 0.5, hspace = 0.2)
                n += 1
        plt.tight_layout()
        st.pyplot(fig)

        st.write("Things to consider:\n"
                 "- There is almost no change in the order of happiness index from 2019 to 2020. However, Latin America "
                 "& Caribbean (LA & C) and Central & Eastern Europe (EU) switched position between 3rd and 4th place as "
                 "illustrated in the chart above. Only Western, Central & Eastern EU, East Asia, Independent States, "
                 "Southeast Asia (SEA) and Sub-Saharan Africa recorded increases in Happiness index. This might have "
                 "something to do with all other variables in the context of a pandemic.\n"
                 "- GDP per capita and life expectancy all increased.\n"
                 "- Social support: LA & C, SEA, South Asia went down.\n"
                 "- Freedom: SEA and North America & ANZ (NA & ANZ) lead but decreased from 19 to 20, South Asia also "
                 "decreased.\n"
                 "- The absence of corruption: South Asia, SEA, and NA & ANZ improved.")

    ex3 = st.beta_expander('World overview')
    with ex3:
        world_comparison = get_world_comparison(df1, df2)
        st.write(world_comparison)

        st.write("Overall, the world seems happier in 2020 than it was in 2019 and all factors seemed more favorable "
                 "despite COVID-19. However, the degrees to which the pandemic hit each region differ from each other, "
                 "and we'll have to examine the change in the criteria included in the happiness score.")

def page2(df1, df2):
    df1 = update_data(df1, 'Country name', 'Macedonia', 'North Macedonia')
    st.title('Analysis of Happiness index and contributing factors')
    top_bottom_2019 = get_top_bottom(df1)
    top_bottom_2020 = get_top_bottom(df2)

    ex1 = st.beta_expander('Breakdown by top and bottom countries')
    with ex1:
        viz_country(top_bottom_2019, '2019')
        viz_country(top_bottom_2020, '2020')
        st.write('By making stacked bar charts of the breakdown scores for top 10 and bottom 10 countries, we see that '
                 'the Happiness score breakdown by 6 factors are somewhat similar for happiest countries, while the '
                 'breakdown scores for unhappiest ones varies.')

    ex2 = st.beta_expander('Happy versus Unhappy countries in 2020')
    with ex2:
        st.write("First, we will categorize countries into 2 classes: happy (above mean Ladder score) and unhappy (below "
                 "mean Ladder score). By using t-tests, we will see if there are significant differences "
                 "between the mean values of each factors among happy and unhappy countries.\n\n"
                 "For each factor (Logged GDP per capita, Social support, Healthy life expectancy, Freedom to make life "
                 "choices, Perceptions of corruption, Generosity): \n"
                 "- **Null hypothesis:** Happy and unhappy countries have the same mean values of the factor; or $H_0: μ_1 = μ_2$ \n"
                 "- **Alternative hypothesis:** Happy and unhappy countries do not have the same mean values of the factor; or $H_1: μ_1 != μ_2$")

        # ladder score
        world_comparison = get_world_comparison(df1, df2)
        world_ladder_mean_2020 = world_comparison.iloc[0,2]

        for x in factors[1:]:
            happy = np.array(df2[df2['Ladder score'] > world_ladder_mean_2020][x])
            unhappy = np.array(df2[df2['Ladder score'] < world_ladder_mean_2020][x])
            t, p = stats.ttest_ind(happy, unhappy)
            if p < 0.05:
                st.write(f'**Reject hypothesis:** There is a significant difference in **{x}** between happy and unhappy countries.')
            else:
                st.write(f'**Fail to reject hypothesis:** There is no significant difference in **{x}** between happy and unhappy countries.')

        st.write("There is no significance difference between Generosity index between happy and unhappy countries, and "
                 "that is why the explained portion by Generosity is often the lowest in the overall Happiness index equation.")

    ex3 = st.beta_expander('Correlation between factors')
    with ex3:
        st.write("**Correlation matrix:** We will bring in all independent variables here and also include the scores, "
                 "explained by those factors, that contributed to the final scores that indicate Happiness index.")
        #2019
        get_corr_map(df1, 'Correlation heatmap 2019')
        #2020
        get_corr_map(df2, 'Correlation heatmap 2020')
        #total
        df_total = pd.concat([df1, df2])
        get_corr_map(df_total, 'Correlation heatmap total dataset')

        st.write("It can be implied from this correlation matrix that the independent variables are highly proportional "
                 "to the score translated to the sum of Happniess index, except for Perceptions of corruption. In other "
                 "words, the higher the raw data, the higher the explained scores (again, except for Perceptions of "
                 "corruption). Therefore, we can look at the correlations between the independent variables to determine "
                 "their direct effects on the final score.\n\n"
                 "As tested earlier, Generosity index among happy and unhappy countries does not have a significant "
                 "difference. The regression plot here also supports the result that Generosity has no clear correlation "
                 "with Ladder score. On the other hands, the first four factors were positively correlated with Ladder score.")

        st.write("**Correlated pairs of variables on Ladder score in 2020:** GDP per capita and life expectancy are the "
                 "most representative variables that were mostly strongly correlated (as seen in correlation matrices), "
                 "so we would check the relationship between these two variables. Social support and Freedom relationship "
                 "was fairly strong, and the correlation between Corruption perception and Freedom was negative and the "
                 "strongest between the latter and other factors.\n\n"
                 "We will produce visualizations to better understand these relationships on Happiness index, with "
                 "Western EU, and Sub-Saharan Africa highlighted because we found that most of the top 10 countries are "
                 "in Western Europe, while most of the bottom 10 are in Sub-Saharan Africa.")

        df_copy = df2.copy()
        conditions = [(df_copy['Regional indicator'] == 'Western Europe'), (df_copy['Regional indicator'] == 'Sub-Saharan Africa'),
                      (df_copy['Regional indicator'] != 'Western Europe') & (df_copy['Regional indicator'] != 'Sub-Saharan Africa')]
        values = ['Western Europe', 'Sub-Saharan Africa', 'Others']
        df_copy['Category'] = np.select(conditions, values)

        # Life expectancy and GDP per capita on Ladder score
        viz_factors(df_copy, world_ladder_mean_2020, 'Healthy life expectancy', 'Logged GDP per capita')
        # Social support and Freedom on Ladder score
        viz_factors(df_copy, world_ladder_mean_2020,'Social support', 'Freedom to make life choices')
        # Corruption perceptions and Freedom on Ladder score
        viz_factors(df_copy, world_ladder_mean_2020, 'Perceptions of corruption', 'Freedom to make life choices')

        st.write("Indeed, all countries from Western EU are above the average line. This is because they have very high "
                 "GDP per capita, life expectancy, good social support and freedom to make life choices, although "
                 "corruption perceptions vary and only play a minor role in determining a high Ladder score.")

        # African outlier
        st.write("On the other spectrum, all Sub-Saharan African countries are below the line, having low GDP per capita, "
                 "life expectancy, good social support and freedom to make life choices, with the exception of the "
                 "country below:")
        st.write(df2[(df2['Ladder score'] > world_ladder_mean_2020) & (df2['Regional indicator'] == 'Sub-Saharan Africa')])

        # corruption outlier
        st.write("There was 1 Sub-Saharan African outlier (low Perceptions of corruption but low Ladder score) on the "
                 "Corruption-Freedom chart (low Perceptions of corruption but low Ladder score), but we can keep it in "
                 "our analysis since the correlation between this factor and Happiness index might not change significantly "
                 "by the exclusion of the latter.")
        st.write(df2[(df2['Perceptions of corruption'] < 0.2) & (df2['Regional indicator'] == 'Sub-Saharan Africa')])

    ex4 = st.beta_expander("Which factors affect the change in happiness on a regional scale?")
    with ex4:
        st.write('To answer this question, we can first examine the score breakdown in terms of the 6 factors.')

        explained_factors = factor_distribution[:-1]

        mean_explained_2019 = pd.DataFrame(df1.groupby('Regional indicator')[explained_factors].mean())
        mean_explained_2019['Year'] = 2019
        mean_explained_2020 = pd.DataFrame(df2.groupby('Regional indicator')[explained_factors].mean())
        mean_explained_2020['Year'] = 2020

        mean_explained = (np.array(mean_explained_2019.iloc[:,:-1]) + np.array(mean_explained_2020.iloc[:,:-1])) / 2
        mean_explained = pd.DataFrame(mean_explained).set_index(mean_explained_2019.index)
        mean_explained.columns = factor_distribution[:-1]
        mean_explained['Year'] = 'Mean'

        explained_comparison = pd.concat([mean_explained_2019, mean_explained_2020, mean_explained]).pivot(columns='Year')

        fig,ax = plt.subplots(3,2,figsize=(20,17))
        n=0
        for i in range(3):
            for j in range(2):
                f = explained_comparison.sort_values([(explained_factors[n], 'Mean')], ascending=False)[explained_factors[n]].plot(kind='barh', ax=ax[i,j])
                f.set_ylabel(None)
                f.set_title(f'{explained_factors[n]} comparison by Region')
                f.invert_yaxis()
                plt.subplots_adjust(wspace = 0.5, hspace = 0.3)
                l = f.legend(loc='lower right', frameon=False)
                l.get_frame().set_facecolor('none')
                n += 1
        plt.tight_layout()
        st.pyplot(fig)

        st.write("Let's recall: North America & ANZ, Latin America & the Caribbean, and South Asia are the three regions "
                 "recording decreases in Happiness index. Specifically for each region, there are several unfavorable "
                 "variables that might drag the numbers down:\n\n"
                 "- North America & ANZ: freedom, corruption perception, generosity\n"
                 "- Latin America & Caribbean: social support, corruption\n"
                 "- South Asia: social support, freedom")

        regions = ['North America and ANZ','Latin America and Caribbean','South Asia']

        for region in regions:
            n = 0
            print(region)
            for x in factor_distribution[:-1]:
                factor_2019 = np.array(df1[df1['Regional indicator']==region][x])
                factor_2020 = np.array(df2[df2['Regional indicator']==region][x])

                t, p = stats.ttest_ind(factor_2019, factor_2020)
                if p < 0.05:
                    print(f'There is a significant difference in {x} between 2019 and 2020.')
                    n += 1
                else:
                    continue
            if n == 0:
                print('None')

        st.write("We can draw a conclusion that from 2019 to 2020:"
                 "- In North America & ANZ: Generosity has the greatest effect in the decrease in Happiness index."
                 "- In Latin America & Caribbean: Social support has the greatest effect in the decrease in Happiness index."
                 "- In South Asia: There is no factor that drastically drag its Happiness index down.\n\n"
                 "In the context of the COVID-19 pandemic, it is easily understood that social support, generosity and "
                 "possibly freedom are viewed worsened by these regions where the rate of infection was accelerating on "
                 "a daily basis in 2020.")

def page3(df1, df2):
    df1 = update_data(df1, 'Country name', 'Macedonia', 'North Macedonia')
    st.title("Regression model")
    st.write("The datasets calculated the happiness index based on the standardized scores translated from the 6 "
             "independent variables. To simplify it, we can build a regression model that directly calculates the "
             "Happiness index based on the raw data. We'll keep Generosity in the equation since for some countries, "
             "this index composes of a considerable share in the overall Happiness index.\n\n"
             "Training the model with 2019 and 2020 data would help get more accurate results.")

    ex1 = st.beta_expander('Splitting datasets into 2 portions')
    with ex1:
        st.write('I randomly selected approximately 80% of the dataset to train, 20% to test, so we ended up having 241 '
                 'data for training and 61 for testing.')
        X = pd.concat([df1,df2])[factors[1:]].reset_index(drop=True)
        y = pd.concat([df1,df2])['Ladder score'].reset_index(drop=True)

        # choose approximately 80% dataset to train, 20% to test
        rows_train = np.random.choice(X.index, int(len(X)*0.8), False)
        training = X.index.isin(rows_train)

        X_train = X[training]
        X_test = X[~training]
        y_train = y[training]
        y_test = y[~training]

        st.write('Number of independent variable values for training:', len(X_train))
        st.write('Number of independent variable values for testing:', len(X_test))
        st.write('Number of dependent variable values for training:', len(y_train))
        st.write('Number of dependent variable values for testing:',  len(y_test))

    ex2 = st.beta_expander('Building and fitting a regression model')
    with ex2:
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)

        # intercept and coefficients
        st.write('**Intercept:**', float(regr.intercept_))
        st.write('**Coefficients:**', pd.Series(regr.coef_, index=X.columns))

        # mean squared error
        st.write('**Mean squared error:** %.2f'% mean_squared_error(y_test, y_pred))

        # coefficient of determination: 1 is perfect prediction
        st.write('**Coefficient of determination:** %.2f' % r2_score(y_test, y_pred))

        st.write(' ')
        # compare actual and predicted Ladder scores
        y_comparison = pd.concat([y_test, pd.Series(y_pred, index=y_test.index)], axis=1)
        y_comparison.columns = ['Actual Ladder', 'Predicted Ladder']

        st.write(y_comparison)
        st.write("The coefficients above align with the correlation matrix in Part 2, that is, only the perceptions of "
                 "corruption with a negative correlation with happiness has a negative coefficient.\n\n"
                 "The reason why the predicted Ladder score is not similar to the actual score is because the original "
                 "equation accounts for residual errors (represented as Residual + Dystopian index). The built model had "
                 "not accounted for other confounding factors or residual errors, so the model was the closest I could "
                 "predict with just six factors.")

def page4():
    st.title('Conclusions')

    st.header("**1**\n\n")
    st.write("The world is happier despite COVID-19.")

    st.header("**2**\n\n")
    st.write("GPD per capita, life expectancy, social support and freedom to make life choices are strongly correlated "
             "with each other and with happiness, so it is safe to assume that by just looking at a minimum of two of "
             "these factors, we can guess which country is unhappy or unhappy.")

    st.header("**3**\n\n")
    st.write("European countries are the happiest thanks to their high living standards, while African countries suffer "
             "the most due to low living standards and corruption.")

    st.header("**4**\n\n")
    c1, c2, c3 = st.beta_columns((4,1,4))
    c1.write("<h1 style='text-align: center; color: #1f77b4;'>Finland</h1>", unsafe_allow_html=True)
    c1.write("<p style='text-align: center; color: black;'>Happiest country in the world<br>in 2019 and 2020</h1>",
             unsafe_allow_html=True)
    c3.write("<h1 style='text-align: center; color: #ff7f0e;'>Afghanistan</h1>", unsafe_allow_html=True)
    c3.write("<p style='text-align: center; color: black;'>Unhappiest country in the world<br>in 2019 and 2020</h1>",
             unsafe_allow_html=True)

    st.header("**5**\n\n")
    st.write("In the context of the COVID-19 pandemic, social support, generosity and possibly freedom are"
             "viewed more negatively by some regions where the rate of infection was accelerating on a"
             "daily basis in 2020.")

    st.header("**6**\n\n")
    st.write("The regression model only accounts for the existing factors and has not yet considered any "
             "other confounding determinants that decide whether a country is happier than another.")


def main():
    st.sidebar.title('MA 346 Final Project Spring 2021')
    st.sidebar.write('Analysis on the World Happiness Index of 2019 and 2020\n\n'
                     'Linh Tran - 04/26/2021\n\n'
                     'MA 346 Data Science\n\n'
                     'Professor  Chow\n\n'
                     'Bentley University\n\n'
                     '**Dashboard link:** https://final-project-happiness-index.herokuapp.com/')
    df1, df2 = import_data()
    select = st.sidebar.selectbox('Choose Section', MENU)

    st.sidebar.write("The Ladder scores, or happiness indices, are estimated by the extent to which each of six factors – "
             "economic production, social support, life expectancy, freedom, absence of corruption, and generosity – "
             "contribute to making life evaluations higher in each country than they are in Dystopia. This hypothetical "
             "country has values equal to the world’s lowest national averages for each of the six factors and is set as "
             "a benchmark for any countries in the world. Dystopia has happiness scores of 1.97 in 2019 and 2.43 in 2020, "
             "which have no impact on the total score reported for each country, but they do explain why some countries "
             "rank higher than others do.")

    if select == MENU[0]:
        page0(df1, df2)
    elif select == MENU[1]:
        page1(df1, df2)
    elif select == MENU[2]:
        page2(df1, df2)
    elif select == MENU[3]:
        page3(df1, df2)
    else:
        page4()


######### main program #########
main()
